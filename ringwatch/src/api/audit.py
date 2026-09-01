"""Audit trail for RingWatch.

``PRD.md`` section 8 makes this non-optional: *every* flag is logged with the contributing
accounts, edges, and features that drove it. An advisory system whose advice cannot be traced back
to its evidence is not auditable, and auditability is most of what Track 02 is asking for.

Three tables, per ``design.md`` section 8:

* ``transactions_scored`` - one row per scoring call: the transaction, its score, the model, and
  the contributing features.
* ``rings`` - one row per flagged ring cluster: members, size, typology, score.
* ``audit_log`` - the append-only event stream: what was asked, what was answered, when.

Storage is PostgreSQL when ``DATABASE_URL`` is set, and a local JSONL file otherwise, so the demo
runs end-to-end with no database. The writer never raises into a request: an audit backend that is
down degrades to the fallback rather than taking the API with it.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSONL = REPO_ROOT / "data" / "audit_log.jsonl"

_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS transactions_scored (
    id              BIGSERIAL PRIMARY KEY,
    transaction_id  TEXT        NOT NULL,
    from_account    TEXT        NOT NULL,
    to_account      TEXT        NOT NULL,
    amount          DOUBLE PRECISION,
    currency        TEXT,
    risk_score      DOUBLE PRECISION NOT NULL,
    model           TEXT        NOT NULL,
    ring_id         TEXT,
    top_features    TEXT,
    scored_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS rings (
    ring_id             TEXT PRIMARY KEY,
    size                INTEGER NOT NULL,
    n_transactions      INTEGER NOT NULL,
    mean_risk_score     DOUBLE PRECISION NOT NULL,
    max_risk_score      DOUBLE PRECISION,
    detected_typology   TEXT,
    accounts            TEXT,
    flagged_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    event       TEXT        NOT NULL,
    subject_id  TEXT,
    payload     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS transactions_scored (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  TEXT        NOT NULL,
    from_account    TEXT        NOT NULL,
    to_account      TEXT        NOT NULL,
    amount          REAL,
    currency        TEXT,
    risk_score      REAL NOT NULL,
    model           TEXT        NOT NULL,
    ring_id         TEXT,
    top_features    TEXT,
    scored_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS rings (
    ring_id             TEXT PRIMARY KEY,
    size                INTEGER NOT NULL,
    n_transactions      INTEGER NOT NULL,
    mean_risk_score     REAL NOT NULL,
    max_risk_score      REAL,
    detected_typology   TEXT,
    accounts            TEXT,
    flagged_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event       TEXT        NOT NULL,
    subject_id  TEXT,
    payload     TEXT        NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


class AuditWriter:
    """Writes advisory decisions to PostgreSQL, or to JSONL when no database is configured.

    Attributes:
        backend: ``"postgresql"`` or ``"jsonl"`` - surfaced on ``/health`` so a reviewer can see
            which one the running demo used.
    """

    def __init__(self, database_url: str | None = None, jsonl_path: Path = DEFAULT_JSONL) -> None:
        """Connect to PostgreSQL if possible, else fall back to a local JSONL file.

        Args:
            database_url: SQLAlchemy URL. Defaults to the ``DATABASE_URL`` environment variable.
            jsonl_path: Fallback file, created on first write.
        """
        self.jsonl_path = jsonl_path
        # Default to a local SQLite database so we get real database behavior without Postgres.
        default_db = f"sqlite:///{REPO_ROOT / 'data' / 'audit_log.db'}"
        self.database_url = database_url or os.environ.get("DATABASE_URL") or default_db
        self._lock = threading.Lock()
        self._engine: Any = None
        self.backend = "jsonl"
        self.error: str | None = None

        if self.database_url:
            try:
                from sqlalchemy import create_engine, text

                # SQLite needs a different connection argument for multithreading
                connect_args = {"check_same_thread": False} if "sqlite" in self.database_url else {}
                self._engine = create_engine(self.database_url, pool_pre_ping=True, future=True, connect_args=connect_args)
                
                schema = _SCHEMA_SQLITE if "sqlite" in self.database_url else _SCHEMA_PG
                
                with self._engine.begin() as conn:
                    for statement in filter(None, (s.strip() for s in schema.split(";"))):
                        conn.execute(text(statement))
                self.backend = "sqlite" if "sqlite" in self.database_url else "postgresql"
            except Exception as exc:  # pragma: no cover - depends on a live database
                self.error = f"{type(exc).__name__}: {exc}"
                self._engine = None

        if self.backend == "jsonl":
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _jsonl(self, record: dict[str, Any]) -> None:
        """Append one record to the fallback file."""
        with self._lock, self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def log_event(self, event: str, payload: dict[str, Any], subject_id: str | None = None) -> None:
        """Append one advisory decision to the audit trail.

        Args:
            event: Event type, e.g. ``"score.transaction"`` or ``"rings.explain"``.
            payload: Contributing accounts, edges, features, and the score that drove the flag.
            subject_id: The transaction or ring the event is about.
        """
        record = {
            "event": event,
            "subject_id": subject_id,
            "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._engine is not None:
            try:
                from sqlalchemy import text

                with self._engine.begin() as conn:
                    conn.execute(
                        text(
                            "INSERT INTO audit_log (event, subject_id, payload) "
                            "VALUES (:event, :subject_id, :payload)"
                        ),
                        {
                            "event": event,
                            "subject_id": subject_id,
                            "payload": json.dumps(payload, default=str),
                        },
                    )
                return
            except Exception as exc:  # pragma: no cover - degrade, never take the API down
                self.error = f"{type(exc).__name__}: {exc}"
        self._jsonl(record)

    def log_score(self, score: dict[str, Any]) -> None:
        """Persist one transaction score to ``transactions_scored`` (or the fallback).

        Args:
            score: A serialized ``ScoreOut`` payload.
        """
        if self._engine is not None:
            try:
                from sqlalchemy import text

                with self._engine.begin() as conn:
                    conn.execute(
                        text(
                            "INSERT INTO transactions_scored "
                            "(transaction_id, from_account, to_account, amount, currency, "
                            " risk_score, model, ring_id, top_features) "
                            "VALUES (:transaction_id, :from_account, :to_account, :amount, "
                            " :currency, :risk_score, :model, :ring_id, :top_features)"
                        ),
                        {
                            "transaction_id": score.get("transaction_id"),
                            "from_account": score.get("from_account"),
                            "to_account": score.get("to_account"),
                            "amount": score.get("amount"),
                            "currency": score.get("currency"),
                            "risk_score": score.get("risk_score"),
                            "model": score.get("model"),
                            "ring_id": score.get("ring_id"),
                            "top_features": json.dumps(score.get("top_features"), default=str),
                        },
                    )
            except Exception as exc:  # pragma: no cover
                self.error = f"{type(exc).__name__}: {exc}"
                self._jsonl({"event": "score.transaction", "payload": score})
        else:
            self._jsonl({"event": "score.transaction", "payload": score})
        self.log_event("score.transaction", score, score.get("transaction_id"))

    def log_rings(self, rings: list[dict[str, Any]]) -> None:
        """Persist the current ring set to the ``rings`` table (or the fallback).

        Args:
            rings: Serialized ring summaries.
        """
        if self._engine is None:
            self._jsonl({"event": "rings.upsert", "payload": {"n_rings": len(rings)}})
            return
        try:  # pragma: no cover - depends on a live database
            from sqlalchemy import text

            with self._engine.begin() as conn:
                for ring in rings:
                    conn.execute(
                        text(
                            "INSERT INTO rings (ring_id, size, n_transactions, mean_risk_score, "
                            " max_risk_score, detected_typology, accounts) "
                            "VALUES (:ring_id, :size, :n_transactions, :mean_risk_score, "
                            " :max_risk_score, :detected_typology, :accounts) "
                            "ON CONFLICT (ring_id) DO UPDATE SET "
                            " size = EXCLUDED.size, "
                            " n_transactions = EXCLUDED.n_transactions, "
                            " mean_risk_score = EXCLUDED.mean_risk_score"
                        ) if "postgresql" in self.database_url else text(
                            "INSERT INTO rings (ring_id, size, n_transactions, mean_risk_score, "
                            " max_risk_score, detected_typology, accounts) "
                            "VALUES (:ring_id, :size, :n_transactions, :mean_risk_score, "
                            " :max_risk_score, :detected_typology, :accounts) "
                            "ON CONFLICT(ring_id) DO UPDATE SET "
                            " size = excluded.size, "
                            " n_transactions = excluded.n_transactions, "
                            " mean_risk_score = excluded.mean_risk_score"
                        ),
                        {
                            "ring_id": ring.get("ring_id"),
                            "size": ring.get("size"),
                            "n_transactions": ring.get("n_transactions"),
                            "mean_risk_score": ring.get("mean_risk_score"),
                            "max_risk_score": ring.get("max_risk_score"),
                            "detected_typology": ring.get("detected_typology"),
                            "accounts": json.dumps(ring.get("accounts", []), default=str),
                        },
                    )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, Any]:
        """Backend, destination, and last error - reported on ``/health``."""
        return {
            "backend": self.backend,
            "destination": self.database_url if self._engine is not None else str(self.jsonl_path),
            "last_error": self.error,
        }
