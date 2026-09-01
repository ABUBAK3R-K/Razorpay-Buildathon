"""Runtime scoring for the RingWatch API.

Holds the artifacts the API serves from - the built graph, the trained models, and the extracted
rings - and reconstructs a feature vector for an arbitrary incoming transaction so that a single
transaction can be scored with the same model that was evaluated offline.

Everything here is read-only. The scorer returns a number and the features behind it; it has no
code path that mutates an account or a transaction.

Loading is lazy and failure-tolerant on purpose: the API must still start and serve its documented
shape when nothing has been trained yet, so a missing artifact degrades to placeholder mode rather
than crashing the service.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class RingWatchScorer:
    """Loads artifacts once and answers scoring/explanation queries against them.

    Attributes:
        bundle: The built :class:`~src.data.build_graph.GraphBundle`, or ``None``.
        baseline: Fitted GBT booster, or ``None``.
        gnn: Trained :class:`~src.models.gnn.EdgeGNN`, or ``None``.
        rings: Ring dicts loaded from ``data/rings.json``.
        scores: Per-edge risk scores for the whole graph, from the best available model.
        model_name: Which model the scores and predictions come from.
    """

    def __init__(self, data_dir: Path = DATA_DIR, prefer: str = "gnn") -> None:
        """Load whatever artifacts exist.

        Args:
            data_dir: Directory holding ``graph.pkl``, ``baseline.pkl``, ``gnn.pt``, ``rings.json``.
            prefer: Which model to serve from when both are available - ``"gnn"`` or ``"baseline"``.
        """
        self.data_dir = data_dir
        self.prefer = prefer
        self.bundle: Any = None
        self.baseline: Any = None
        self.gnn: Any = None
        self.rings: list[dict[str, Any]] = []
        self.ring_meta: dict[str, Any] = {}
        self.scores: np.ndarray | None = None
        self.model_name = "placeholder-v0 (nothing trained yet)"
        self.load_errors: dict[str, str] = {}
        self._stats: tuple[np.ndarray, np.ndarray] | None = None
        self._account_to_ring: dict[str, str] = {}
        self._pair_counts: dict[tuple[int, int], int] = {}

        self._load_graph()
        self._load_models()
        self._load_rings()

    # ---------------------------------------------------------------- loading

    def _load_graph(self) -> None:
        try:
            from src.data.build_graph import load_graph

            self.bundle = load_graph(self.data_dir / "graph.pkl")
        except Exception as exc:
            self.load_errors["graph"] = f"{type(exc).__name__}: {exc}"

    def _load_models(self) -> None:
        try:
            from src.models.baseline import load as load_baseline

            self.baseline = load_baseline(self.data_dir / "baseline.pkl")
            self.model_name = "baseline-xgboost"
        except Exception as exc:
            self.load_errors["baseline"] = f"{type(exc).__name__}: {exc}"

        if self.prefer == "gnn":
            try:
                from src.models.gnn import load as load_gnn

                self.gnn = load_gnn(self.data_dir / "gnn.pt")
                self.model_name = "gnn-gine"
            except Exception as exc:
                self.load_errors["gnn"] = f"{type(exc).__name__}: {exc}"

        for name in ("scores_gnn.npz", "scores_baseline.npz"):
            path = self.data_dir / name
            if path.exists():
                self.scores = np.load(path)["scores"]
                break

    def _load_rings(self) -> None:
        import json

        path = self.data_dir / "rings.json"
        if not path.exists():
            self.load_errors["rings"] = f"not found: {path}"
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.rings = payload.get("rings", [])
        self.ring_meta = {
            "threshold": payload.get("threshold"),
            "method": payload.get("method"),
            "recovery": payload.get("recovery", {}),
            "source_model": payload.get("source_model", "unknown"),
        }
        for ring in self.rings:
            for account in ring.get("accounts", []):
                self._account_to_ring.setdefault(account, ring["ring_id"])

    @property
    def ready(self) -> bool:
        """True when a real graph and at least one trained model are loaded."""
        return self.bundle is not None and (self.baseline is not None or self.gnn is not None)

    # ---------------------------------------------------------------- scoring

    def _node_features(self, account_key: str) -> np.ndarray:
        """Historical node features for an account, or zeros if it has never been seen.

        An unseen account is genuinely uninformative rather than safe - the response says so via
        ``known_accounts``, so a reviewer can tell a real low score from a cold-start one.
        """
        idx = self.bundle.account_index.get(account_key)
        if idx is None:
            return np.zeros(self.bundle.x.shape[1], dtype=np.float32)
        return self.bundle.x[idx]

    def build_edge_features(
        self,
        amount_paid: float,
        amount_received: float,
        payment_currency: str,
        receiving_currency: str,
        payment_format: str,
        timestamp: datetime,
        from_key: str,
        to_key: str,
    ) -> np.ndarray:
        """Reconstruct the edge feature vector for an incoming transaction.

        Built by name from ``bundle.edge_feature_names`` so it stays in lockstep with whatever
        ``src.data.build_graph.compute_edge_features`` produced - including the one-hot currency
        and payment-format columns, whose vocabulary was fixed at build time.

        Args:
            amount_paid: Amount debited from the sender.
            amount_received: Amount credited to the recipient.
            payment_currency: Currency debited.
            receiving_currency: Currency credited.
            payment_format: ACH, Wire, Cheque, Cash, Credit Card, Bitcoin, Reinvestment.
            timestamp: Transaction time.
            from_key: Composite sender account key.
            to_key: Composite recipient account key.

        Returns:
            A ``(n_edge_features,)`` float32 vector aligned with ``bundle.edge_feature_names``.
        """
        names = self.bundle.edge_feature_names
        values: dict[str, float] = {
            "log_amount_paid": float(np.log1p(amount_paid)),
            "log_amount_received": float(np.log1p(amount_received)),
            "amount_delta_ratio": float(amount_received / (amount_paid + 1.0)),
            "is_cross_currency": float(receiving_currency != payment_currency),
            "is_self_loop": float(from_key == to_key),
            "hour_of_day": float(timestamp.hour),
            "day_index": 0.0,
            "is_night": float(timestamp.hour < 6 or timestamp.hour >= 22),
            "pair_repeat_count": float(self._pair_repeat_count(from_key, to_key)),
        }
        fmt_key = f"fmt_{payment_format.replace(' ', '_').lower()}"
        cur_key = f"cur_{payment_currency.replace(' ', '_').lower()}"
        values[fmt_key] = 1.0
        values[cur_key] = 1.0
        return np.array([values.get(n, 0.0) for n in names], dtype=np.float32)

    def _pair_repeat_count(self, from_key: str, to_key: str) -> int:
        """How many times this exact account pair already transacted in the historical graph."""
        src = self.bundle.account_index.get(from_key)
        dst = self.bundle.account_index.get(to_key)
        if src is None or dst is None:
            return 1
        if not self._pair_counts:
            pairs, counts = np.unique(self.bundle.edge_index.T, axis=0, return_counts=True)
            self._pair_counts = {
                (int(p[0]), int(p[1])): int(c) for p, c in zip(pairs, counts)
            }
        return self._pair_counts.get((src, dst), 1)

    def score_transaction(
        self,
        transaction_id: str,
        from_bank: str,
        from_account: str,
        to_bank: str,
        to_account: str,
        amount: float,
        currency: str,
        payment_format: str,
        timestamp: datetime | None = None,
        top_k: int = 6,
    ) -> dict[str, Any]:
        """Score one transaction and return the features that drove the score.

        Advisory only. The return value contains a risk score and its evidence, and no field that
        instructs any system to act.

        Args:
            transaction_id: Caller's id for the transaction.
            from_bank: Sender's bank id.
            from_account: Sender's account number.
            to_bank: Recipient's bank id.
            to_account: Recipient's account number.
            amount: Transaction amount.
            currency: Transaction currency.
            payment_format: Payment rail.
            timestamp: Transaction time; defaults to now.
            top_k: How many contributing features to return.

        Returns:
            A dict matching the API's ``ScoreOut`` schema fields.
        """
        ts = timestamp or datetime.now(timezone.utc)
        from_key = f"{from_bank.strip()}:{from_account.strip()}"
        to_key = f"{to_bank.strip()}:{to_account.strip()}"

        if not self.ready:
            return self._placeholder_score(transaction_id, from_key, to_key, ts)

        edge_feat = self.build_edge_features(
            amount, amount, currency, currency, payment_format, ts, from_key, to_key
        )
        x_src = self._node_features(from_key)
        x_dst = self._node_features(to_key)

        from src.models.baseline import feature_names as baseline_feature_names

        names = self.bundle.node_feature_names
        deg, pt = names.index("degree"), names.index("pass_through_ratio")
        pair = np.array(
            [
                x_src[deg] + x_dst[deg],
                abs(x_src[deg] - x_dst[deg]),
                abs(x_src[pt] - x_dst[pt]),
            ],
            dtype=np.float32,
        )
        row = np.concatenate([edge_feat, x_src, x_dst, pair])[None, :]

        score = float(self.baseline.predict_proba(row)[0, 1])
        contributions = self._contributions(row[0], baseline_feature_names(self.bundle), top_k)

        known = [
            k for k in (from_key, to_key) if k in self.bundle.account_index
        ]
        return {
            "transaction_id": transaction_id,
            "risk_score": score,
            "model": self.model_name if self.gnn is None else "baseline-xgboost (single-transaction path)",
            "top_features": contributions,
            "ring_id": self._account_to_ring.get(from_key) or self._account_to_ring.get(to_key),
            "known_accounts": known,
            "from_account": from_key,
            "to_account": to_key,
            "amount": amount,
            "currency": currency,
            "scored_at": ts,
        }

    def _feature_stats(self, sample: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
        """Per-column mean and std of the baseline feature matrix, from a sampled subset.

        Cached after the first call. Sampled rather than computed over all 492K edges because the
        full matrix is only needed to normalize an explanation, not to train anything.
        """
        if getattr(self, "_stats", None) is None:
            from src.models.baseline import build_feature_matrix

            rng = np.random.default_rng(0)
            mask = np.zeros(self.bundle.n_edges, dtype=bool)
            idx = rng.choice(self.bundle.n_edges, size=min(sample, self.bundle.n_edges), replace=False)
            mask[idx] = True
            X, _ = build_feature_matrix(self.bundle, mask)
            self._stats = (X.mean(0), X.std(0) + 1e-6)
        return self._stats

    def _contributions(
        self, row: np.ndarray, names: list[str], top_k: int
    ) -> list[dict[str, Any]]:
        """Rank this transaction's features by importance times how unusual the value is.

        Tree importances are global, not per-prediction, so the contribution shown is
        ``importance * |z-score of the value against the graph's own distribution|``. Normalizing
        matters: without it, features measured in millions of currency units swamp high-importance
        binary features like the payment rail purely because their raw magnitude is larger.

        This is an attribution, not a SHAP value - it says which signals put this transaction in
        front of an analyst, and the audit trail records it as such.
        """
        importances = np.asarray(self.baseline.feature_importances_, dtype=float)
        mean, std = self._feature_stats()
        z = np.abs((row - mean) / std)
        weighted = importances * np.clip(z, 0.0, 10.0)
        order = np.argsort(weighted)[::-1][:top_k]
        return [
            {
                "feature": names[i],
                "value": float(row[i]),
                "contribution": float(weighted[i]),
            }
            for i in order
            if weighted[i] > 0
        ]

    def _placeholder_score(
        self, transaction_id: str, from_key: str, to_key: str, ts: datetime
    ) -> dict[str, Any]:
        """Documented-shape response for when no model has been trained yet."""
        return {
            "transaction_id": transaction_id,
            "risk_score": 0.0,
            "model": "placeholder-v0 (nothing trained yet)",
            "top_features": [],
            "ring_id": None,
            "known_accounts": [],
            "from_account": from_key,
            "to_account": to_key,
            "amount": 0.0,
            "currency": "USD",
            "scored_at": ts,
        }

    # ------------------------------------------------------------ ring lookup

    def list_rings(self, min_size: int = 3, limit: int = 50, typology: str | None = None) -> list[dict[str, Any]]:
        """Flagged rings, largest risk first.

        Args:
            min_size: Minimum accounts for a cluster to be returned.
            limit: Max rings to return.
            typology: Optional filter on the detected shape.

        Returns:
            Ring summary dicts.
        """
        out = [r for r in self.rings if r.get("size", 0) >= min_size]
        if typology:
            out = [r for r in out if r.get("detected_typology") == typology.upper()]
        return out[:limit]

    def get_ring(self, ring_id: str) -> dict[str, Any] | None:
        """One ring by id, or ``None``."""
        return next((r for r in self.rings if r["ring_id"] == ring_id), None)

    def ring_edges(self, ring_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """The flagged transactions inside a ring, with their scores and amounts.

        Args:
            ring_id: Ring to expand.
            limit: Max transactions to return.

        Returns:
            One dict per transaction, ordered by risk score descending.
        """
        ring = self.get_ring(ring_id)
        if ring is None or self.bundle is None:
            return []
        edge_ids = ring.get("edge_ids", [])[: limit * 4]
        amount_col = self.bundle.edge_feature_names.index("log_amount_paid")
        rows = []
        for eid in edge_ids:
            src = int(self.bundle.edge_index[0, eid])
            dst = int(self.bundle.edge_index[1, eid])
            rows.append(
                {
                    "transaction_id": f"edge_{eid}",
                    "edge_id": int(eid),
                    "from_account": str(self.bundle.accounts[src]),
                    "to_account": str(self.bundle.accounts[dst]),
                    "amount": float(np.expm1(self.bundle.edge_attr[eid, amount_col])),
                    "timestamp": datetime.fromtimestamp(
                        int(self.bundle.edge_time[eid]), tz=timezone.utc
                    ),
                    "risk_score": float(self.scores[eid]) if self.scores is not None else 0.0,
                    "is_laundering_ground_truth": int(self.bundle.y[eid]),
                }
            )
        rows.sort(key=lambda r: r["risk_score"], reverse=True)
        return rows[:limit]

    def status(self) -> dict[str, Any]:
        """What actually loaded - surfaced on ``/health`` so the demo is self-describing."""
        return {
            "graph_loaded": self.bundle is not None,
            "n_accounts": int(self.bundle.n_nodes) if self.bundle is not None else 0,
            "n_transactions": int(self.bundle.n_edges) if self.bundle is not None else 0,
            "baseline_loaded": self.baseline is not None,
            "gnn_loaded": self.gnn is not None,
            "n_rings": len(self.rings),
            "serving_model": self.model_name,
            "load_errors": self.load_errors,
        }
