"""RingWatch serving API (FastAPI).

Three endpoints from ``design.md`` section 8, all **advisory only**:

* ``POST /score/transaction``       - risk score + top contributing features for one transaction
* ``GET  /rings``                   - currently flagged ring clusters
* ``GET  /rings/{ring_id}/explain`` - full audit trail for one ring

Plus three supporting reads the dashboard needs: ``/health``, ``/metrics``, and
``/rings/{ring_id}/graph``.

GUARDRAIL - this is a hard requirement, not a convention (``PRD.md`` section 8). No endpoint here
blocks, freezes, holds, reverses, or otherwise acts on an account or a transaction, and none may
ever be added. Every response is a score plus an explanation. Nothing in this service is capable of
executing an account or transaction action, which is what keeps RingWatch unambiguously defensive.
Every route below is a GET or a scoring POST that mutates nothing but the audit log. Any future
endpoint must be read-only; if a reviewer asks "could this take an action?", the answer must stay
no by construction.

Every score and ring explanation is written to the audit trail (PostgreSQL when ``DATABASE_URL`` is
set, local JSONL otherwise) before it is returned.

Run: ``uvicorn src.api.main:app --reload`` then open http://127.0.0.1:8000/dashboard
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.audit import AuditWriter
from src.api.scoring import DATA_DIR, REPO_ROOT, RingWatchScorer

DASHBOARD_DIR = REPO_ROOT / "dashboard"

STATE: dict[str, Any] = {"scorer": None, "audit": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the graph, models, and rings once at startup."""
    STATE["scorer"] = RingWatchScorer(DATA_DIR)
    STATE["audit"] = AuditWriter()
    status = STATE["scorer"].status()
    print(
        f"RingWatch ready | graph={status['graph_loaded']} "
        f"baseline={status['baseline_loaded']} gnn={status['gnn_loaded']} "
        f"rings={status['n_rings']} | audit={STATE['audit'].backend}"
    )
    if STATE["scorer"].rings:
        STATE["audit"].log_rings(STATE["scorer"].rings[:200])
    yield


app = FastAPI(
    title="RingWatch",
    version="0.2.0",
    description=(
        "Graph-based abuse-ring sentinel for Razorpay AI Buildathon Track 02. "
        "**Advisory only**: RingWatch flags and explains, it never blocks, freezes, or acts on "
        "an account or transaction."
    ),
    lifespan=lifespan,
)

# The dashboard is a static page that calls this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def scorer() -> RingWatchScorer:
    """The loaded scorer, or a 503 if startup has not finished."""
    if STATE["scorer"] is None:
        raise HTTPException(status_code=503, detail="Scorer not loaded yet.")
    return STATE["scorer"]


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------


class TransactionIn(BaseModel):
    """One transaction to score. Mirrors the AMLWorld ledger columns."""

    transaction_id: str = Field(..., examples=["txn_00417"])
    from_bank: str = Field(..., examples=["021174"])
    from_account: str = Field(..., examples=["800737690"])
    to_bank: str = Field(..., examples=["020"])
    to_account: str = Field(..., examples=["80020C5B0"])
    amount: float = Field(..., examples=[48250.0])
    currency: str = Field("US Dollar", examples=["US Dollar", "Euro", "Rupee"])
    payment_format: str = Field(
        "ACH", examples=["ACH", "Wire", "Cheque", "Cash", "Credit Card", "Bitcoin"]
    )
    timestamp: datetime | None = Field(None, description="Defaults to now if omitted.")


class FeatureContribution(BaseModel):
    """One feature's contribution to a score, for the audit trail."""

    feature: str
    value: float
    contribution: float = Field(..., description="Importance-weighted contribution to the score.")


class ScoreOut(BaseModel):
    """Advisory score for one transaction. Contains no action and no instruction to act."""

    transaction_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    model: str
    top_features: list[FeatureContribution]
    ring_id: str | None = Field(None, description="Ring the accounts belong to, if any.")
    known_accounts: list[str] = Field(
        default_factory=list,
        description="Which endpoints have transaction history in the graph. An unknown account "
        "means the score is a cold start, not a clean record.",
    )
    from_account: str
    to_account: str
    scored_at: datetime
    advisory_only: Literal[True] = True


class RingSummary(BaseModel):
    """One flagged ring, as listed by ``GET /rings``."""

    ring_id: str
    size: int = Field(..., description="Number of accounts in the ring.")
    n_transactions: int
    mean_risk_score: float
    max_risk_score: float
    total_amount: float = 0.0
    detected_typology: str = Field(..., examples=["FAN-OUT", "FAN-IN", "CYCLE", "GATHER-SCATTER"])
    accounts: list[str]
    matched_pattern_group: str | None = Field(
        None, description="Ground-truth pattern group, for offline evaluation only."
    )


class RingList(BaseModel):
    """Response body for ``GET /rings``."""

    rings: list[RingSummary]
    total: int
    threshold: float | None = Field(None, description="Edge risk threshold used for the subgraph.")
    method: str | None = None
    advisory_only: Literal[True] = True


class RingEdge(BaseModel):
    """One flagged transaction inside a ring's audit trail."""

    transaction_id: str
    edge_id: int
    from_account: str
    to_account: str
    amount: float
    timestamp: datetime
    risk_score: float
    is_laundering_ground_truth: int = Field(
        ..., description="Dataset label, shown for demo transparency; not a model input."
    )


class RingAccount(BaseModel):
    """One account inside a ring, with the role the graph structure implies."""

    account: str
    role: str = Field(..., examples=["source", "intermediary", "sink", "peripheral"])


class RingExplanation(BaseModel):
    """Full audit trail for one ring - who, which transactions, and what drove the flag."""

    ring_id: str
    detected_typology: str
    summary: str
    size: int
    n_transactions: int
    mean_risk_score: float
    max_risk_score: float
    total_amount: float
    accounts: list[RingAccount]
    edges: list[RingEdge]
    detection_method: str
    model_version: str
    flagged_at: datetime
    advisory_only: Literal[True] = True
    disclaimer: str = (
        "Advisory output for human review. RingWatch does not block, freeze, or otherwise act on "
        "any account or transaction."
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Liveness check, loaded-artifact status, and a restatement of the advisory-only contract."""
    s = STATE["scorer"]
    audit = STATE["audit"]
    return {
        "status": "ok",
        "version": app.version,
        "mode": "advisory-only",
        "capabilities": ["score", "explain"],
        "cannot": ["block", "freeze", "reverse", "act on any account or transaction"],
        "artifacts": s.status() if s else {},
        "audit": audit.status() if audit else {},
    }


@app.get("/metrics", tags=["meta"])
def metrics() -> dict[str, Any]:
    """Baseline-vs-GNN evaluation results and ring-level recovery, for the dashboard panel.

    Reads the JSON the training and ring-extraction runs wrote. Returns whatever exists - a
    half-finished pipeline shows the half that is done rather than erroring.
    """
    out: dict[str, Any] = {"models": {}, "rings": {}}
    for name, filename in (("baseline", "metrics_baseline.json"), ("gnn", "metrics_gnn.json")):
        path = DATA_DIR / filename
        if path.exists():
            out["models"][name] = json.loads(path.read_text(encoding="utf-8"))
    s = STATE["scorer"]
    if s is not None:
        out["rings"] = s.ring_meta
        out["graph"] = {
            "n_accounts": s.status()["n_accounts"],
            "n_transactions": s.status()["n_transactions"],
        }
    return out


@app.post("/score/transaction", response_model=ScoreOut, tags=["scoring"])
def score_transaction(txn: TransactionIn) -> ScoreOut:
    """Score a single transaction and return the features that drove the score.

    Advisory only - the response carries a score and an explanation, never an action. The score and
    its evidence are written to the audit trail before the response is returned.
    """
    s = scorer()
    result = s.score_transaction(
        transaction_id=txn.transaction_id,
        from_bank=txn.from_bank,
        from_account=txn.from_account,
        to_bank=txn.to_bank,
        to_account=txn.to_account,
        amount=txn.amount,
        currency=txn.currency,
        payment_format=txn.payment_format,
        timestamp=txn.timestamp,
    )
    out = ScoreOut(**{k: v for k, v in result.items() if k in ScoreOut.model_fields})
    STATE["audit"].log_score({**result, "scored_at": result["scored_at"].isoformat()})
    return out


@app.get("/rings", response_model=RingList, tags=["rings"])
def list_rings(
    min_size: int = Query(3, ge=2, description="Minimum accounts for a cluster to count as a ring."),
    limit: int = Query(50, ge=1, le=500),
    typology: str | None = Query(None, description="Filter by detected shape, e.g. FAN-OUT."),
) -> RingList:
    """List currently flagged ring clusters, highest mean risk score first."""
    s = scorer()
    rings = s.list_rings(min_size=min_size, limit=limit, typology=typology)
    return RingList(
        rings=[
            RingSummary(**{k: v for k, v in r.items() if k in RingSummary.model_fields})
            for r in rings
        ],
        total=len(rings),
        threshold=s.ring_meta.get("threshold"),
        method=s.ring_meta.get("method"),
    )


@app.get("/rings/{ring_id}/explain", response_model=RingExplanation, tags=["rings"])
def explain_ring(ring_id: str, limit: int = Query(200, ge=1, le=1000)) -> RingExplanation:
    """Return the full audit trail for one ring.

    Contributing accounts with the role the graph structure implies, the flagged transactions
    between them, and the score that flagged each - everything a human reviewer needs to agree or
    disagree with the flag.
    """
    s = scorer()
    ring = s.get_ring(ring_id)
    if ring is None:
        raise HTTPException(status_code=404, detail=f"Unknown ring_id: {ring_id}")

    roles = ring.get("role_by_account", {})
    edges = s.ring_edges(ring_id, limit=limit)
    typology = ring.get("detected_typology", "RANDOM")
    explanation = RingExplanation(
        ring_id=ring_id,
        detected_typology=typology,
        summary=(
            f"{ring['size']} accounts moving funds in a {typology} pattern across "
            f"{ring['n_transactions']} flagged transactions "
            f"(mean risk {ring['mean_risk_score']:.2f}); surfaced for analyst review."
        ),
        size=ring["size"],
        n_transactions=ring["n_transactions"],
        mean_risk_score=ring["mean_risk_score"],
        max_risk_score=ring["max_risk_score"],
        total_amount=ring.get("total_amount", 0.0),
        accounts=[
            RingAccount(account=a, role=roles.get(a, "peripheral")) for a in ring["accounts"]
        ],
        edges=[RingEdge(**e) for e in edges],
        # Name the model the rings were actually extracted from, not whichever model happens to
        # be loaded - an audit trail that misattributes its own evidence is worse than none.
        detection_method=(
            f"{s.ring_meta.get('source_model', 'unknown')} + "
            f"{s.ring_meta.get('method', 'louvain')}"
        ),
        model_version=app.version,
        flagged_at=_now(),
    )
    STATE["audit"].log_event(
        "rings.explain",
        {
            "ring_id": ring_id,
            "size": ring["size"],
            "accounts": ring["accounts"],
            "edge_ids": ring.get("edge_ids", [])[:200],
            "typology": typology,
            "mean_risk_score": ring["mean_risk_score"],
        },
        subject_id=ring_id,
    )
    return explanation


@app.get("/rings/{ring_id}/graph", tags=["rings"])
def ring_graph(ring_id: str, limit: int = Query(300, ge=1, le=2000)) -> dict[str, Any]:
    """Nodes and edges for the dashboard's force-directed view of one ring."""
    s = scorer()
    ring = s.get_ring(ring_id)
    if ring is None:
        raise HTTPException(status_code=404, detail=f"Unknown ring_id: {ring_id}")

    roles = ring.get("role_by_account", {})
    edges = s.ring_edges(ring_id, limit=limit)
    return {
        "ring_id": ring_id,
        "detected_typology": ring.get("detected_typology"),
        "nodes": [
            {"id": a, "label": a.split(":")[-1], "role": roles.get(a, "peripheral")}
            for a in ring["accounts"]
        ],
        "edges": [
            {
                "from": e["from_account"],
                "to": e["to_account"],
                "risk": e["risk_score"],
                "amount": e["amount"],
                "truth": e["is_laundering_ground_truth"],
            }
            for e in edges
        ],
    }


if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")


@app.get("/", include_in_schema=False)
def root() -> Any:
    """Send the browser to the dashboard, or to the API docs if it is not built."""
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"docs": "/docs", "health": "/health"}
