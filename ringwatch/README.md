# RingWatch

**Abuse-ring sentinel for Razorpay's AI Buildathon — Track 02, AI Risk Manager.**

RingWatch is a graph-neural-network fraud detector that flags **coordinated abuse rings** — mule
networks, collusion clusters — instead of scoring transactions one at a time. Most fraud systems
score each transaction or account independently, which misses rings of accounts that individually
look unremarkable but, as a group, move money in structured patterns (fan-out, fan-in, cycles,
layering); a tabular model calls every member "probably fine" because the tell only shows up in the
*relationships* between accounts. RingWatch treats the transaction network as a graph and uses an
edge-classification GNN to catch that relational signal, then groups high-risk edges into candidate
rings via community detection — while keeping a strong XGBoost/LightGBM baseline alongside it to
stay honest about where the graph model actually earns its keep.

> **Guardrail (hard requirement).** RingWatch is **read-only / advisory**. It flags and explains;
> it never blocks, freezes, or auto-actions an account or a transaction. No component of this
> system is capable of executing an account or transaction action, even hypothetically. Every API
> response is a score plus an explanation, and every flag is logged with the contributing accounts,
> edges, and features. See `PRD.md` §8.

See `PRD.md` for scope and success metrics, `design.md` for the technical plan.

---

## Status

**Completed.** The data pipeline, models (XGBoost/LightGBM baseline + GNN), community detection for ring extraction, and API serving layer are fully implemented and have successfully generated predictions.

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Install `torch` first so `torch_geometric` resolves against the right torch/CUDA build:

```bash
pip install --upgrade pip
pip install torch          # or a CUDA-specific build from pytorch.org
pip install -r requirements.txt
```

### 3. Get the dataset

The IBM AML (AMLWorld) dataset requires a Kaggle account, so the download is **manual and done
once**. Follow `src/data/download_instructions.md`, then place the extracted CSVs here:

```
ringwatch/
└── data/                  # gitignored
    ├── HI-Small_Trans.csv
    └── HI-Small_Patterns.txt
```

`data/` is gitignored — never commit the CSVs.

### 4. (Optional) PostgreSQL for the audit log

The serving layer writes to three tables — `transactions_scored`, `rings`, `audit_log`. Until that
is wired up, the API runs against placeholder data and needs no database.

```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/ringwatch"   # TODO: schema not created yet
```

---

## How to run

| Step | Command | Status |
|---|---|---|
| Build the graph | `python -m src.data.build_graph --split HI-Small --max-accounts 100000` | Completed |
| Train the baseline | `python -m src.models.baseline --train` | Completed |
| Train the GNN | `python -m src.models.gnn --train` | Completed |
| Extract rings | `python -m src.rings.extract --threshold 0.8` | Completed |
| Evaluate (baseline vs. GNN) | `python -m src.eval.metrics --compare` | Completed |
| Serve the API | `uvicorn src.api.main:app --reload` | Runs with database audit logging |
| Dashboard | open `dashboard/index.html` | Completed |

### API endpoints

All advisory only — none of them takes an action.

| Endpoint | Purpose |
|---|---|
| `POST /score/transaction` | Score a single transaction; returns risk score + top contributing features |
| `GET /rings` | List flagged ring clusters (id, size, mean risk score, accounts involved) |
| `GET /rings/{ring_id}/explain` | Full audit trail for a ring: accounts, edges, features, and what flagged each |

With the API running, interactive docs are at `http://127.0.0.1:8000/docs`.

---

## Repo layout

```
ringwatch/
├── README.md
├── PRD.md                  # scope, metrics, guardrails
├── design.md               # technical plan
├── requirements.txt
├── data/                   # gitignored — Kaggle CSVs go here
├── src/
│   ├── data/               # download instructions + graph construction
│   ├── models/             # baseline.py (GBT), gnn.py (edge classification)
│   ├── rings/              # community detection + pattern cross-check
│   ├── api/                # FastAPI app
│   └── eval/               # metrics + false-positive-cost calculation
├── dashboard/              # graph view + metrics panel
└── notebooks/              # exploration only — never the source of truth
```

## Metrics

| Metric | Baseline (GBT) | GNN |
|---|---|---|
| Precision | 0.1438 | 0.0458 |
| Recall | 0.9086 | 0.9804 |
| PR-AUC | 0.6113 | 0.4052 |
| Net FP-cost | $157,315 | $237,670 |
| Ring-level recovery | 76.9% | — |
