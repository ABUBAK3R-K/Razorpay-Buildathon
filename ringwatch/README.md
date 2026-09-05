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

The audit trail works out of the box — it falls back to SQLite at `data/audit_log.db`, and to a
JSONL file if even that is unavailable. Point it at PostgreSQL only if you want the scores and ring
explanations in a shared database:

```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/ringwatch"
```

Every score and ring explanation is written to the audit trail **before** it is returned. Check
which backend is live with `curl http://127.0.0.1:8000/health`.

---

## How to run

| Step | Command | Status |
|---|---|---|
| Build the graph | `python -m src.data.build_graph --split HI-Small --max-accounts 100000` | Completed |
| Train the baseline | `python -m src.models.baseline --train` | Completed |
| Train the GNN | `python -m src.models.gnn --train` | Completed |
| Extract rings | `python -m src.rings.extract --threshold 0.8` | Completed |
| Evaluate (baseline vs. GNN) | `python -m src.eval.metrics --compare` | Completed |
| Serve the API | `uvicorn src.api.main:app --reload` | Runs with SQLite audit logging |
| Triage console | open `http://127.0.0.1:8000/dashboard/` | Completed |
| Landing page | open `docs/index.html` (static, needs no API) | Completed |

### API endpoints

All advisory only — none of them takes an action.

| Endpoint | Purpose |
|---|---|
| `POST /score/transaction` | Score a single transaction; returns risk score + top contributing features |
| `GET /rings` | List flagged ring clusters (id, size, mean risk score, accounts involved) |
| `GET /rings/{ring_id}/explain` | Full audit trail for a ring: accounts, edges, features, and what flagged each |
| `GET /rings/{ring_id}/graph` | Nodes and edges for the console's graph view |
| `GET /metrics` | Baseline-vs-GNN evaluation results and ring recovery |
| `GET /health` | Liveness, loaded artifacts, and the advisory-only contract |

With the API running, interactive docs are at `http://127.0.0.1:8000/docs`.

---

## Two front ends

| | What it is | Where |
|---|---|---|
| **Triage console** | The operational tool. Ring queue, transaction graph, audit record. Reads the API live. | `dashboard/` — served at `/dashboard/` |
| **Landing page** | The pitch. Explains the problem, the pipeline, and the results. Fully static: one real ring is baked in, so it needs no API. | `docs/` |

To publish the landing page on GitHub Pages, set **Settings → Pages → Deploy from a branch**,
branch `main`, folder `/ (root)`. It then serves at
`https://<user>.github.io/Razorpay-Buildathon/ringwatch/docs/`.

---

## Deployment

**The landing page deploys anywhere; the API does not deploy from a clone alone.**

`docs/index.html` is a single self-contained file with one real ring baked in. It needs no API, no
build step and no server — GitHub Pages, Netlify, or a double-click all work.

The API is the constrained half, for one reason: **`data/` is gitignored, so a fresh clone has no
artifacts.** `graph.pkl` (85 MB), `baseline.pkl`, `gnn.pt`, `rings.json` and the two `.npz` score
files total ~91 MB and are produced by the pipeline, not committed. A clone starts cleanly — every
load is wrapped and failures are reported in `/health` under `load_errors` — but `/rings` will be
empty until those files are present. Any host must either run the pipeline (which needs the Kaggle
dataset) or receive `data/` out of band.

### Serving does not need torch

The serving path imports NumPy, pandas, XGBoost, scikit-learn, FastAPI and SQLAlchemy — and nothing
else. `graph.pkl` is pure NumPy/pandas, `rings.json` is JSON, and the per-edge scores are `.npz`.
`torch` and `torch_geometric` are needed *only* to load `gnn.pt` for scoring a brand-new
transaction; the GNN's already-computed scores are served from `scores_gnn.npz` without it.

Verified with torch, PyG and lightgbm blocked at import: `/metrics`, `/rings`,
`/rings/{id}/explain`, `/rings/{id}/graph`, `/dashboard/` and `POST /score/transaction` all return
200, the console renders identically, and `/health` reports `gnn_loaded: false` with the reason.

```bash
pip install -r requirements-serve.txt
uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

That drops the single biggest install risk — the torch/PyG pair, which is multiple GB and has to be
resolved against a matching CUDA build.

### Constraints to plan around

| Constraint | Detail |
|---|---|
| Artifacts not in git | ~91 MB under `data/`, gitignored. Ship them separately or run the pipeline. |
| Memory | ~450 MB RSS once `graph.pkl` is expanded (492,163 edges x 31 features). A 512 MB tier is too tight; give it 1 GB. |
| Cold start | The graph unpickles at startup, so first boot takes seconds, not milliseconds. Avoid scale-to-zero. |
| Pickle compatibility | Artifacts were written with NumPy 2.x / XGBoost 3.x. NumPy 1.x **cannot** read them. The pins in both requirements files enforce this. |
| Dataset | IBM AML (AMLWorld) needs a Kaggle account, so the pipeline is not reproducible unattended. |
| Console on a static host | `dashboard/index.html` calls the API on its own origin. Served from GitHub Pages with no API behind it, it shows its "API is not answering" screen by design. Deploy it with the API, or point a reverse proxy at both. |

---

## Repo layout

```
ringwatch/
├── README.md
├── PRD.md                  # scope, metrics, guardrails
├── design.md               # technical plan
├── requirements.txt        # full stack (training + serving)
├── requirements-serve.txt  # API only - no torch/PyG
├── data/                   # gitignored — Kaggle CSVs go here
├── src/
│   ├── data/               # download instructions + graph construction
│   ├── models/             # baseline.py (GBT), gnn.py (edge classification)
│   ├── rings/              # community detection + pattern cross-check
│   ├── api/                # FastAPI app
│   └── eval/               # metrics + false-positive-cost calculation
├── dashboard/              # triage console (served at /dashboard/)
├── docs/                   # landing page (static, GitHub Pages)
└── notebooks/              # exploration only — never the source of truth
```

## Metrics

Held-out, time-based split. PR-AUC is threshold-free; precision, recall and net cost are read at
each model's own **cost-optimal** threshold, not at 0.5.

| Metric | Baseline (XGBoost) | GNN (GINE) |
|---|---|---|
| PR-AUC | **0.6113** | 0.4052 |
| Recall | 0.9086 | **0.9804** |
| Precision | **0.1438** | 0.0458 |
| Net cost | **$157,315** | $237,670 |

Cost model is illustrative and is **not** Razorpay figures: $5 per false positive (one manual
review), $500 per missed laundering transaction, against 2,036 positives in the split. Flagging
nothing at all would cost $1,018,000.

**Ring recovery.** Community detection over the high-risk subgraph (threshold 0.80, Louvain)
extracted 663 candidate rings and recovered **180 of 234** ground-truth pattern groups (76.9%) at
0.30 minimum overlap. By shape: fan-in 88%, gather-scatter 80%, fan-out 77%, random 71%, cycle 70%.
538 candidates matched no ground-truth group — the precision cost of extracting at that threshold.

**The honest finding.** Neither model wins outright. The tabular baseline is the better
transaction scorer on both PR-AUC and cost; the graph stage is what turns a scored edge set into
ring-level structure an analyst can open and read. Ring extraction runs on whichever scored edge
set it is given — in the shipped artifact that is the baseline scorer's output
(`rings.json` → `source_model`).
