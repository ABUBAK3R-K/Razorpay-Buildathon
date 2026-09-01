# RingWatch — Kickoff Prompt

Paste this into a coding agent (Claude Code, Cursor, etc.) with `PRD.md` and `design.md` in the same folder/repo so it can read them for context.

---

You are setting up the initial scaffold for **RingWatch**, a 3-day hackathon project for Razorpay's AI Buildathon (Track 02 — AI Risk Manager). Read `PRD.md` and `design.md` in this repo before doing anything — they define the scope, the dataset, the model architecture, the API, and the hard guardrails (the system must be flag/explain-only and must never be able to take an autonomous action on an account or transaction).

For this first session, do ONLY the following. Do not start model training, real data loading, or full feature engineering yet — this session is scaffolding only.

1. Create the repo structure exactly as specified in `design.md` §10:
   `ringwatch/` with `src/data/`, `src/models/`, `src/rings/`, `src/api/`, `src/eval/`, `dashboard/`, `notebooks/`.

2. Write a `requirements.txt` covering: `torch`, `torch_geometric`, `xgboost` (or `lightgbm`), `fastapi`, `uvicorn`, `psycopg2-binary` (or `sqlalchemy`), `networkx`, `python-louvain`, `pandas`, `scikit-learn`.

3. Write a `README.md` with:
   - A one-paragraph pitch, pulled from `PRD.md` §1–2.
   - Setup instructions: create a virtual environment, install requirements, where the downloaded Kaggle CSVs should go (`data/`).
   - A "how to run" section with TODO placeholders for scripts that don't exist yet.

4. Write `src/data/download_instructions.md` (documentation, not code — this dataset needs a Kaggle account to fetch) with the exact dataset link from `design.md` §2 and which split to grab (HI-Small or LI-Small).

5. Stub out the following files with function signatures and docstrings only — no implementation yet:
   - `src/data/build_graph.py` — load CSVs → construct the account/transaction graph → time-based train/test split → subsample to ~50–150K accounts.
   - `src/models/baseline.py` — train/eval an XGBoost or LightGBM baseline on aggregated features.
   - `src/models/gnn.py` — train/eval an edge-classification GNN with a label-balanced sampler.
   - `src/rings/extract.py` — community detection over a flagged subgraph, plus cross-check against labeled pattern files.
   - `src/api/main.py` — a FastAPI app with the three endpoints from `design.md` §8 (`POST /score/transaction`, `GET /rings`, `GET /rings/{ring_id}/explain`), returning mock/placeholder data for now so the API shape is real even before the model is trained.
   - `src/eval/metrics.py` — precision/recall/PR-AUC plus the false-positive-cost calculation described in `design.md` §7.

6. Do not fabricate a Kaggle API key or attempt to auto-download the dataset — assume the download will be done manually once, and just print/document the instructions.

7. Add a `.gitignore` that excludes `data/`, virtual environment folders, and any `.env` files.

When you're done, give me a short summary of what was created and the one next command I should run. Stop there — don't move on to actual data loading, feature engineering, or training until I confirm the scaffold looks right.
