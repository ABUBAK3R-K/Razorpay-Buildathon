# RingWatch — Technical Design

Companion doc to `PRD.md`. This is the "how" behind that "what."

## 1. System overview

```
 Kaggle CSVs
     │
     ▼
 Graph construction (accounts=nodes, transactions=edges)
     │
     ├──────────────┐
     ▼              ▼
 Baseline (GBT)   GNN (edge classification)
     │              │
     └──────┬───────┘
            ▼
   Flagged-edge subgraph
            │
            ▼
   Ring extraction (community detection)
            │
            ▼
   FastAPI serving layer  ──►  PostgreSQL (audit log)
            │
            ▼
      Dashboard (graph view + metrics panel)
```

## 2. Dataset

### Primary: IBM "Transactions for Anti-Money Laundering (AML)" (a.k.a. AMLWorld)
- **Download:** https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- **Docs / PDF spec:** https://github.com/IBM/AML-Data
- **Reference GNN baseline code** (GIN / GAT / PNA / RGCN in PyTorch Geometric, from the dataset's own authors): https://github.com/IBM/Multi-GNN
- **Why this one:** it's generated from a multi-agent virtual-world simulation that tags illicit funds from their origin through every transaction used to launder them, and it ships **labeled pattern files naming the specific laundering typology** (fan-out, fan-in, cycle, scatter-gather, etc.) each group of accounts belongs to. That's the ring ground-truth Elliptic doesn't have.
- **Recommended split for a weekend build:** HI-Small (~515K accounts, ~5.1M transactions, 10 days) or LI-Small (~706K accounts, ~6.9M transactions, 10 days). Even these are large for a 3-day laptop/Colab build — plan to subsample further (§3).
- **Important modeling detail:** `Is Laundering` is an **edge-level** (transaction-level) label, not a node-level one. This shapes the architecture choice in §5 — RingWatch is an edge-classification problem, not node classification.

### Secondary / comparison benchmark: Elliptic
- **Download:** https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- 203,769 nodes (transactions), 234,355 edges, 49 timesteps, ~2% labeled illicit — the standard academic benchmark, worth citing in your architecture write-up even if you don't train on it directly.
- **Known gotcha:** labels here are per-transaction-*node*, and the graph is documented as heterophilous (illicit nodes often connect to licit ones). A plain GCN can underwhelm a Random Forest baseline on this specific dataset — that's a documented property of Elliptic, not a bug in your code, so don't burn hours debugging it if you experiment here.

### Optional / stretch: generate your own labeled rings
- IBM's AMLSim simulator (build your own transaction network with known typologies baked in, if you want more control over ring shapes than the fixed CSVs give you): https://github.com/IBM/AMLSim
- Pre-generated AMLSim sample dataset: https://www.kaggle.com/datasets/anshankul/ibm-amlsim-example-dataset
- PaySim (simpler, mobile-money framing, no explicit ring structure — fallback only if AMLWorld/AMLSim setup eats too much time): https://www.kaggle.com/ealaxi/paysim1

## 3. Data pipeline
1. Download the HI-Small (or LI-Small) CSVs from Kaggle.
2. **Time-based** train/test split — don't shuffle randomly, that leaks future structure into training.
3. Subsample to a workable size: keep every account that appears in at least one labeled laundering pattern, plus a random sample of same-order-of-magnitude non-laundering accounts and their transactions. Target roughly 50–150K accounts total for the weekend build.
4. Construct the graph: **nodes = accounts**, **edges = transactions** (directed, multi-edge — one account pair can have many transactions). Store in PyTorch Geometric's edge-feature format, following the same convention as IBM's own Multi-GNN repo.
5. Node features: transaction count, in/out degree, amount mean/std/max, unique-counterparty count, active time span.
6. Edge features: amount, currency, payment format, timestamp, repeat-count between the same account pair.

## 4. Baseline model
- XGBoost or LightGBM on the aggregated node + edge features — the same style of baseline IBM's own AML paper uses.
- This is not a placeholder step: it's the comparison point your entire pitch narrative rests on ("here's what the graph catches that this doesn't").

## 5. GNN model
- **Task framing:** edge classification (the label lives on the transaction, not the account) — use an edge-featured message-passing architecture rather than a pure node classifier. GIN or PNA with edge features (as in IBM's Multi-GNN repo) is a reasonable starting point; GraphSAGE with edge-conditioned aggregation is a simpler fallback if time is short.
- **Class imbalance:** illicit edges are typically ~1% of the data (IBM AML) or ~2% (Elliptic). Borrow PC-GNN's idea of a label-balanced sampler for minibatch construction instead of training on the raw imbalance — reference implementation: https://github.com/PonderLY/PC-GNN
- **If accounts look adversarially camouflaged** in your data exploration (rings deliberately structured to blend into normal transaction bursts), consider a lightweight version of CARE-GNN's neighbor-similarity filtering rather than its full reinforcement-learning version — a simple similarity threshold is enough for a demo.
- **Framework:** PyTorch + PyTorch Geometric.

## 6. Ring extraction
1. After edge-level scoring, build a subgraph of all edges above a risk threshold.
2. Run connected-components or Louvain community detection on that subgraph to group flagged transactions into candidate rings.
3. Cross-check each candidate ring against the dataset's labeled pattern files to compute ring-level recovery — do your discovered clusters line up with the ground-truth fan-out/cycle/etc. groups?

## 7. Evaluation
- Precision, recall, and PR-AUC on the held-out (later-in-time) split, at the edge level, for **both** baseline and GNN — side by side, not just the GNN's number.
- False-positive cost estimate: assign an illustrative cost per false positive (manual review) and per false negative (missed fraud loss), and report net cost — this is what "honest metrics including false positive cost" in the track's bar is asking for.
- Ring-level recovery: fraction of true pattern groups at least partially recovered by §6.

## 8. Serving API (FastAPI)
| Endpoint | Purpose |
|---|---|
| `POST /score/transaction` | Score a single transaction; returns risk score + top contributing features |
| `GET /rings` | List currently flagged ring clusters (id, size, mean risk score, accounts involved) |
| `GET /rings/{ring_id}/explain` | Full audit trail for a ring: contributing accounts, edges, features, and the score/rule that flagged each |

All responses are advisory only — no endpoint takes an action on an account or transaction (see PRD §8 Guardrails). Every score and audit trail gets written to PostgreSQL across three tables: `transactions_scored`, `rings`, `audit_log`.

## 9. Dashboard
- A lightweight force-directed graph view (vis-network or d3, plain HTML/JS, or a small React page since that's already in your stack) showing:
  - Accounts as nodes, colored by risk score.
  - Transactions as edges, highlighted if flagged.
  - A side panel with the baseline-vs-GNN metrics table and the false-positive-cost estimate.
- Keep this intentionally simple. A working, honest metrics panel beats a polished but decorative UI for this track.

## 10. Repo structure
```
ringwatch/
├── README.md
├── PRD.md
├── design.md
├── requirements.txt
├── data/                    # gitignored — Kaggle CSVs go here
├── src/
│   ├── data/                 # download instructions + preprocessing + graph construction
│   ├── models/
│   │   ├── baseline.py        # XGBoost / LightGBM
│   │   └── gnn.py             # edge-classification GNN
│   ├── rings/                 # community detection + ring cross-check
│   ├── api/                   # FastAPI app
│   └── eval/                  # metrics, false-positive-cost calculation
├── dashboard/                 # HTML/JS or React front-end
└── notebooks/                 # exploration only — never the source of truth
```

## 11. Tech stack mapping
| Layer | Tool |
|---|---|
| Graph + GNN | Python, PyTorch, PyTorch Geometric |
| Baseline | XGBoost / LightGBM |
| Graph ops / community detection | NetworkX, python-louvain |
| API | FastAPI |
| Storage | PostgreSQL |
| Dashboard | HTML/JS (vis-network or d3) or React |

## 12. Key papers behind these choices
- Weber et al., 2019 — *Anti-Money Laundering in Bitcoin* (foundational GCN + Elliptic paper) — https://arxiv.org/abs/1908.02591
- Altman et al., 2023 — *Realistic Synthetic Financial Transactions for AML Models* (introduces the AMLWorld dataset used here) — https://arxiv.org/abs/2306.16424
- Dou et al., 2020 — *CARE-GNN* (camouflage resistance) — https://arxiv.org/abs/2008.08692
- Liu et al., 2021 — *PC-GNN* (class-imbalance sampler) — https://github.com/PonderLY/PC-GNN
- Bellei et al., 2024 — *Elliptic2 / Shape of Money Laundering* (subgraph-level ring classification, relevant to the stretch goal) — https://arxiv.org/abs/2404.19109
