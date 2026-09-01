# RingWatch — Product Requirements Document

## 1. One-line pitch
RingWatch is a graph-neural-network fraud detector built for Razorpay's AI Buildathon (**Track 02 — AI Risk Manager**) that flags **coordinated abuse rings** — mule networks, collusion clusters — instead of scoring transactions one at a time.

## 2. Problem statement
Most fraud systems score each transaction or account independently. This misses coordinated fraud: rings of accounts that individually look unremarkable but, as a group, move money in structured patterns (fan-out, fan-in, cycles, layering). A tabular model scores every member of a ring as "probably fine" because no single account is extreme on its own — the tell only shows up in the *relationships* between accounts. RingWatch treats the transaction network as a graph and uses a GNN to catch that relational signal, while keeping a strong tabular baseline honest about where the graph model actually earns its keep.

## 3. Track alignment
- **Track:** 02 — AI Risk Manager
- **Direction:** Abuse-ring sentinel
- **Why this track:** the direction is explicitly about detecting coordinated abuse across accounts — a graph problem by definition, and the closest match to a GNN-recsys background applied to a different graph.

## 4. Goals (in scope for the 3-day build)
- **G1** — Construct a transaction graph from a public synthetic AML dataset that ships ground-truth ring/pattern labels.
- **G2** — Train and evaluate a tabular baseline (XGBoost/LightGBM) as an honest comparison point.
- **G3** — Train a GNN (edge classification, imbalance-aware) that measurably beats the baseline on ring-structured fraud specifically.
- **G4** — Extract candidate "rings" — not just flagged individual transactions — via community detection over high-risk subgraphs.
- **G5** — Serve scores and ring explanations through a FastAPI API with a full audit trail.
- **G6** — Build a minimal dashboard visualizing flagged rings on the transaction graph.
- **G7** — Ship the required deliverables: public repo, 5-minute pitch video, architecture write-up.

## 5. Non-goals (explicitly out of scope)
- Real-time streaming ingestion — batch/offline scoring is enough for a demo.
- Any autonomous action (freezing accounts, blocking transactions). RingWatch only flags and explains — this is a hard requirement, not a time-saving cut (see §8).
- Training at full dataset scale (the multi-million-edge Medium/Large splits) — use a Small split, further subsampled.
- Multi-institution / federated detection.
- Production-grade auth, rate limiting, or deployment infra — a local or Colab-hosted demo is sufficient.

## 6. Users & what they're judging
The primary "user" this weekend is the Razorpay Buildathon evaluator. Based on the track's stated bar, they'll be checking for:
- Measured precision/recall on a held-out test set, including an honest false-positive-cost estimate — not just an accuracy number.
- Evidence the system is strictly defensive (flag/explain only, never act).
- A clear audit trail per decision: why was this flagged, which accounts/edges contributed.

## 7. Success metrics
| Metric | Target / what to report |
|---|---|
| Precision / Recall / PR-AUC (edge-level) | On a time-based held-out split; report both baseline and GNN, side by side |
| False-positive cost estimate | Assign an illustrative cost per false positive (manual review) and per missed fraud (loss); report net cost, not just accuracy |
| Ring-level recovery | % of ground-truth laundering pattern groups (from the dataset's pattern files) recovered by the community-detection step |
| Latency | Time to score a batch and return ring explanations (informal — not a hard SLA for a demo) |

## 8. Guardrails (hard requirements, not nice-to-haves)
- RingWatch is **read-only / advisory**. It never blocks, freezes, or auto-actions anything — every API response is a score plus an explanation, nothing else.
- Every flag is logged with the contributing accounts, edges, and features that drove it.
- No component of the system is capable of executing a transaction or account action, even hypothetically — this is what keeps the project unambiguously in the "defense" category the track requires.

## 9. Milestones (3-day timeline)
| Day | Focus | Key outputs |
|---|---|---|
| **Day 1** | Data + graph + baseline | Dataset downloaded and subsampled; account/transaction graph built with node & edge features; XGBoost/LightGBM baseline trained with time-based train/test split; baseline precision/recall/PR-AUC recorded |
| **Day 2** | GNN + ring extraction + evaluation | Edge-classification GNN trained with a label-balanced sampler; GNN vs. baseline comparison; community detection over flagged subgraph to produce candidate rings; ring-level recovery measured against labeled pattern files; false-positive cost estimate computed |
| **Day 3** | Serving + dashboard + deliverables | FastAPI endpoints (`/score`, `/rings`, `/rings/{id}/explain`) with audit logging to PostgreSQL; lightweight graph dashboard; architecture doc finalized; repo cleaned up with README; 5-minute pitch video recorded |

See `design.md` for the detailed technical plan behind each day.

## 10. Deliverables checklist
- [ ] Public GitHub repo with README and one-command setup
- [ ] Architecture document (`design.md`, this pair)
- [ ] 5-minute pitch video
- [ ] Live or recorded dashboard demo
- [ ] Metrics table: baseline vs. GNN — precision, recall, PR-AUC, FP-cost, ring-recovery

## 11. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Dataset too large to train in a weekend | Use the Small split (HI-Small/LI-Small) and further subsample to ~50–150K accounts; scaling up is a stretch goal only |
| GNN doesn't obviously beat the tabular baseline | Expected on some benchmarks (documented heterophily issue on Elliptic) — the pitch narrative is "graph wins specifically on ring-structured cases," not "graph wins everywhere"; report both honestly |
| Class imbalance tanks recall | Use a label-balanced minibatch sampler (PC-GNN-style) rather than training on raw ~1–2% positive rate |
| Running out of time before deliverables | Treat the video + repo cleanup as fixed, non-negotiable blocks on Day 3 afternoon — cut model scope before cutting deliverable polish |
| Scope creep into "cool but unnecessary" features | Anything not in §4 Goals is explicitly out of scope until all G1–G7 are done |

## 12. Stretch goals (only if Day 3 finishes early)
- Subgraph-level classification instead of edge-level (à la Elliptic2) — classify a *whole cluster* as a ring rather than scoring edges one at a time.
- Temporal modeling (EvolveGCN-style) since laundering behavior shifts across the 10-day window.
- CARE-GNN-style camouflage resistance if flagged accounts show adversarial-looking neighbor patterns in your data exploration.
