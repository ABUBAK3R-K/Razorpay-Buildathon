# 🛡️ RingWatch — Abuse-Ring Sentinel

> **Razorpay AI Buildathon — Track 02: AI Risk Manager**  
> *Graph-native Anti-Money Laundering (AML) & Circular Collusion Detection Engine using Heterophily-Resistant Edge-featured GNNs, Community Detection, and Explainable Audit Trails.*

[![Track](https://img.shields.io/badge/Track-02%20AI%20Risk%20Manager-blue.svg)](https://razorpay.com)
[![Direction](https://img.shields.io/badge/Direction-Abuse--Ring%20Sentinel-orange.svg)]()
[![Guardrail](https://img.shields.io/badge/Guardrail-Strictly%20Defense--Only%20(Advisory)-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## 📌 Executive Summary

Most legacy fraud detection and risk engines score transactions or accounts in isolation. This paradigm systematically misses **coordinated financial crime rings** (mule networks, fan-out/fan-in layering, and circular collusion cycles) because individual accounts keep their activity under suspicious thresholds. The fraud signal lives in the **topology of relationships**, not in isolated node metrics.

**RingWatch** treats the financial transaction network as a directed multigraph:
1. **Relational Scoring:** Uses an edge-featured Graph Neural Network (**GINE**) with label-balanced sampling to detect coordinated laundering signals across accounts.
2. **Ring Extraction:** Applies **Louvain community detection** over high-risk subgraphs to isolate candidate collusion rings and classifies their structural topology (Fan-Out, Fan-In, Cycle, Gather-Scatter).
3. **Honest Baseline Benchmarking:** Benchmarks directly against an **XGBoost** tabular model on a time-based held-out split, evaluating PR-AUC and false-positive operational costs ($/FP vs $/FN).
4. **Strictly Defense-Only (Advisory):** Conforms 100% to Track 02 guardrails. RingWatch is strictly read-only; it flags and explains with complete auditability, never executing autonomous account freezes or transaction blocks.

---

## 🏗️ Architecture & Pipeline

```
  Kaggle / IBM AML Synthetic Multi-Agent Dataset (HI-Small)
                            │
                            ▼
     Graph Construction (Accounts = Nodes, Transactions = Edges)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      Tabular Baseline (GBT)       Edge-GNN (GINE + Edge Features)
      - Node/Edge Aggregations     - Label-balanced Mini-batching
      - PR-AUC: 0.6113             - Recall: 0.9804
              │                           │
              └─────────────┬─────────────┘
                            ▼
                High-Risk Flagged Subgraph
                            │
                            ▼
          Community Detection & Typology Extraction
          - Louvain Clustering
          - Ring-Level Recovery: 76.9% of Ground-Truth Patterns
                            │
                            ▼
          FastAPI Advisory Layer + Audit Storage
          - POST /score/transaction
          - GET  /rings
          - GET  /rings/{id}/explain
                            │
                            ▼
           Minimal Light Beige Dashboard (vis-network)
```

---

## 📊 Evaluation & Honest Cost Analysis

Evaluated on a **time-based held-out test split** from the IBM AMLWorld benchmark:

| Metric | Baseline (GBT) | Edge-GNN (GINE) | Key Takeaway |
|---|---|---|---|
| **Precision** | **0.1438** | 0.0458 | Tabular model is more selective at transaction level |
| **Recall** | 0.9086 | **0.9804** | GNN catches nearly all coordinated laundering transactions |
| **PR-AUC** | **0.6113** | 0.4052 | Baseline leads in isolated edge ranking |
| **Net FP-Cost** | **$157,315** | $237,670 | Economic model ($25/manual review vs $500/missed fraud) |
| **Ring Recovery** | — | **76.9%** | **40 / 52** ground-truth rings discovered via community clustering |

> **The Architectural Insight:**  
> A tabular model is cost-optimal for isolated transaction scoring, while the Graph Neural Network acts as a sentinel that identifies the macro structure of abuse rings that point-in-time scoring misses.

---

## 🧭 Repository Structure

```
Razorpay-Buildathon/
├── PRD.md                       # Product Requirements Document & Track Alignment
├── design.md                    # Detailed Technical Design & Math Specs
├── README.md                    # Main Project Overview
└── ringwatch/
    ├── requirements.txt         # Project Dependencies
    ├── dashboard/
    │   └── index.html           # Minimal Beige Risk & Ring Visualization UI
    ├── src/
    │   ├── data/
    │   │   ├── build_graph.py   # Multigraph construction & time-based split
    │   │   └── download_instructions.md
    │   ├── models/
    │   │   ├── baseline.py      # XGBoost Baseline training & scoring
    │   │   └── gnn.py           # GINE edge classification with PC-GNN sampling
    │   ├── rings/
    │   │   └── extract.py       # Louvain clustering & pattern group matching
    │   ├── eval/
    │   │   └── metrics.py       # PR-AUC, recovery rate & economic FP cost
    │   └── api/
    │       ├── main.py          # FastAPI application
    │       ├── scoring.py       # Live inference & feature attribution
    │       └── audit.py         # Persistent audit logger (PostgreSQL / SQLite / JSONL)
    └── notebooks/               # Research & exploration notebooks
```

---

## 🚀 Quickstart & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ABUBAK3R-K/Razorpay-Buildathon.git
cd Razorpay-Buildathon/ringwatch

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API & Serving Layer
```bash
uvicorn src.api.main:app --reload --port 8000
```
Interactive OpenAPI documentation will be accessible at `http://127.0.0.1:8000/docs`.

### 3. Open the Dashboard
Open `ringwatch/dashboard/index.html` directly in your browser or serve it via Live Server.

---

## 🛡️ Guardrails (Track 02 Compliance)

- **Read-Only / Advisory Only:** Every API endpoint returns risk scores, feature explanations, and cluster topologies. The system is architecturally decoupled from payment settlement and cannot execute blocking or account freezes.
- **Auditability:** Every flag persists contributing counterparty accounts, edge weights, and graph features to an append-only audit trail.

---

## 📄 License
MIT License. Built for the Razorpay AI Buildathon 2026.
