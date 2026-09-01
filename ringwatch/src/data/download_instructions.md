# Dataset download instructions

The dataset behind RingWatch is **not** fetched automatically and is **not** committed to this
repo — it requires a Kaggle account, so it is downloaded manually, once. There is no Kaggle API
key in this repository and none should ever be added to it.

## Primary dataset — IBM "Transactions for Anti-Money Laundering (AML)" (AMLWorld)

- **Kaggle:** https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- **Docs / format spec (PDF):** https://github.com/IBM/AML-Data
- **Reference GNN baselines from the dataset's authors:** https://github.com/IBM/Multi-GNN
- **Paper:** Altman et al., 2023 — *Realistic Synthetic Financial Transactions for AML Models*
  (https://arxiv.org/abs/2306.16424)

### Which split to grab

Grab **one Small split** — either works, pick one and stay with it:

| Split | Accounts | Transactions | Window | Notes |
|---|---|---|---|---|
| **HI-Small** (recommended) | ~515K | ~5.1M | 10 days | Higher illicit ratio → more positive edges to learn from |
| LI-Small | ~706K | ~6.9M | 10 days | Lower illicit ratio; harder imbalance |

Do **not** download the Medium or Large splits — they are multi-million-edge and out of scope for a
3-day build (`PRD.md` §5). Even the Small split gets subsampled further to ~50–150K accounts by
`src/data/build_graph.py` (`design.md` §3).

### Files you need from the split

Two files per split:

1. `HI-Small_Trans.csv` — the transaction ledger. One row per transaction, with the
   **edge-level** `Is Laundering` label. This label lives on the transaction, not the account,
   which is why RingWatch is framed as edge classification (`design.md` §2).
2. `HI-Small_Patterns.txt` — the labeled laundering **pattern groups**, naming the typology
   (fan-out, fan-in, cycle, scatter-gather, ...) for each group of accounts. This is the ring-level
   ground truth used to score community detection in `src/rings/extract.py`.

### Steps

1. Sign in to Kaggle and open the dataset link above.
2. Accept the dataset terms, then **Download** (the full archive is large — you can select
   individual files from the Data tab to pull only the HI-Small pair).
3. Extract the two files into `data/` at the repo root:

   ```
   ringwatch/
   └── data/
       ├── HI-Small_Trans.csv
       └── HI-Small_Patterns.txt
   ```

4. `data/` is gitignored. Keep it that way — do not commit the CSVs.
5. Sanity-check the download before building the graph:

   ```bash
   head -3 data/HI-Small_Trans.csv
   wc -l data/HI-Small_Trans.csv
   head -20 data/HI-Small_Patterns.txt
   ```

## Secondary / comparison benchmark — Elliptic (optional)

- **Kaggle:** https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- 203,769 transaction *nodes*, 234,355 edges, 49 timesteps, ~2% labeled illicit.
- Worth citing in the architecture write-up even if never trained on.
- **Gotcha:** labels are per-transaction-*node* and the graph is documented as heterophilous, so a
  plain GCN can underperform a Random Forest here. That is a known property of Elliptic, not a bug —
  do not burn hours debugging it (`design.md` §2).

## Stretch / fallback options

- **AMLSim** — generate your own network with known typologies: https://github.com/IBM/AMLSim
- **Pre-generated AMLSim sample:** https://www.kaggle.com/datasets/anshankul/ibm-amlsim-example-dataset
- **PaySim** (simpler, no explicit ring structure — fallback only): https://www.kaggle.com/ealaxi/paysim1
