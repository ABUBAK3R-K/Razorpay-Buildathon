"""Graph construction for RingWatch.

Loads the IBM AMLWorld Small-split CSVs from ``data/archive/`` (see
``src/data/download_instructions.md``), subsamples them to a weekend-sized problem, and builds the
directed multi-graph the models train on: **nodes = accounts, edges = transactions**.

Design notes (``design.md`` section 3):

* The ``Is Laundering`` label is **edge-level**, so the split and the sampler both operate on edges.
* The train/test split is **time-based**, never random - a random shuffle leaks future graph
  structure into training.
* Node features are aggregated over the **training window only**, then reused for both splits, so
  no future information reaches a test-time prediction.
* Target size after subsampling is ~50-150K accounts.

Facts about the real HI-Small file that shaped this module:

* The CSV has **two columns literally named "Account"** (positions 2 and 4) - source then
  destination - so columns are read positionally, not by name.
* Account numbers are not quite globally unique (a handful collide across banks), so nodes are
  keyed by the composite ``"<bank>:<account>"``.
* ~26% of rows are ``Reinvestment`` self-loops (account to itself). They carry no relational
  signal but do carry amount signal, so they are kept as edges and flagged via ``is_self_loop``.
* The observed laundering rate is ~0.075% of edges - over 10x more imbalanced than the ~1% quoted
  in ``design.md``. This is why the sampler and the cost model matter more than raw accuracy.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARCHIVE_DIR = DATA_DIR / "archive"

# The ledger's real header is: Timestamp, From Bank, Account, To Bank, Account, Amount Received,
# Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering.
# Two columns share the name "Account", so we read positionally and rename.
TRANS_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

TRANS_DTYPES = {
    "from_bank": "str",
    "from_account": "str",
    "to_bank": "str",
    "to_account": "str",
    "amount_received": "float64",
    "amount_paid": "float64",
    "receiving_currency": "category",
    "payment_currency": "category",
    "payment_format": "category",
    "is_laundering": "int8",
}

_PATTERN_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z\- ]+?)\s*:\s*(.*)$")
_PATTERN_END = re.compile(r"^END LAUNDERING ATTEMPT")


@dataclass
class GraphBundle:
    """Everything downstream code needs from one built graph.

    Nodes are accounts, edges are transactions (directed, multi-edge - one account pair can have
    many transactions). Arrays are plain numpy so this module stays importable without torch;
    :meth:`to_pyg` converts to PyTorch Geometric on demand for ``src.models.gnn``.

    Attributes:
        edge_index: ``(2, n_edges)`` int64 array of ``[source_node, target_node]``.
        edge_attr: ``(n_edges, n_edge_features)`` float32 edge feature matrix.
        edge_feature_names: Column names for ``edge_attr``.
        x: ``(n_nodes, n_node_features)`` float32 node feature matrix.
        node_feature_names: Column names for ``x``.
        y: ``(n_edges,)`` int8 edge-level ``Is Laundering`` labels.
        edge_time: ``(n_edges,)`` int64 unix seconds, used for the time-based split.
        accounts: Node index to composite account key (``"<bank>:<account>"``).
        account_index: Composite account key to node index.
        train_edge_mask: Boolean mask selecting the earlier-in-time training edges.
        test_edge_mask: Boolean mask selecting the later-in-time held-out edges.
        split_timestamp: The cut point separating train from test.
        pattern_groups: Ground-truth laundering pattern groups keyed by group id.
        edge_pattern_group: ``(n_edges,)`` object array naming each edge's ground-truth pattern
            group, or ``None``. Evaluation only - never a model input.
        meta: Provenance - split name, sizes, subsample settings, build time.
    """

    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_feature_names: list[str]
    x: np.ndarray
    node_feature_names: list[str]
    y: np.ndarray
    edge_time: np.ndarray
    accounts: np.ndarray
    account_index: dict[str, int]
    train_edge_mask: np.ndarray
    test_edge_mask: np.ndarray
    split_timestamp: pd.Timestamp
    pattern_groups: dict[str, dict[str, Any]]
    edge_pattern_group: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        """Number of accounts in the graph."""
        return int(self.x.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of transactions in the graph."""
        return int(self.edge_index.shape[1])

    def summary(self) -> str:
        """One-block human-readable description, printed at the end of a build."""
        n_pos = int(self.y.sum())
        tr, te = int(self.train_edge_mask.sum()), int(self.test_edge_mask.sum())
        return (
            f"GraphBundle: {self.n_nodes:,} accounts | {self.n_edges:,} transactions\n"
            f"  laundering edges : {n_pos:,} ({n_pos / max(self.n_edges, 1):.4%})\n"
            f"  train / test     : {tr:,} / {te:,} (split at {self.split_timestamp})\n"
            f"  train positives  : {int(self.y[self.train_edge_mask].sum()):,}\n"
            f"  test positives   : {int(self.y[self.test_edge_mask].sum()):,}\n"
            f"  node features    : {len(self.node_feature_names)} | "
            f"edge features: {len(self.edge_feature_names)}\n"
            f"  pattern groups   : {len(self.pattern_groups):,}"
        )

    def to_pyg(self) -> Any:
        """Convert to a PyTorch Geometric ``Data`` object for ``src.models.gnn``.

        Imported lazily so that graph building, the baseline, and evaluation all work in an
        environment without torch installed.

        Returns:
            ``torch_geometric.data.Data`` with ``x``, ``edge_index``, ``edge_attr``, ``y``, the
            train/test edge masks, and ``edge_time``.
        """
        import torch
        from torch_geometric.data import Data

        return Data(
            x=torch.from_numpy(self.x),
            edge_index=torch.from_numpy(self.edge_index),
            edge_attr=torch.from_numpy(self.edge_attr),
            y=torch.from_numpy(self.y.astype(np.int64)),
            edge_time=torch.from_numpy(self.edge_time),
            train_edge_mask=torch.from_numpy(self.train_edge_mask),
            test_edge_mask=torch.from_numpy(self.test_edge_mask),
        )


def _account_key(bank: pd.Series, account: pd.Series) -> pd.Series:
    """Composite node key. Account numbers collide across banks, so the bank is part of the id."""
    return bank.astype(str).str.strip() + ":" + account.astype(str).str.strip()


def load_transactions(csv_path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a ``*_Trans.csv`` ledger into a DataFrame with parsed timestamps and typed columns.

    Args:
        csv_path: Path to e.g. ``data/archive/HI-Small_Trans.csv``.
        nrows: Optional row cap, for quick smoke runs.

    Returns:
        One row per transaction with composite ``from_key`` / ``to_key`` account ids, a parsed
        ``timestamp``, and the edge-level ``is_laundering`` label.

    Raises:
        FileNotFoundError: If the CSV is missing - the dataset is downloaded manually, see
            ``src/data/download_instructions.md``.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Transaction ledger not found at {csv_path}. The AMLWorld dataset is downloaded "
            f"manually - see src/data/download_instructions.md."
        )
    df = pd.read_csv(
        csv_path,
        names=TRANS_COLUMNS,
        skiprows=1,
        dtype=TRANS_DTYPES,
        nrows=nrows,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    df["from_key"] = _account_key(df["from_bank"], df["from_account"])
    df["to_key"] = _account_key(df["to_bank"], df["to_account"])
    return df


def load_pattern_groups(patterns_path: Path) -> dict[str, dict[str, Any]]:
    """Parse a ``*_Patterns.txt`` file into ground-truth ring groups.

    The file is a sequence of blocks::

        BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
        <transaction rows, same schema as the ledger>
        END LAUNDERING ATTEMPT - FAN-OUT

    Args:
        patterns_path: Path to e.g. ``data/archive/HI-Small_Patterns.txt``.

    Returns:
        Mapping of pattern group id to ``{"typology", "description", "accounts",
        "transaction_keys"}`` - the typology being FAN-OUT, FAN-IN, CYCLE, GATHER-SCATTER,
        SCATTER-GATHER, STACK, BIPARTITE, or RANDOM. This is the ring-level ground truth that
        ``src.eval.metrics.ring_level_recovery`` scores discovered communities against.

    Raises:
        FileNotFoundError: If the patterns file is missing.
    """
    if not patterns_path.exists():
        raise FileNotFoundError(
            f"Patterns file not found at {patterns_path}. See src/data/download_instructions.md."
        )

    groups: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    counter = 0

    with patterns_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            begin = _PATTERN_BEGIN.match(line)
            if begin:
                counter += 1
                current = {
                    "group_id": f"pat_{counter:05d}",
                    "typology": begin.group(1).strip(),
                    "description": begin.group(2).strip(),
                    "accounts": set(),
                    "transaction_keys": set(),
                }
                continue
            if _PATTERN_END.match(line):
                if current is not None:
                    groups[current["group_id"]] = current
                current = None
                continue
            if current is None:
                continue

            parts = line.split(",")
            if len(parts) < 11:
                continue
            ts, from_bank, from_acct, to_bank, to_acct = parts[0], parts[1], parts[2], parts[3], parts[4]
            amount_paid = parts[7]
            src = f"{from_bank.strip()}:{from_acct.strip()}"
            dst = f"{to_bank.strip()}:{to_acct.strip()}"
            current["accounts"].update((src, dst))
            current["transaction_keys"].add(f"{ts}|{src}|{dst}|{amount_paid}")

    return groups


def _transaction_keys(df: pd.DataFrame) -> pd.Series:
    """Build the join key that matches ledger rows to pattern-file rows."""
    return (
        df["timestamp"].dt.strftime("%Y/%m/%d %H:%M")
        + "|"
        + df["from_key"]
        + "|"
        + df["to_key"]
        + "|"
        + df["amount_paid"].map(lambda v: f"{v:.2f}")
    )


def subsample(
    transactions: pd.DataFrame,
    pattern_groups: dict[str, dict[str, Any]],
    target_accounts: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Cut the Small split down to a weekend-sized graph without destroying the ring structure.

    Keeps every account appearing in a labeled laundering pattern or on a labeled laundering edge,
    then adds a random sample of non-laundering accounts up to ``target_accounts``, and finally
    keeps the **induced** subgraph - transactions where both endpoints survived. Taking the induced
    subgraph rather than all edges touching a kept account is what preserves ring topology: a
    fan-out is only a fan-out if its spokes are still there.

    Args:
        transactions: Full ledger from :func:`load_transactions`.
        pattern_groups: Ground-truth groups from :func:`load_pattern_groups`.
        target_accounts: Roughly how many accounts to end up with (50K-150K is the intended range).
        seed: RNG seed, so the subsample is reproducible across runs and across models.

    Returns:
        The subsampled transaction frame.
    """
    rng = np.random.default_rng(seed)

    illicit_accounts: set[str] = set()
    for group in pattern_groups.values():
        illicit_accounts.update(group["accounts"])
    flagged = transactions["is_laundering"] == 1
    illicit_accounts.update(transactions.loc[flagged, "from_key"].unique())
    illicit_accounts.update(transactions.loc[flagged, "to_key"].unique())

    all_accounts = pd.unique(
        pd.concat([transactions["from_key"], transactions["to_key"]], ignore_index=True)
    )
    illicit_present = np.array([a for a in all_accounts if a in illicit_accounts])
    clean = np.array([a for a in all_accounts if a not in illicit_accounts])

    n_clean = max(0, min(len(clean), target_accounts - len(illicit_present)))
    sampled_clean = rng.choice(clean, size=n_clean, replace=False) if n_clean else np.array([])

    keep = set(illicit_present.tolist()) | set(sampled_clean.tolist())
    mask = transactions["from_key"].isin(keep) & transactions["to_key"].isin(keep)
    out = transactions.loc[mask].reset_index(drop=True)

    print(
        f"  subsample: {len(illicit_present):,} ring-linked + {n_clean:,} sampled clean accounts "
        f"-> {len(out):,} induced transactions ({out['is_laundering'].sum():,} laundering)"
    )
    return out


def time_based_split(
    transactions: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split edges chronologically - earlier transactions train, later ones are held out.

    A random shuffle would leak future graph structure into training (both endpoints of a test
    transaction would already have been seen transacting), so the cut is strictly by time.

    Args:
        transactions: Subsampled ledger.
        test_fraction: Fraction of edges (by time, not by row shuffle) reserved for test.

    Returns:
        ``(train_df, test_df, split_timestamp)``.
    """
    ordered = transactions.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - test_fraction))
    split_timestamp = ordered.loc[cut, "timestamp"]
    train = ordered.loc[ordered["timestamp"] < split_timestamp].reset_index(drop=True)
    test = ordered.loc[ordered["timestamp"] >= split_timestamp].reset_index(drop=True)
    return train, test, split_timestamp


def compute_node_features(
    transactions: pd.DataFrame, accounts: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Aggregate per-account features (``design.md`` section 3.5).

    Transaction count, in/out degree, amount mean/std/max, unique-counterparty count, and active
    time span. Computed on the **training window only** by the caller, so the held-out split stays
    clean, and reindexed onto the full account list (accounts unseen in training get zeros).

    Args:
        transactions: Transactions to aggregate over - pass the training slice.
        accounts: Full ordered account list, defining node index order.

    Returns:
        ``(x, feature_names)`` with ``x`` of shape ``(len(accounts), n_features)``.
    """
    out = pd.DataFrame(index=pd.Index(accounts, name="account"))

    sent = transactions.groupby("from_key", observed=True).agg(
        out_count=("amount_paid", "size"),
        out_amount_mean=("amount_paid", "mean"),
        out_amount_std=("amount_paid", "std"),
        out_amount_max=("amount_paid", "max"),
        out_amount_sum=("amount_paid", "sum"),
        out_unique_counterparties=("to_key", "nunique"),
        first_sent=("timestamp", "min"),
        last_sent=("timestamp", "max"),
    )
    received = transactions.groupby("to_key", observed=True).agg(
        in_count=("amount_received", "size"),
        in_amount_mean=("amount_received", "mean"),
        in_amount_std=("amount_received", "std"),
        in_amount_max=("amount_received", "max"),
        in_amount_sum=("amount_received", "sum"),
        in_unique_counterparties=("from_key", "nunique"),
        first_received=("timestamp", "min"),
        last_received=("timestamp", "max"),
    )
    out = out.join(sent).join(received)

    first = out[["first_sent", "first_received"]].min(axis=1)
    last = out[["last_sent", "last_received"]].max(axis=1)
    out["active_span_hours"] = (last - first).dt.total_seconds() / 3600.0
    out = out.drop(columns=["first_sent", "last_sent", "first_received", "last_received"])

    out["txn_count"] = out["out_count"].fillna(0) + out["in_count"].fillna(0)
    out["degree"] = out["out_unique_counterparties"].fillna(0) + out["in_unique_counterparties"].fillna(0)
    # Ratios are the ring-relevant part: a mule passes through nearly everything it receives, and
    # a fan-out hub sends to far more counterparties than it receives from.
    out["net_flow"] = out["in_amount_sum"].fillna(0) - out["out_amount_sum"].fillna(0)
    out["pass_through_ratio"] = out["out_amount_sum"].fillna(0) / (out["in_amount_sum"].fillna(0) + 1.0)
    out["fan_ratio"] = (out["out_unique_counterparties"].fillna(0) + 1.0) / (
        out["in_unique_counterparties"].fillna(0) + 1.0
    )

    out = out.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    names = list(out.columns)
    return out.to_numpy(dtype=np.float32), names


def compute_edge_features(
    transactions: pd.DataFrame, currency_levels: list[str], format_levels: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Build per-transaction features (``design.md`` section 3.6).

    Amount, currency, payment format, timestamp, and repeat-count between the same account pair.
    Amounts are log1p-compressed - they span cents to millions in this dataset.

    Args:
        transactions: Transactions to featurize, in final edge order.
        currency_levels: Fixed currency vocabulary, so train and test encode identically.
        format_levels: Fixed payment-format vocabulary.

    Returns:
        ``(edge_attr, feature_names)``.
    """
    df = transactions
    feats: dict[str, np.ndarray] = {}

    feats["log_amount_paid"] = np.log1p(df["amount_paid"].to_numpy(dtype=np.float64))
    feats["log_amount_received"] = np.log1p(df["amount_received"].to_numpy(dtype=np.float64))
    # A mismatch between paid and received means a currency conversion happened mid-hop, a
    # classic layering tell.
    feats["amount_delta_ratio"] = (
        df["amount_received"].to_numpy() / (df["amount_paid"].to_numpy() + 1.0)
    )
    feats["is_cross_currency"] = (
        df["receiving_currency"].astype(str) != df["payment_currency"].astype(str)
    ).to_numpy(dtype=np.float64)
    feats["is_self_loop"] = (df["from_key"] == df["to_key"]).to_numpy(dtype=np.float64)

    ts = df["timestamp"]
    feats["hour_of_day"] = ts.dt.hour.to_numpy(dtype=np.float64)
    feats["day_index"] = (ts - ts.min()).dt.total_seconds().to_numpy() / 86400.0
    feats["is_night"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).to_numpy(dtype=np.float64)

    pair = df["from_key"].astype(str) + ">" + df["to_key"].astype(str)
    feats["pair_repeat_count"] = pair.map(pair.value_counts()).to_numpy(dtype=np.float64)

    for level in format_levels:
        feats[f"fmt_{level.replace(' ', '_').lower()}"] = (
            df["payment_format"].astype(str) == level
        ).to_numpy(dtype=np.float64)
    for level in currency_levels:
        feats[f"cur_{level.replace(' ', '_').lower()}"] = (
            df["payment_currency"].astype(str) == level
        ).to_numpy(dtype=np.float64)

    names = list(feats.keys())
    matrix = np.column_stack([feats[n] for n in names]).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, names


def build_graph(
    split: str = "HI-Small",
    data_dir: Path = ARCHIVE_DIR,
    target_accounts: int = 100_000,
    test_fraction: float = 0.2,
    seed: int = 42,
    nrows: int | None = None,
) -> GraphBundle:
    """Full pipeline: load -> patterns -> subsample -> time-split -> featurize -> graph.

    Args:
        split: Which split to build from - ``"HI-Small"`` or ``"LI-Small"``.
        data_dir: Directory holding the CSVs.
        target_accounts: Subsample target, see :func:`subsample`.
        test_fraction: Held-out fraction, see :func:`time_based_split`.
        seed: RNG seed for reproducibility.
        nrows: Optional ledger row cap, for smoke runs.

    Returns:
        A :class:`GraphBundle` ready for ``src.models.baseline`` and ``src.models.gnn``.
    """
    t0 = time.time()
    trans_path = data_dir / f"{split}_Trans.csv"
    patterns_path = data_dir / f"{split}_Patterns.txt"

    print(f"[1/6] loading {trans_path.name} ...")
    transactions = load_transactions(trans_path, nrows=nrows)
    print(
        f"  {len(transactions):,} transactions | "
        f"{transactions['is_laundering'].sum():,} laundering "
        f"({transactions['is_laundering'].mean():.4%})"
    )

    print(f"[2/6] parsing {patterns_path.name} ...")
    pattern_groups = load_pattern_groups(patterns_path)
    typologies = pd.Series([g["typology"] for g in pattern_groups.values()]).value_counts()
    print(f"  {len(pattern_groups):,} pattern groups | {typologies.to_dict()}")

    print("[3/6] subsampling ...")
    transactions = subsample(transactions, pattern_groups, target_accounts, seed)

    print("[4/6] time-based split ...")
    ordered = transactions.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    train_df, _, split_timestamp = time_based_split(ordered, test_fraction)
    train_mask = (ordered["timestamp"] < split_timestamp).to_numpy()
    test_mask = ~train_mask
    print(
        f"  split at {split_timestamp} | train {train_mask.sum():,} / test {test_mask.sum():,}"
    )

    print("[5/6] building features ...")
    accounts = pd.unique(
        pd.concat([ordered["from_key"], ordered["to_key"]], ignore_index=True)
    )
    account_index = {acct: i for i, acct in enumerate(accounts)}

    # Node features come from the TRAINING window only - no future information.
    x, node_names = compute_node_features(train_df, accounts)

    currency_levels = sorted(ordered["payment_currency"].astype(str).unique())
    format_levels = sorted(ordered["payment_format"].astype(str).unique())
    edge_attr, edge_names = compute_edge_features(ordered, currency_levels, format_levels)

    edge_index = np.vstack(
        [
            ordered["from_key"].map(account_index).to_numpy(dtype=np.int64),
            ordered["to_key"].map(account_index).to_numpy(dtype=np.int64),
        ]
    )
    y = ordered["is_laundering"].to_numpy(dtype=np.int8)
    edge_time = (ordered["timestamp"].astype("int64") // 10**9).to_numpy()

    print("[6/6] mapping edges to ground-truth pattern groups ...")
    key_to_group: dict[str, str] = {}
    for gid, group in pattern_groups.items():
        for k in group["transaction_keys"]:
            key_to_group[k] = gid
    edge_pattern_group = _transaction_keys(ordered).map(key_to_group).to_numpy(dtype=object)
    matched = int(pd.notna(edge_pattern_group).sum())
    n_groups_present = len({g for g in edge_pattern_group if isinstance(g, str)})
    print(f"  {matched:,} edges matched to {n_groups_present:,} pattern groups")

    bundle = GraphBundle(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_feature_names=edge_names,
        x=x,
        node_feature_names=node_names,
        y=y,
        edge_time=edge_time,
        accounts=np.asarray(accounts, dtype=object),
        account_index=account_index,
        train_edge_mask=train_mask,
        test_edge_mask=test_mask,
        split_timestamp=split_timestamp,
        pattern_groups=pattern_groups,
        edge_pattern_group=edge_pattern_group,
        meta={
            "split": split,
            "target_accounts": target_accounts,
            "test_fraction": test_fraction,
            "seed": seed,
            "build_seconds": round(time.time() - t0, 1),
            "pattern_groups_in_graph": n_groups_present,
        },
    )
    print(f"\n{bundle.summary()}\n  built in {bundle.meta['build_seconds']}s")
    return bundle


def save_graph(bundle: GraphBundle, out_path: Path) -> None:
    """Serialize a built graph so training runs don't rebuild it every time.

    Pickles a plain ``dict`` of arrays and builtins rather than the :class:`GraphBundle` instance
    itself. Pickling the dataclass records whichever module defined it, which is ``__main__`` when
    this file is run via ``python -m`` - the artifact would then only load inside a process that
    happens to have that same name bound, and would fail under uvicorn. A plain dict has no such
    dependency.

    Args:
        bundle: The bundle to persist.
        out_path: Destination ``.pkl`` file (gitignored - lives under ``data/``).
    """
    import pickle

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f.name: getattr(bundle, f.name) for f in fields(bundle)}
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved graph -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def load_graph(path: Path | str) -> GraphBundle:
    """Load a graph previously written by :func:`save_graph`.

    Args:
        path: File written by :func:`save_graph`.

    Returns:
        The reconstructed :class:`GraphBundle`.

    Raises:
        FileNotFoundError: If the graph has not been built yet.
    """
    import pickle

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No graph at {path}. Build it first: python -m src.data.build_graph"
        )
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, GraphBundle):  # artifact from an older build
        return payload
    return GraphBundle(**payload)


def main() -> None:
    """CLI entry point: ``python -m src.data.build_graph --split HI-Small --max-accounts 100000``."""
    parser = argparse.ArgumentParser(description="Build the RingWatch transaction graph.")
    parser.add_argument("--split", default="HI-Small", help="HI-Small or LI-Small")
    parser.add_argument("--data-dir", type=Path, default=ARCHIVE_DIR)
    parser.add_argument("--max-accounts", type=int, default=100_000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrows", type=int, default=None, help="Row cap for a smoke run.")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "graph.pkl")
    args = parser.parse_args()

    bundle = build_graph(
        split=args.split,
        data_dir=args.data_dir,
        target_accounts=args.max_accounts,
        test_fraction=args.test_fraction,
        seed=args.seed,
        nrows=args.nrows,
    )
    save_graph(bundle, args.out)


if __name__ == "__main__":
    main()
