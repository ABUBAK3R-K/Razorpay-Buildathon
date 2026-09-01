"""Ring extraction for RingWatch.

Turns edge-level risk scores into candidate *rings*: threshold the scored edges into a high-risk
subgraph, run community detection over it, and cross-check the resulting clusters against the
dataset's labeled pattern groups to measure ring-level recovery (``design.md`` section 6).

This is the step that makes RingWatch a ring detector rather than a transaction scorer - a flagged
edge on its own is not the deliverable, the cluster it belongs to is. It is also what an analyst
actually reads: "these 14 accounts moved money in a fan-out over three days" is reviewable in a way
that a list of 14 separately-flagged transactions is not.

Nothing here acts on an account or a transaction. Rings are produced for review only.

Run: ``python -m src.rings.extract --threshold 0.8``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.data.build_graph import DATA_DIR, GraphBundle, load_graph

TYPOLOGIES = ("FAN-OUT", "FAN-IN", "CYCLE", "GATHER-SCATTER", "RANDOM")


@dataclass
class Ring:
    """One candidate abuse ring.

    Attributes:
        ring_id: Stable identifier, used by the ``/rings`` API endpoints.
        accounts: Account keys (``"<bank>:<account>"``) in the cluster.
        edge_ids: Transaction edge indices into the bundle's edge arrays.
        mean_risk_score: Mean model score across the ring's flagged edges.
        max_risk_score: Highest single edge score in the ring.
        size: Number of accounts.
        n_transactions: Number of flagged transactions in the ring.
        total_amount: Sum of the flagged transaction amounts (log-space feature reversed to
            currency units where available, else 0).
        detected_typology: Structural shape inferred by :func:`classify_typology` - derived from
            the discovered subgraph, never read from the ground-truth pattern file.
        n_true_positive_edges: How many of the ring's edges are actually labeled laundering.
            Evaluation only - not available at serving time.
        matched_pattern_group: Ground-truth pattern group this ring overlaps with, if any.
            Evaluation only.
    """

    ring_id: str
    accounts: list[str]
    edge_ids: list[int]
    mean_risk_score: float
    max_risk_score: float
    size: int
    n_transactions: int = 0
    total_amount: float = 0.0
    detected_typology: str = "RANDOM"
    n_true_positive_edges: int = 0
    matched_pattern_group: str | None = None
    role_by_account: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, for JSON export and the API layer."""
        return asdict(self)


def build_flagged_subgraph(
    bundle: GraphBundle, scores: np.ndarray, threshold: float = 0.8
) -> tuple[Any, np.ndarray]:
    """Keep only edges scored above the risk threshold.

    Self-loops are dropped here: ~26% of AMLWorld rows are ``Reinvestment`` transactions from an
    account to itself, and they connect nothing, so they inflate the subgraph without contributing
    any community structure. They stay in the edge-level scoring, just not in ring formation.

    Args:
        bundle: Built graph.
        scores: Per-edge risk scores from ``src.models.gnn.predict`` or the baseline, one per edge
            in the full graph.
        threshold: Score above which an edge enters the flagged subgraph. Tunable - it trades ring
            recall against the manual-review cost priced in ``src.eval.metrics``.

    Returns:
        ``(directed_graph, flagged_edge_ids)`` - a NetworkX ``DiGraph`` whose nodes are account
        keys and whose edges carry ``risk``, ``edge_id``, and ``weight``, plus the edge indices
        that survived thresholding.
    """
    import networkx as nx

    src_all, dst_all = bundle.edge_index[0], bundle.edge_index[1]
    keep = (scores >= threshold) & (src_all != dst_all)
    edge_ids = np.flatnonzero(keep)

    graph = nx.DiGraph()
    accounts = bundle.accounts
    for eid in edge_ids:
        u, v = accounts[src_all[eid]], accounts[dst_all[eid]]
        risk = float(scores[eid])
        if graph.has_edge(u, v):
            data = graph[u][v]
            data["weight"] += risk
            data["count"] += 1
            data["edge_ids"].append(int(eid))
            data["risk"] = max(data["risk"], risk)
        else:
            graph.add_edge(u, v, weight=risk, risk=risk, count=1, edge_ids=[int(eid)])
    return graph, edge_ids


def detect_communities(
    graph: Any, method: str = "louvain", resolution: float = 1.0, seed: int = 42
) -> dict[str, int]:
    """Partition the flagged subgraph into candidate rings.

    Louvain needs an undirected graph, so the directed subgraph is collapsed with edge weights
    summed across both directions. Direction is not lost - it is recovered from the original
    ``DiGraph`` in :func:`classify_typology`, which is where it actually matters.

    Args:
        graph: Directed graph from :func:`build_flagged_subgraph`.
        method: ``"louvain"`` (python-louvain) or ``"connected_components"``.
        resolution: Louvain resolution - higher yields more, smaller communities.
        seed: RNG seed; Louvain is randomized.

    Returns:
        Mapping of account key to community id.
    """
    import networkx as nx

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes())
    for u, v, data in graph.edges(data=True):
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += data["weight"]
        else:
            undirected.add_edge(u, v, weight=data["weight"])

    if method == "connected_components":
        return {
            node: cid
            for cid, comp in enumerate(nx.connected_components(undirected))
            for node in comp
        }
    if method == "louvain":
        import community as community_louvain

        return community_louvain.best_partition(
            undirected, weight="weight", resolution=resolution, random_state=seed
        )
    raise ValueError(f"Unknown method {method!r} - use 'louvain' or 'connected_components'.")


def classify_typology(graph: Any, accounts: list[str]) -> tuple[str, dict[str, str]]:
    """Infer a ring's structural shape from its own topology.

    Heuristics over the induced directed subgraph, checked in order:

    * **CYCLE** - the subgraph contains a directed cycle spanning at least half the ring.
    * **GATHER-SCATTER** - one account concentrates inflow and another (or the same one) fans it
      back out; both a high-in-degree and a high-out-degree hub exist.
    * **FAN-OUT** - a single account sends a large share of the ring's transactions.
    * **FAN-IN** - a single account receives a large share of the ring's transactions.
    * **RANDOM** - none of the above.

    Hub dominance is measured as a share of the ring's **edges**, not of its accounts. Louvain
    regularly merges several injected patterns into one community, and a node-count threshold then
    misses a clear 30-spoke fan-in sitting inside a 71-account cluster purely because the cluster
    is large.

    This is derived from the graph, not read from the ground-truth pattern file, so it is available
    at serving time and is what the explanation endpoint shows an analyst.

    Args:
        graph: The full flagged directed subgraph.
        accounts: Accounts belonging to this ring.

    Returns:
        ``(typology, role_by_account)`` where roles are ``"source"``, ``"sink"``,
        ``"intermediary"``, or ``"peripheral"``.
    """
    import networkx as nx

    sub = graph.subgraph(accounts)
    n = max(len(accounts), 1)
    out_deg = dict(sub.out_degree())
    in_deg = dict(sub.in_degree())
    max_out = max(out_deg.values(), default=0)
    max_in = max(in_deg.values(), default=0)
    hub_out = max(out_deg, key=lambda k: out_deg[k], default=None)
    hub_in = max(in_deg, key=lambda k: in_deg[k], default=None)

    roles: dict[str, str] = {}
    for acct in accounts:
        o, i = out_deg.get(acct, 0), in_deg.get(acct, 0)
        if o and i:
            roles[acct] = "intermediary"
        elif o:
            roles[acct] = "source"
        elif i:
            roles[acct] = "sink"
        else:
            roles[acct] = "peripheral"

    try:
        longest_cycle = max((len(c) for c in nx.simple_cycles(sub)), default=0)
    except Exception:  # pragma: no cover - very dense subgraphs
        longest_cycle = 0
    if longest_cycle >= max(3, 0.5 * n):
        return "CYCLE", roles

    n_edges = max(sub.number_of_edges(), 1)
    out_share, in_share = max_out / n_edges, max_in / n_edges
    if max_out >= 2 and max_in >= 2 and out_share >= 0.25 and in_share >= 0.25:
        return "GATHER-SCATTER", roles
    if max_out >= 2 and out_share >= 0.3:
        return "FAN-OUT", roles
    if max_in >= 2 and in_share >= 0.3:
        return "FAN-IN", roles
    return "RANDOM", roles


def extract_rings(
    bundle: GraphBundle,
    scores: np.ndarray,
    threshold: float = 0.8,
    method: str = "louvain",
    min_size: int = 3,
    resolution: float = 1.0,
    seed: int = 42,
) -> list[Ring]:
    """Full pipeline: threshold -> community detection -> typology -> :class:`Ring` objects.

    Args:
        bundle: Built graph.
        scores: Per-edge risk scores for every edge in the graph.
        threshold: Edge risk threshold.
        method: Community-detection method.
        min_size: Drop communities smaller than this - a two-account cluster is not a ring.
        resolution: Louvain resolution.
        seed: RNG seed.

    Returns:
        Candidate rings, highest mean risk score first.
    """
    graph, flagged_ids = build_flagged_subgraph(bundle, scores, threshold)
    print(
        f"  flagged subgraph: {graph.number_of_nodes():,} accounts, "
        f"{graph.number_of_edges():,} account pairs, {len(flagged_ids):,} transactions"
    )
    if graph.number_of_nodes() == 0:
        return []

    partition = detect_communities(graph, method, resolution, seed)
    by_community: dict[int, list[str]] = {}
    for account, cid in partition.items():
        by_community.setdefault(cid, []).append(account)
    print(f"  {len(by_community):,} communities before the min-size filter")

    amount_col = (
        bundle.edge_feature_names.index("log_amount_paid")
        if "log_amount_paid" in bundle.edge_feature_names
        else None
    )

    rings: list[Ring] = []
    for cid, accounts in by_community.items():
        if len(accounts) < min_size:
            continue
        member = set(accounts)
        # Every flagged transaction between ring members, not just one per account pair.
        edge_ids = sorted(
            {
                int(e)
                for u, v, d in graph.edges(data=True)
                if u in member and v in member
                for e in d["edge_ids"]
            }
        )
        if not edge_ids:
            continue

        ring_scores = scores[edge_ids]
        typology, roles = classify_typology(graph, accounts)
        total_amount = (
            float(np.expm1(bundle.edge_attr[edge_ids, amount_col]).sum())
            if amount_col is not None
            else 0.0
        )
        rings.append(
            Ring(
                ring_id=f"ring_{cid:05d}",
                accounts=sorted(accounts),
                edge_ids=edge_ids,
                mean_risk_score=float(ring_scores.mean()),
                max_risk_score=float(ring_scores.max()),
                size=len(accounts),
                n_transactions=len(edge_ids),
                total_amount=total_amount,
                detected_typology=typology,
                n_true_positive_edges=int(bundle.y[edge_ids].sum()),
                role_by_account=roles,
            )
        )

    rings.sort(key=lambda r: r.mean_risk_score, reverse=True)
    for i, ring in enumerate(rings, start=1):
        ring.ring_id = f"ring_{i:05d}"
    print(f"  {len(rings):,} rings with >= {min_size} accounts")
    return rings


def match_to_pattern_groups(
    rings: list[Ring],
    pattern_groups: dict[str, dict[str, Any]],
    min_overlap: float = 0.3,
) -> dict[str, str | None]:
    """Cross-check discovered rings against the dataset's ground-truth pattern groups.

    Args:
        rings: Candidate rings from :func:`extract_rings`.
        pattern_groups: Ground truth from ``src.data.build_graph.load_pattern_groups``.
        min_overlap: Fraction of the ground-truth group's accounts the ring must contain to count
            as a match. Recall over the group, not Jaccard: a large community that contains a small
            ring has still surfaced it for review.

    Returns:
        Mapping of ``ring_id`` to matched pattern group id (or ``None``). Also writes
        ``matched_pattern_group`` onto each ring in place. Feeds
        ``src.eval.metrics.ring_level_recovery``.
    """
    out: dict[str, str | None] = {}
    for ring in rings:
        member = set(ring.accounts)
        best_gid, best_frac = None, 0.0
        for gid, group in pattern_groups.items():
            accounts = group["accounts"]
            if not accounts:
                continue
            frac = len(accounts & member) / len(accounts)
            if frac > best_frac:
                best_gid, best_frac = gid, frac
        ring.matched_pattern_group = best_gid if best_frac >= min_overlap else None
        out[ring.ring_id] = ring.matched_pattern_group
    return out


def main() -> None:
    """CLI entry point: ``python -m src.rings.extract --threshold 0.8``."""
    from src.eval.metrics import ring_level_recovery

    parser = argparse.ArgumentParser(description="Extract candidate abuse rings.")
    parser.add_argument("--graph", type=Path, default=DATA_DIR / "graph.pkl")
    parser.add_argument(
        "--scores",
        type=Path,
        default=DATA_DIR / "scores_gnn.npz",
        help="npz from a training run; falls back to the baseline scores if absent.",
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--method", default="louvain", choices=["louvain", "connected_components"])
    parser.add_argument("--min-size", type=float, default=3)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-overlap", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "rings.json")
    args = parser.parse_args()

    scores_path = args.scores
    if not scores_path.exists():
        fallback = DATA_DIR / "scores_baseline.npz"
        if not fallback.exists():
            raise SystemExit(
                f"No scores at {scores_path} or {fallback}. Train a model first:\n"
                f"  python -m src.models.baseline --train"
            )
        print(f"{scores_path.name} not found, using {fallback.name}")
        scores_path = fallback

    bundle = load_graph(args.graph)
    scores = np.load(scores_path)["scores"]
    print(f"loaded {len(scores):,} edge scores from {scores_path.name}")

    print(f"\nextracting rings at threshold {args.threshold} ({args.method}) ...")
    rings = extract_rings(
        bundle,
        scores,
        threshold=args.threshold,
        method=args.method,
        min_size=int(args.min_size),
        resolution=args.resolution,
        seed=args.seed,
    )
    match_to_pattern_groups(rings, bundle.pattern_groups, args.min_overlap)

    groups_in_graph = {g for g in bundle.edge_pattern_group if isinstance(g, str)}
    recovery = ring_level_recovery(
        rings, bundle.pattern_groups, args.min_overlap, groups_in_graph
    )

    print(f"\nring-level recovery: {recovery['recovery_rate']:.1%} "
          f"({recovery['n_recovered']}/{recovery['n_groups']} ground-truth groups)")
    for typology, stats in recovery["by_typology"].items():
        print(f"    {typology:<16} {stats['recovered']:>3}/{stats['total']:<3} "
              f"({stats['recovery_rate']:.0%})")

    shapes: dict[str, int] = {}
    for ring in rings:
        shapes[ring.detected_typology] = shapes.get(ring.detected_typology, 0) + 1
    print(f"\ndetected shapes across {len(rings):,} rings: {shapes}")
    precise = sum(1 for r in rings if r.n_true_positive_edges > 0)
    print(f"rings containing >=1 truly-laundering edge: {precise:,}/{len(rings):,} "
          f"({precise / max(len(rings), 1):.1%})")

    args.out.write_text(
        json.dumps(
            {
                "threshold": args.threshold,
                "method": args.method,
                "source_scores": scores_path.name,
                "source_model": "gnn" if "gnn" in scores_path.name else "baseline",
                "recovery": recovery,
                "rings": [r.to_dict() for r in rings],
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out} ({len(rings):,} rings)")


if __name__ == "__main__":
    main()
