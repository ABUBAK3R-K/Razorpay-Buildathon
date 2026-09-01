"""Evaluation for RingWatch.

Three things get reported, for **both** the baseline and the GNN, side by side (``design.md``
section 7):

1. Edge-level precision / recall / PR-AUC on the later-in-time held-out split.
2. A false-positive **cost** estimate - an illustrative cost per false positive (manual review) and
   per false negative (missed fraud loss), reported as net cost rather than as accuracy. This is
   what the track's "honest metrics including false positive cost" bar is asking for.
3. Ring-level recovery - the fraction of ground-truth laundering pattern groups at least partially
   recovered by community detection.

PR-AUC is the headline number, not ROC-AUC and certainly not accuracy: at the observed positive
rate a model that flags nothing scores over 99.8% accurate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@dataclass
class CostModel:
    """Illustrative unit costs for the false-positive-cost estimate.

    These are **demo assumptions, not Razorpay figures** - state them explicitly wherever the net
    cost is reported, so the number is read as a comparison between models under a shared
    assumption rather than as a claim about real losses.

    The 100:1 ratio between a missed laundering transaction and an analyst review is what makes the
    trade-off interesting: at that ratio the cost-optimal threshold sits far below 0.5, and a model
    tuned to maximize F1 is leaving money on the table.

    Attributes:
        cost_per_false_positive: Analyst time to manually review one wrongly flagged transaction.
        cost_per_false_negative: Expected loss from one missed laundering transaction.
        cost_per_true_positive: Review cost for a correctly flagged transaction - it still needs a
            human look, because RingWatch never acts on its own.
        currency: Label for reporting.
    """

    cost_per_false_positive: float = 5.0
    cost_per_false_negative: float = 500.0
    cost_per_true_positive: float = 5.0
    currency: str = "USD"


def edge_level_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Precision, recall, F1, PR-AUC, and ROC-AUC at the edge level.

    Args:
        y_true: Ground-truth ``Is Laundering`` labels for the held-out edges.
        y_score: Predicted risk scores for the same edges, in the same order.
        threshold: Score cutoff for the point-estimate precision/recall/F1. PR-AUC is
            threshold-free and is the headline number under a <1% positive rate.

    Returns:
        ``{"precision", "recall", "f1", "pr_auc", "roc_auc", "n_positives", "n_flagged",
        "threshold"}``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pred = (y_score >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    n_pos = int(y_true.sum())
    pr_auc = float(average_precision_score(y_true, y_score)) if 0 < n_pos < len(y_true) else float("nan")
    roc_auc = float(roc_auc_score(y_true, y_score)) if 0 < n_pos < len(y_true) else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "n_positives": n_pos,
        "n_flagged": int(pred.sum()),
        "threshold": float(threshold),
    }


def precision_recall_curve_points(
    y_true: np.ndarray, y_score: np.ndarray, max_points: int = 500
) -> dict[str, list[float]]:
    """Return the PR curve, downsampled, for the dashboard's metrics panel.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted risk scores.
        max_points: Cap on returned points, so the JSON stays small enough to ship to a browser.

    Returns:
        ``{"precision": [...], "recall": [...], "thresholds": [...]}``.
    """
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float)
    )
    precision, recall = precision[:-1], recall[:-1]
    if len(thresholds) > max_points:
        idx = np.linspace(0, len(thresholds) - 1, max_points).astype(int)
        precision, recall, thresholds = precision[idx], recall[idx], thresholds[idx]
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def false_positive_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    cost_model: CostModel | None = None,
) -> dict[str, float]:
    """Price the confusion matrix at a given threshold.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted risk scores.
        threshold: Score cutoff defining the flag set.
        cost_model: Unit costs; defaults to :class:`CostModel`.

    Returns:
        ``{"tp", "fp", "tn", "fn", "review_cost", "missed_fraud_cost", "net_cost",
        "cost_per_1k_transactions", "baseline_do_nothing_cost", "cost_saved_vs_do_nothing"}``.
    """
    cm = cost_model or CostModel()
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    review_cost = tp * cm.cost_per_true_positive + fp * cm.cost_per_false_positive
    missed_cost = fn * cm.cost_per_false_negative
    net_cost = review_cost + missed_cost
    n = max(len(y_true), 1)

    # "Flag nothing" is the honest floor to beat: it costs zero review and loses every fraud.
    do_nothing = int(y_true.sum()) * cm.cost_per_false_negative

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "review_cost": review_cost,
        "missed_fraud_cost": missed_cost,
        "net_cost": net_cost,
        "cost_per_1k_transactions": net_cost / n * 1000,
        "baseline_do_nothing_cost": do_nothing,
        "cost_saved_vs_do_nothing": do_nothing - net_cost,
        "threshold": float(threshold),
    }


def optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_model: CostModel | None = None,
    n_steps: int = 200,
) -> tuple[float, dict[str, float]]:
    """Sweep thresholds and return the one minimizing net cost.

    A model that looks worse on PR-AUC can still win on net cost, and vice versa - reporting both
    is the point. The sweep uses score quantiles rather than a uniform grid, because scores under
    heavy imbalance pile up near zero.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted risk scores.
        cost_model: Unit costs; defaults to :class:`CostModel`.
        n_steps: Number of candidate thresholds to evaluate.

    Returns:
        ``(best_threshold, cost_breakdown_at_that_threshold)``.
    """
    y_score = np.asarray(y_score, dtype=float)
    candidates = np.unique(np.quantile(y_score, np.linspace(0.0, 1.0, n_steps)))
    best_t, best = candidates[0], None
    for t in candidates:
        breakdown = false_positive_cost(y_true, y_score, float(t), cost_model)
        if best is None or breakdown["net_cost"] < best["net_cost"]:
            best_t, best = float(t), breakdown
    return best_t, best  # type: ignore[return-value]


def ring_level_recovery(
    rings: Sequence[Any],
    pattern_groups: dict[str, dict[str, Any]],
    min_overlap: float = 0.3,
    groups_in_graph: set[str] | None = None,
) -> dict[str, Any]:
    """Fraction of ground-truth pattern groups recovered by community detection.

    A group counts as recovered when some discovered ring shares at least ``min_overlap`` of its
    accounts (recall over the group's account set, not Jaccard - a big community that swallows a
    small ring still found it, and an analyst reading the explanation would see it).

    Args:
        rings: Candidate rings from ``src.rings.extract.extract_rings``.
        pattern_groups: Ground truth from ``src.data.build_graph.load_pattern_groups``.
        min_overlap: Fraction of a group's accounts a ring must contain to count as recovering it.
        groups_in_graph: Restrict scoring to the groups that actually survived subsampling. Scoring
            against all 234 groups when only a fraction are present in the subsampled graph would
            understate recovery dishonestly in the other direction.

    Returns:
        ``{"recovery_rate", "n_groups", "n_recovered", "by_typology", "unmatched_rings",
        "n_rings"}`` - ``by_typology`` breaking recovery down by FAN-OUT / FAN-IN / CYCLE /
        GATHER-SCATTER / RANDOM, since the pitch claim is about ring-structured cases specifically.
    """
    considered = {
        gid: g
        for gid, g in pattern_groups.items()
        if groups_in_graph is None or gid in groups_in_graph
    }
    ring_account_sets = [set(getattr(r, "accounts", [])) for r in rings]
    matched_rings: set[int] = set()

    by_typology: dict[str, dict[str, int]] = {}
    n_recovered = 0
    for gid, group in considered.items():
        accounts = set(group["accounts"])
        typology = group["typology"]
        slot = by_typology.setdefault(typology, {"total": 0, "recovered": 0})
        slot["total"] += 1
        if not accounts:
            continue
        best_i, best_frac = -1, 0.0
        for i, ring_accounts in enumerate(ring_account_sets):
            frac = len(accounts & ring_accounts) / len(accounts)
            if frac > best_frac:
                best_i, best_frac = i, frac
        if best_frac >= min_overlap:
            n_recovered += 1
            slot["recovered"] += 1
            matched_rings.add(best_i)

    n_groups = len(considered)
    return {
        "recovery_rate": n_recovered / n_groups if n_groups else 0.0,
        "n_groups": n_groups,
        "n_recovered": n_recovered,
        "n_rings": len(rings),
        "unmatched_rings": len(rings) - len(matched_rings),
        "min_overlap": min_overlap,
        "by_typology": {
            t: {
                **v,
                "recovery_rate": v["recovered"] / v["total"] if v["total"] else 0.0,
            }
            for t, v in sorted(by_typology.items())
        },
    }


def evaluate_model(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_model: CostModel | None = None,
) -> dict[str, Any]:
    """Full metric bundle for one model at both the default and the cost-optimal threshold.

    Args:
        name: Model name for reporting.
        y_true: Ground-truth labels on the held-out split.
        y_score: Predicted risk scores.
        cost_model: Unit costs; defaults to :class:`CostModel`.

    Returns:
        ``{"model", "metrics", "cost_at_default", "best_threshold", "cost_at_best",
        "metrics_at_best"}``.
    """
    cm = cost_model or CostModel()
    best_t, cost_at_best = optimal_threshold(y_true, y_score, cm)
    return {
        "model": name,
        "metrics": edge_level_metrics(y_true, y_score, 0.5),
        "cost_at_default": false_positive_cost(y_true, y_score, 0.5, cm),
        "best_threshold": best_t,
        "cost_at_best": cost_at_best,
        "metrics_at_best": edge_level_metrics(y_true, y_score, best_t),
        "cost_model": asdict(cm),
    }


def compare_models(
    results: dict[str, dict[str, Any]],
    cost_model: CostModel | None = None,
    ring_recovery: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render the baseline-vs-GNN table that goes in the README and the dashboard panel.

    Reports both models honestly, including where the baseline wins - the narrative is "the graph
    wins specifically on ring-structured cases", not "the graph wins everywhere".

    Args:
        results: Mapping of model name to ``{"y_true", "y_score"}`` arrays.
        cost_model: Unit costs; defaults to :class:`CostModel`.
        ring_recovery: Optional mapping of model name to a :func:`ring_level_recovery` result.

    Returns:
        A Markdown table of precision, recall, PR-AUC, net FP-cost, and ring-level recovery.
    """
    cm = cost_model or CostModel()
    rows = []
    for name, payload in results.items():
        ev = evaluate_model(name, payload["y_true"], payload["y_score"], cm)
        m, mb, cb = ev["metrics"], ev["metrics_at_best"], ev["cost_at_best"]
        recovery = (ring_recovery or {}).get(name, {}).get("recovery_rate")
        rows.append(
            f"| {name} | {m['pr_auc']:.4f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{ev['best_threshold']:.4f} | {mb['precision']:.3f} | {mb['recall']:.3f} | "
            f"{cb['net_cost']:,.0f} | "
            + (f"{recovery:.1%} |" if recovery is not None else "n/a |")
        )

    header = (
        "| Model | PR-AUC | P@0.5 | R@0.5 | Cost-opt thr | P@opt | R@opt | "
        f"Net cost ({cm.currency}) | Ring recovery |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    note = (
        f"\n\nCost model (illustrative, not Razorpay figures): "
        f"{cm.currency} {cost_model_str(cm)}."
    )
    return header + "\n" + "\n".join(rows) + note


def cost_model_str(cm: CostModel) -> str:
    """One-line rendering of the cost assumptions, for footnotes under any cost table."""
    return (
        f"{cm.cost_per_false_positive:.0f}/false positive (manual review), "
        f"{cm.cost_per_false_negative:.0f}/missed laundering transaction"
    )


def main() -> None:
    """CLI entry point: ``python -m src.eval.metrics --compare``."""
    parser = argparse.ArgumentParser(description="Compare RingWatch models on the held-out split.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DATA_DIR / "predictions.npz",
        help="npz written by the training scripts: y_true plus one column per model.",
    )
    parser.add_argument("--fp-cost", type=float, default=5.0)
    parser.add_argument("--fn-cost", type=float, default=500.0)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "metrics.json")
    parser.add_argument("--compare", action="store_true", help="Print the comparison table.")
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(
            f"No predictions at {args.predictions}. Train a model first, e.g.\n"
            f"  python -m src.models.baseline --train"
        )

    npz = np.load(args.predictions, allow_pickle=True)
    y_true = npz["y_true"]
    cm = CostModel(cost_per_false_positive=args.fp_cost, cost_per_false_negative=args.fn_cost)

    results = {k: {"y_true": y_true, "y_score": npz[k]} for k in npz.files if k != "y_true"}
    if not results:
        raise SystemExit(f"{args.predictions} contains no model score columns.")

    table = compare_models(results, cm)
    print(table)

    payload = {
        name: evaluate_model(name, y_true, p["y_score"], cm) for name, p in results.items()
    }
    args.out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
