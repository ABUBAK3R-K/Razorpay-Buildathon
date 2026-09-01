"""Gradient-boosted-tree baseline for RingWatch.

XGBoost (or LightGBM) trained on aggregated node + edge features - the same style of baseline
IBM's own AML paper uses. This is **not** a throwaway step: it is the comparison point the entire
pitch narrative rests on ("here is what the graph catches that this does not"), so its numbers get
reported side by side with the GNN's, honestly, in both directions (``design.md`` section 4).

What the baseline can and cannot see, by construction:

* It sees each transaction's own features and the **one-hop aggregates** of its two endpoint
  accounts - degree, amounts, counterparty counts, pass-through ratio.
* It does not see who those counterparties are, or whether they are the same accounts that show up
  around other flagged transactions. A ring is a multi-hop, shared-neighbour phenomenon, and that
  is exactly the signal the GNN gets and this model does not.

Run: ``python -m src.models.baseline --train``
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.data.build_graph import DATA_DIR, GraphBundle, load_graph
from src.eval.metrics import CostModel, evaluate_model


@dataclass
class BaselineConfig:
    """Hyperparameters for the GBT baseline.

    Attributes:
        library: ``"xgboost"`` or ``"lightgbm"``.
        n_estimators: Number of boosting rounds.
        max_depth: Tree depth cap.
        learning_rate: Boosting learning rate.
        subsample: Row subsampling fraction per tree.
        colsample_bytree: Feature subsampling fraction per tree.
        scale_pos_weight: Positive-class weight. ``None`` computes it from the training split as
            ``n_negative / n_positive``, which is the right default under ~1% positives.
        seed: RNG seed for reproducibility.
        params: Any additional library-specific parameters.
    """

    library: str = "xgboost"
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    scale_pos_weight: float | None = None
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)


def feature_names(bundle: GraphBundle) -> list[str]:
    """Column names for the matrix :func:`build_feature_matrix` produces."""
    return (
        list(bundle.edge_feature_names)
        + [f"src_{n}" for n in bundle.node_feature_names]
        + [f"dst_{n}" for n in bundle.node_feature_names]
        + ["pair_degree_sum", "pair_degree_diff", "pair_passthrough_gap"]
    )


def build_feature_matrix(bundle: GraphBundle, edge_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the graph into the tabular view the baseline sees.

    For each transaction edge, concatenate its own edge features with the aggregated node features
    of both endpoints, plus three cheap pair interactions. The baseline gets no relational signal
    beyond these one-hop aggregates - that gap is exactly what the GNN comparison measures.

    Args:
        bundle: Built graph from ``src.data.build_graph``.
        edge_mask: Which edges to materialize (train or test mask from the bundle).

    Returns:
        ``(X, y)`` - feature matrix and edge-level ``Is Laundering`` labels.
    """
    idx = np.flatnonzero(edge_mask)
    src, dst = bundle.edge_index[0, idx], bundle.edge_index[1, idx]
    x_src, x_dst = bundle.x[src], bundle.x[dst]

    names = bundle.node_feature_names
    deg = names.index("degree")
    pt = names.index("pass_through_ratio")
    pair = np.column_stack(
        [
            x_src[:, deg] + x_dst[:, deg],
            np.abs(x_src[:, deg] - x_dst[:, deg]),
            np.abs(x_src[:, pt] - x_dst[:, pt]),
        ]
    ).astype(np.float32)

    X = np.hstack([bundle.edge_attr[idx], x_src, x_dst, pair]).astype(np.float32)
    return X, bundle.y[idx].astype(int)


def train(bundle: GraphBundle, config: BaselineConfig | None = None) -> Any:
    """Fit the baseline on the training (earlier-in-time) edges.

    Args:
        bundle: Built graph, whose ``train_edge_mask`` selects the training edges.
        config: Hyperparameters; defaults to :class:`BaselineConfig`.

    Returns:
        The fitted booster, carrying ``feature_names_`` for the explanation endpoint.

    Raises:
        ImportError: If the requested library is not installed.
    """
    cfg = config or BaselineConfig()
    X, y = build_feature_matrix(bundle, bundle.train_edge_mask)
    names = feature_names(bundle)

    n_pos = int(y.sum())
    spw = cfg.scale_pos_weight or ((len(y) - n_pos) / max(n_pos, 1))
    print(f"  training on {len(y):,} edges ({n_pos:,} positive), scale_pos_weight={spw:.1f}")

    t0 = time.time()
    if cfg.library == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            scale_pos_weight=spw,
            random_state=cfg.seed,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            **cfg.params,
        )
    elif cfg.library == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            scale_pos_weight=spw,
            random_state=cfg.seed,
            n_jobs=-1,
            verbose=-1,
            **cfg.params,
        )
    else:
        raise ValueError(f"Unknown library {cfg.library!r} - use 'xgboost' or 'lightgbm'.")

    model.fit(X, y)
    model.feature_names_ = names  # type: ignore[attr-defined]
    print(f"  fitted {cfg.library} in {time.time() - t0:.1f}s")
    return model


def predict(model: Any, bundle: GraphBundle, edge_mask: np.ndarray) -> np.ndarray:
    """Score edges with a fitted baseline.

    Args:
        model: Booster from :func:`train`.
        bundle: Built graph.
        edge_mask: Which edges to score.

    Returns:
        Risk scores in [0, 1], one per selected edge, aligned with the mask's edge order.
    """
    X, _ = build_feature_matrix(bundle, edge_mask)
    return model.predict_proba(X)[:, 1]


def evaluate(
    model: Any, bundle: GraphBundle, cost_model: CostModel | None = None
) -> dict[str, Any]:
    """Score the held-out split and compute the metrics reported next to the GNN's.

    Args:
        model: Booster from :func:`train`.
        bundle: Built graph, whose ``test_edge_mask`` selects the held-out edges.
        cost_model: Unit costs for the false-positive-cost estimate.

    Returns:
        The bundle from ``src.eval.metrics.evaluate_model``.
    """
    scores = predict(model, bundle, bundle.test_edge_mask)
    y_true = bundle.y[bundle.test_edge_mask].astype(int)
    return evaluate_model("baseline", y_true, scores, cost_model)


def feature_importances(model: Any, top_k: int = 20) -> list[tuple[str, float]]:
    """Return the top contributing features, for the API's per-transaction explanation.

    Args:
        model: Fitted booster.
        top_k: How many features to return.

    Returns:
        ``(feature_name, importance)`` pairs, most important first.
    """
    names = getattr(model, "feature_names_", None)
    importances = np.asarray(model.feature_importances_, dtype=float)
    if names is None:
        names = [f"f{i}" for i in range(len(importances))]
    order = np.argsort(importances)[::-1][:top_k]
    return [(names[i], float(importances[i])) for i in order]


def save(model: Any, path: Path) -> None:
    """Persist a fitted baseline.

    Args:
        model: Fitted booster.
        path: Destination file (gitignored).
    """
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved baseline -> {path}")


def load(path: Path) -> Any:
    """Load a baseline written by :func:`save`.

    Args:
        path: File written by :func:`save`.

    Returns:
        The fitted booster.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"No baseline at {path}. Train it first: python -m src.models.baseline --train"
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


def main() -> None:
    """CLI entry point: ``python -m src.models.baseline --train``."""
    parser = argparse.ArgumentParser(description="Train/evaluate the RingWatch GBT baseline.")
    parser.add_argument("--train", action="store_true", help="Train (otherwise load and evaluate).")
    parser.add_argument("--graph", type=Path, default=DATA_DIR / "graph.pkl")
    parser.add_argument("--library", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=DATA_DIR / "baseline.pkl")
    parser.add_argument("--scores-out", type=Path, default=DATA_DIR / "scores_baseline.npz")
    args = parser.parse_args()

    print(f"loading graph from {args.graph} ...")
    bundle = load_graph(args.graph)
    print(bundle.summary())

    if args.train:
        cfg = BaselineConfig(
            library=args.library,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            seed=args.seed,
        )
        print(f"\ntraining {args.library} baseline ...")
        model = train(bundle, cfg)
        save(model, args.model_out)
    else:
        model = load(args.model_out)

    print("\nevaluating on the held-out (later-in-time) split ...")
    result = evaluate(model, bundle)
    m, mb, cb = result["metrics"], result["metrics_at_best"], result["cost_at_best"]
    print(
        f"  PR-AUC        : {m['pr_auc']:.4f}\n"
        f"  ROC-AUC       : {m['roc_auc']:.4f}\n"
        f"  @0.50  P/R/F1 : {m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f}"
        f"  ({m['n_flagged']:,} flagged of {len(bundle.y[bundle.test_edge_mask]):,})\n"
        f"  cost-opt thr  : {result['best_threshold']:.4f}\n"
        f"  @opt   P/R    : {mb['precision']:.3f} / {mb['recall']:.3f}\n"
        f"  net cost      : {cb['net_cost']:,.0f} "
        f"(do-nothing {cb['baseline_do_nothing_cost']:,.0f}, "
        f"saved {cb['cost_saved_vs_do_nothing']:,.0f})"
    )

    print("\n  top features:")
    for name, imp in feature_importances(model, 12):
        print(f"    {imp:.4f}  {name}")

    # Score every edge, so ring extraction and the API can use the full graph.
    all_scores = predict(model, bundle, np.ones(bundle.n_edges, dtype=bool))
    np.savez_compressed(
        args.scores_out,
        scores=all_scores,
        y=bundle.y,
        test_mask=bundle.test_edge_mask,
    )
    print(f"\nsaved edge scores -> {args.scores_out}")

    (DATA_DIR / "metrics_baseline.json").write_text(
        json.dumps({**result, "config": asdict(BaselineConfig(library=args.library))}, indent=2, default=float),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
