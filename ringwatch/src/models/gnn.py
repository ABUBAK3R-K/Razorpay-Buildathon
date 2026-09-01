"""Edge-classification GNN for RingWatch.

The ``Is Laundering`` label lives on the **transaction**, not the account, so this is edge
classification with an edge-featured message-passing architecture - GINE (GIN with edge features,
as in IBM's Multi-GNN repo), with GraphSAGE as the simpler fallback (``design.md`` section 5).

Two design decisions worth stating explicitly, because they are what keep the comparison with the
baseline honest:

**No temporal leakage.** Message passing runs over the *training* edges only. Node embeddings are
therefore built from history the model is allowed to have seen, and a held-out transaction is
scored from its two endpoints' historical embeddings plus its own features - exactly the
information a real deployment would have at the moment the transaction arrives. Passing messages
over the test edges too would let a test transaction's own existence inform its prediction.

**Label-balanced sampling, not reweighting.** Positives are ~1% of edges after subsampling.
Following PC-GNN (https://github.com/PonderLY/PC-GNN), each minibatch of *target* edges is
constructed to hit a fixed positive ratio rather than drawn uniformly. Message passing still sees
the whole training graph - only the supervision is balanced.

If exploration shows rings camouflaged inside normal transaction bursts, ``camouflage_filter``
enables a lightweight CARE-GNN-style neighbour-similarity cutoff - the similarity filter, not the
full reinforcement-learning neighbour selector.

Run: ``python -m src.models.gnn --train``
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.data.build_graph import DATA_DIR, GraphBundle, load_graph
from src.eval.metrics import CostModel, evaluate_model


@dataclass
class GNNConfig:
    """Architecture and training hyperparameters.

    Attributes:
        architecture: ``"gine"`` (GIN with edge features) or ``"sage"`` (simpler fallback).
        hidden_dim: Hidden channel width.
        num_layers: Message-passing depth - how many hops of ring structure the model can see.
            Two layers already reach a fan-out hub's spokes' other counterparties.
        dropout: Dropout probability.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        epochs: Training epochs.
        steps_per_epoch: Balanced minibatches drawn per epoch.
        batch_size: Target edges supervised per minibatch.
        pos_ratio: Target positive-edge fraction per batch for the label-balanced sampler.
        camouflage_filter: Enable the CARE-GNN-style neighbour-similarity filter.
        similarity_threshold: Cosine-similarity cutoff used when ``camouflage_filter`` is on.
        seed: RNG seed for reproducibility.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
    """

    architecture: str = "gine"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    steps_per_epoch: int = 32
    batch_size: int = 2048
    pos_ratio: float = 0.3
    camouflage_filter: bool = False
    similarity_threshold: float = 0.3
    seed: int = 42
    device: str = "auto"

    def resolved_device(self) -> str:
        """Return the concrete device, resolving ``"auto"`` against CUDA availability."""
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


def _build_module(node_dim: int, edge_dim: int, config: GNNConfig) -> Any:
    """Construct the torch module. Imported lazily so the rest of the repo works without torch."""
    import torch
    from torch import nn
    from torch_geometric.nn import GINEConv, SAGEConv

    hidden = config.hidden_dim

    class _EdgeGNN(nn.Module):
        """Node encoder -> message passing -> edge classifier head."""

        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.node_encoder = nn.Linear(node_dim, hidden)
            self.edge_encoder = nn.Linear(edge_dim, hidden)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(config.num_layers):
                if config.architecture == "gine":
                    mlp = nn.Sequential(
                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                    )
                    self.convs.append(GINEConv(mlp, edge_dim=hidden))
                elif config.architecture == "sage":
                    self.convs.append(SAGEConv(hidden, hidden))
                else:
                    raise ValueError(
                        f"Unknown architecture {config.architecture!r} - use 'gine' or 'sage'."
                    )
                self.norms.append(nn.BatchNorm1d(hidden))
            self.dropout = nn.Dropout(config.dropout)
            # The head sees both endpoint embeddings, their elementwise product (a cheap
            # interaction term), and the transaction's own raw features.
            self.head = nn.Sequential(
                nn.Linear(3 * hidden + edge_dim, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, 1),
            )

        def encode(
            self, x: Any, message_edge_index: Any, message_edge_attr: Any
        ) -> Any:
            """Run message passing over the history graph and return node embeddings."""
            h = self.node_encoder(x)
            e = self.edge_encoder(message_edge_attr)
            for conv, norm in zip(self.convs, self.norms):
                if self.config.architecture == "gine":
                    h = conv(h, message_edge_index, e)
                else:
                    h = conv(h, message_edge_index)
                h = norm(h)
                h = torch.relu(h)
                h = self.dropout(h)
            return h

        def classify(self, h: Any, target_edge_index: Any, target_edge_attr: Any) -> Any:
            """Score target transactions from their endpoints' embeddings and own features."""
            h_src = h[target_edge_index[0]]
            h_dst = h[target_edge_index[1]]
            return self.head(
                torch.cat([h_src, h_dst, h_src * h_dst, target_edge_attr], dim=-1)
            ).squeeze(-1)

        def forward(
            self,
            x: Any,
            message_edge_index: Any,
            message_edge_attr: Any,
            target_edge_index: Any,
            target_edge_attr: Any,
        ) -> Any:
            """Encode then classify - logits, one per target edge."""
            h = self.encode(x, message_edge_index, message_edge_attr)
            return self.classify(h, target_edge_index, target_edge_attr)

    return _EdgeGNN()


class EdgeGNN:
    """Edge-classification message-passing network over the account/transaction graph.

    Thin wrapper around the torch module so the rest of the repo (API, ring extraction, metrics)
    can import this file without importing torch at module load time.
    """

    def __init__(self, node_dim: int, edge_dim: int, config: GNNConfig | None = None) -> None:
        """Build the network.

        Args:
            node_dim: Node feature width.
            edge_dim: Edge feature width.
            config: Architecture hyperparameters; defaults to :class:`GNNConfig`.
        """
        self.config = config or GNNConfig()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.module = _build_module(node_dim, edge_dim, self.config)
        self.device = self.config.resolved_device()
        self.module.to(self.device)

    def forward(self, batch: dict[str, Any]) -> Any:
        """Run one minibatch forward.

        Args:
            batch: Dict with ``x``, ``message_edge_index``, ``message_edge_attr``,
                ``target_edge_index``, ``target_edge_attr``.

        Returns:
            Logits, one per target edge.
        """
        return self.module(
            batch["x"],
            batch["message_edge_index"],
            batch["message_edge_attr"],
            batch["target_edge_index"],
            batch["target_edge_attr"],
        )


def _tensors(bundle: GraphBundle, device: str) -> dict[str, Any]:
    """Move the graph onto the device once, as the tensors every step reuses."""
    import torch

    train_idx = np.flatnonzero(bundle.train_edge_mask)
    x = torch.from_numpy(bundle.x).float().to(device)
    # Standardize node features - raw amounts and counts differ by orders of magnitude and
    # otherwise dominate the first linear layer.
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
    edge_attr = torch.from_numpy(bundle.edge_attr).float().to(device)
    edge_index = torch.from_numpy(bundle.edge_index).long().to(device)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "message_edge_index": edge_index[:, train_idx],
        "message_edge_attr": edge_attr[train_idx],
        "y": torch.from_numpy(bundle.y.astype(np.float32)).to(device),
    }


def label_balanced_sampler(
    bundle: GraphBundle,
    edge_mask: np.ndarray,
    batch_size: int,
    pos_ratio: float = 0.3,
    steps: int = 32,
    seed: int = 42,
) -> Iterator[np.ndarray]:
    """Yield minibatches of target-edge indices with an enforced positive ratio (PC-GNN-style).

    Training on the raw positive rate tanks recall: the loss is dominated by an overwhelming
    majority of easy negatives. Each batch is instead built by drawing positives (with replacement,
    since there are few) and negatives (without) to hit ``pos_ratio``.

    Args:
        bundle: Built graph.
        edge_mask: Edges eligible for sampling (the training mask).
        batch_size: Target edges per batch.
        pos_ratio: Fraction of each batch that should be laundering edges.
        steps: Number of batches to yield.
        seed: RNG seed.

    Yields:
        Integer arrays of edge indices into the full edge list.
    """
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(edge_mask)
    labels = bundle.y[eligible]
    pos, neg = eligible[labels == 1], eligible[labels == 0]
    n_pos = max(1, int(batch_size * pos_ratio))
    n_neg = max(1, batch_size - n_pos)

    for _ in range(steps):
        take_pos = rng.choice(pos, size=n_pos, replace=len(pos) < n_pos)
        take_neg = rng.choice(neg, size=n_neg, replace=len(neg) < n_neg)
        batch = np.concatenate([take_pos, take_neg])
        rng.shuffle(batch)
        yield batch


def filter_camouflaged_neighbors(
    x: Any, message_edge_index: Any, threshold: float
) -> Any:
    """Drop dissimilar neighbours before aggregation (lightweight CARE-GNN).

    A simple cosine-similarity cutoff between the two endpoints' features, not the
    reinforcement-learning neighbour selector - enough to blunt camouflage for a demo.

    Args:
        x: Node feature matrix.
        message_edge_index: ``(2, n_message_edges)`` message-passing edges.
        threshold: Cosine similarity below which a neighbour edge is dropped.

    Returns:
        Boolean mask over the message edges to keep.
    """
    import torch
    import torch.nn.functional as F

    src, dst = message_edge_index[0], message_edge_index[1]
    sim = F.cosine_similarity(x[src], x[dst], dim=-1)
    return sim >= threshold


def train(bundle: GraphBundle, config: GNNConfig | None = None) -> EdgeGNN:
    """Train the edge classifier on the earlier-in-time split.

    Args:
        bundle: Built graph, whose ``train_edge_mask`` selects training edges.
        config: Hyperparameters; defaults to :class:`GNNConfig`.

    Returns:
        The trained model.
    """
    import torch

    cfg = config or GNNConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.resolved_device()
    t = _tensors(bundle, device)
    model = EdgeGNN(bundle.x.shape[1], bundle.edge_attr.shape[1], cfg)

    if cfg.camouflage_filter:
        keep = filter_camouflaged_neighbors(t["x"], t["message_edge_index"], cfg.similarity_threshold)
        kept, total = int(keep.sum()), keep.numel()
        print(f"  camouflage filter: kept {kept:,}/{total:,} message edges "
              f"({kept / max(total, 1):.1%})")
        t["message_edge_index"] = t["message_edge_index"][:, keep]
        t["message_edge_attr"] = t["message_edge_attr"][keep]

    optimizer = torch.optim.Adam(model.module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    n_pos = int(bundle.y[bundle.train_edge_mask].sum())
    print(
        f"  device={device} arch={cfg.architecture} hidden={cfg.hidden_dim} "
        f"layers={cfg.num_layers}\n"
        f"  message edges: {t['message_edge_index'].shape[1]:,} | "
        f"train targets: {int(bundle.train_edge_mask.sum()):,} ({n_pos:,} positive)\n"
        f"  balanced batches: {cfg.steps_per_epoch}/epoch x {cfg.batch_size} "
        f"@ pos_ratio={cfg.pos_ratio}"
    )

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.module.train()
        losses = []
        for batch_idx in label_balanced_sampler(
            bundle,
            bundle.train_edge_mask,
            cfg.batch_size,
            cfg.pos_ratio,
            cfg.steps_per_epoch,
            seed=cfg.seed + epoch,
        ):
            idx = torch.from_numpy(batch_idx).long().to(device)
            optimizer.zero_grad()
            logits = model.module(
                t["x"],
                t["message_edge_index"],
                t["message_edge_attr"],
                t["edge_index"][:, idx],
                t["edge_attr"][idx],
            )
            loss = loss_fn(logits, t["y"][idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{cfg.epochs}  loss {np.mean(losses):.4f}  "
                  f"({time.time() - t0:.0f}s)")

    print(f"  trained in {time.time() - t0:.1f}s")
    return model


def predict(
    model: EdgeGNN, bundle: GraphBundle, edge_mask: np.ndarray, chunk: int = 100_000
) -> np.ndarray:
    """Score edges with a trained GNN.

    Node embeddings come from message passing over the training edges only, so a held-out
    transaction never informs its own prediction.

    Args:
        model: Trained model.
        bundle: Built graph.
        edge_mask: Which edges to score.
        chunk: Target edges scored per forward pass, to bound memory.

    Returns:
        Risk scores in [0, 1], one per selected edge, aligned with the mask's edge order.
    """
    import torch

    device = model.device
    t = _tensors(bundle, device)
    idx = np.flatnonzero(edge_mask)

    model.module.eval()
    out = np.empty(len(idx), dtype=np.float32)
    with torch.no_grad():
        h = model.module.encode(t["x"], t["message_edge_index"], t["message_edge_attr"])
        for start in range(0, len(idx), chunk):
            piece = torch.from_numpy(idx[start : start + chunk]).long().to(device)
            logits = model.module.classify(h, t["edge_index"][:, piece], t["edge_attr"][piece])
            out[start : start + chunk] = torch.sigmoid(logits).cpu().numpy()
    return out


def evaluate(
    model: EdgeGNN, bundle: GraphBundle, cost_model: CostModel | None = None
) -> dict[str, Any]:
    """Score the held-out split with the same metrics as the baseline.

    Args:
        model: Trained model.
        bundle: Built graph, whose ``test_edge_mask`` selects held-out edges.
        cost_model: Unit costs for the false-positive-cost estimate.

    Returns:
        The bundle from ``src.eval.metrics.evaluate_model``.
    """
    scores = predict(model, bundle, bundle.test_edge_mask)
    y_true = bundle.y[bundle.test_edge_mask].astype(int)
    return evaluate_model("gnn", y_true, scores, cost_model)


def explain_edge(
    model: EdgeGNN, bundle: GraphBundle, edge_id: int, top_k: int = 10
) -> dict[str, Any]:
    """Attribute one edge's score to contributing features and neighbouring accounts.

    Gradient-times-input attribution over the transaction's own features and its two endpoints'
    node features, plus the endpoints' one-hop neighbours in the flagged history. Feeds the audit
    trail: the API must be able to say *which* accounts, edges, and features drove a flag
    (``PRD.md`` section 8).

    Args:
        model: Trained model.
        bundle: Built graph.
        edge_id: Index of the transaction edge to explain.
        top_k: How many contributors to return.

    Returns:
        ``{"score", "top_features", "contributing_accounts", "contributing_edges"}``.
    """
    import torch

    device = model.device
    t = _tensors(bundle, device)
    src = int(bundle.edge_index[0, edge_id])
    dst = int(bundle.edge_index[1, edge_id])

    model.module.eval()
    edge_attr = t["edge_attr"][edge_id : edge_id + 1].clone().requires_grad_(True)
    x = t["x"].clone().requires_grad_(True)
    h = model.module.encode(x, t["message_edge_index"], t["message_edge_attr"])
    logit = model.module.classify(h, t["edge_index"][:, edge_id : edge_id + 1], edge_attr)
    logit.backward()

    edge_contrib = (edge_attr.grad[0] * edge_attr[0]).detach().cpu().numpy()
    node_contrib = (x.grad * x).detach().cpu().numpy()

    features = [
        {"feature": n, "value": float(bundle.edge_attr[edge_id, i]), "contribution": float(edge_contrib[i])}
        for i, n in enumerate(bundle.edge_feature_names)
    ]
    for node, tag in ((src, "src"), (dst, "dst")):
        features += [
            {
                "feature": f"{tag}_{n}",
                "value": float(bundle.x[node, i]),
                "contribution": float(node_contrib[node, i]),
            }
            for i, n in enumerate(bundle.node_feature_names)
        ]
    features.sort(key=lambda f: abs(f["contribution"]), reverse=True)

    neighbours = np.unique(
        np.concatenate(
            [
                bundle.edge_index[1][bundle.edge_index[0] == src],
                bundle.edge_index[0][bundle.edge_index[1] == dst],
            ]
        )
    )
    return {
        "edge_id": int(edge_id),
        "score": float(torch.sigmoid(logit).item()),
        "from_account": str(bundle.accounts[src]),
        "to_account": str(bundle.accounts[dst]),
        "top_features": features[:top_k],
        "contributing_accounts": [str(bundle.accounts[n]) for n in neighbours[:top_k]],
    }


def save(model: EdgeGNN, path: Path) -> None:
    """Persist a trained GNN's weights and config.

    Args:
        model: Trained model.
        path: Destination file (gitignored).
    """
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.module.state_dict(),
            "config": asdict(model.config),
            "node_dim": model.node_dim,
            "edge_dim": model.edge_dim,
        },
        path,
    )
    print(f"saved gnn -> {path}")


def load(path: Path, device: str = "auto") -> EdgeGNN:
    """Load a GNN written by :func:`save`.

    Args:
        path: File written by :func:`save`.
        device: Device to load onto.

    Returns:
        The restored model.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    import torch

    if not path.exists():
        raise FileNotFoundError(
            f"No GNN at {path}. Train it first: python -m src.models.gnn --train"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GNNConfig(**payload["config"])
    if device != "auto":
        cfg.device = device
    model = EdgeGNN(payload["node_dim"], payload["edge_dim"], cfg)
    model.module.load_state_dict(payload["state_dict"])
    return model


def main() -> None:
    """CLI entry point: ``python -m src.models.gnn --train``."""
    parser = argparse.ArgumentParser(description="Train/evaluate the RingWatch edge GNN.")
    parser.add_argument("--train", action="store_true", help="Train (otherwise load and evaluate).")
    parser.add_argument("--graph", type=Path, default=DATA_DIR / "graph.pkl")
    parser.add_argument("--architecture", default="gine", choices=["gine", "sage"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--pos-ratio", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--camouflage-filter", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=Path, default=DATA_DIR / "gnn.pt")
    parser.add_argument("--scores-out", type=Path, default=DATA_DIR / "scores_gnn.npz")
    args = parser.parse_args()

    print(f"loading graph from {args.graph} ...")
    bundle = load_graph(args.graph)
    print(bundle.summary())

    cfg = GNNConfig(
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        pos_ratio=args.pos_ratio,
        lr=args.lr,
        camouflage_filter=args.camouflage_filter,
        device=args.device,
        seed=args.seed,
    )

    if args.train:
        print("\ntraining GNN ...")
        model = train(bundle, cfg)
        save(model, args.model_out)
    else:
        model = load(args.model_out, args.device)

    print("\nevaluating on the held-out (later-in-time) split ...")
    result = evaluate(model, bundle)
    m, mb, cb = result["metrics"], result["metrics_at_best"], result["cost_at_best"]
    print(
        f"  PR-AUC        : {m['pr_auc']:.4f}\n"
        f"  ROC-AUC       : {m['roc_auc']:.4f}\n"
        f"  @0.50  P/R/F1 : {m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f}\n"
        f"  cost-opt thr  : {result['best_threshold']:.4f}\n"
        f"  @opt   P/R    : {mb['precision']:.3f} / {mb['recall']:.3f}\n"
        f"  net cost      : {cb['net_cost']:,.0f} "
        f"(do-nothing {cb['baseline_do_nothing_cost']:,.0f}, "
        f"saved {cb['cost_saved_vs_do_nothing']:,.0f})"
    )

    all_scores = predict(model, bundle, np.ones(bundle.n_edges, dtype=bool))
    np.savez_compressed(
        args.scores_out, scores=all_scores, y=bundle.y, test_mask=bundle.test_edge_mask
    )
    print(f"\nsaved edge scores -> {args.scores_out}")

    (DATA_DIR / "metrics_gnn.json").write_text(
        json.dumps({**result, "config": asdict(cfg)}, indent=2, default=float), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
