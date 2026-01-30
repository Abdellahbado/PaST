"""Train a tiny imitation model to predict ALNS operator choices.

This is a *go/no-go* diagnostic for PPO-style learning:
- If the oracle-best operator is predictable from features, PPO has a chance.
- If it is not predictable (near-random accuracy and low expected advantage),
  PPO will likely not learn meaningful patterns.

Input: NPZ files produced by `compare_alns_vs_baseline_pm.py --oracle_dataset`
  - X: (N, F) state features
  - y: (N,) best action index
  - deltas: (N, A) delta_energy for each action (lower is better)

Output:
- accuracy / top-k accuracy
- expected improvement of policy vs random using stored deltas

Example:
  PYTHONPATH=. python -m PaST.cli.analysis.train_alns_operator_imitation \
    --npz analysis_out/alns_vs_baseline_oracle_inst0_seed42.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_frac", type=float, default=0.25)
    return p


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _split_indices(
    n: int, test_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(float(test_frac) * n))
    n_test = max(1, min(n - 1, n_test))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def _topk_acc(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
    ok = (topk == y.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(ok)


def main() -> None:
    args = build_parser().parse_args()

    path = Path(args.npz)
    data = np.load(str(path), allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    deltas = data["deltas"].astype(np.float32)

    meta = {}
    if "meta" in data:
        meta = json.loads(str(data["meta"]))

    n, f = X.shape
    a = deltas.shape[1]

    train_idx, test_idx = _split_indices(n, float(args.test_frac), int(args.seed))

    device = torch.device("cpu")
    torch.manual_seed(int(args.seed))

    model = MLP(in_dim=f, hidden=int(args.hidden), out_dim=a).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = torch.nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    for ep in range(int(args.epochs)):
        model.train()
        logits = model(X_t[train_idx])
        loss = loss_fn(logits, y_t[train_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits_te = model(X_t[test_idx])
        y_te = y_t[test_idx]
        acc1 = (logits_te.argmax(dim=1) == y_te).float().mean().item()
        acc3 = _topk_acc(logits_te, y_te, k=3)

        # Policy value estimate using stored deltas:
        # reward = -delta_energy (higher is better)
        pred_actions = logits_te.argmax(dim=1).cpu().numpy()
        deltas_te = deltas[test_idx]

        # Random policy expected delta
        rng = np.random.default_rng(int(args.seed))
        rand_actions = rng.integers(0, a, size=len(test_idx))

        pred_delta = deltas_te[np.arange(len(test_idx)), pred_actions]
        rand_delta = deltas_te[np.arange(len(test_idx)), rand_actions]
        oracle_delta = deltas_te.min(axis=1)

        # Convert to reward
        pred_r = -pred_delta
        rand_r = -rand_delta
        oracle_r = -oracle_delta

        print("Dataset")
        print(f"- file: {path}")
        print(f"- N={n} features={f} actions={a}")
        if meta:
            print(f"- meta: {meta}")

        print("Imitation")
        print(f"- test accuracy@1: {float(acc1):.3f}")
        print(f"- test accuracy@3: {float(acc3):.3f}")

        print("Headroom (test set)")
        print(f"- mean delta (pred):  {float(pred_delta.mean()):.4f}")
        print(f"- mean delta (rand):  {float(rand_delta.mean()):.4f}")
        print(f"- mean delta (oracle): {float(oracle_delta.mean()):.4f}")
        print(f"- mean reward gain pred-rand: {float((pred_r - rand_r).mean()):.4f}")
        print(f"- mean reward gap to oracle:  {float((oracle_r - pred_r).mean()):.4f}")


if __name__ == "__main__":
    main()
