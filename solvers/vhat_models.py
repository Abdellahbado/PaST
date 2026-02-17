"""Value function approximation models: Polynomial Ridge, MLP, and LightGBM.

All models implement predict_from_used() for compatibility with the beam DP
evaluation pipeline. They share the same feature extraction (phi_for_state)
and differ only in how they map features → predicted cost-to-go.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PaST.solvers.vhat_linear import FeatureSpec, phi_for_state
from PaST.solvers.vhat_tou_features import TOUFeatureContext


# ============================================================================
#  1. Polynomial Ridge (degree-2)
# ============================================================================


@dataclass
class PolyRidgeValueModel:
    """Ridge regression on degree-2 polynomial features.

    Features: 18 raw → ~171 after polynomial expansion (bias + linear + interactions + squares).
    Training: closed-form ridge (same as linear but higher-dim).
    Inference: polynomial expansion is cheap numpy vectorized ops.
    """

    weights: np.ndarray  # shape (D_poly,)
    spec: FeatureSpec
    powers_: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )  # (n_output_features, n_input_features)
    H: int = 20

    def predict_from_used(
        self,
        *,
        t: int,
        used: Sequence[int],
        totals: np.ndarray,
        lengths: Sequence[int],
        ctx: TOUFeatureContext,
    ) -> float:
        x = phi_for_state(
            t=t, used=used, totals=totals, lengths=lengths, ctx=ctx, spec=self.spec
        )
        x_poly = self._poly_transform_single(x)
        return float(np.dot(self.weights, x_poly))

    def _poly_transform_single(self, x: np.ndarray) -> np.ndarray:
        """Manual degree-2 polynomial expansion for a single sample (no sklearn at inference)."""
        return _poly_expand_single(x, self.powers_)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            weights=self.weights,
            powers=self.powers_,
            include_per_class_counts=int(self.spec.include_per_class_counts),
            include_per_class_now_cost=int(self.spec.include_per_class_now_cost),
            include_bins=int(self.spec.include_bins),
            normalize=int(self.spec.normalize),
            include_len_hist=int(self.spec.include_len_hist),
            pmax_for_hist=int(self.spec.pmax_for_hist),
            include_price_shape=int(self.spec.include_price_shape),
            include_meta=int(self.spec.include_meta),
            model_type="poly",
        )

    @staticmethod
    def load(path: str) -> "PolyRidgeValueModel":
        ckpt = np.load(path, allow_pickle=True)
        spec = FeatureSpec(
            include_per_class_counts=bool(int(ckpt["include_per_class_counts"])),
            include_per_class_now_cost=bool(int(ckpt["include_per_class_now_cost"])),
            include_bins=bool(int(ckpt["include_bins"])),
            normalize=bool(int(ckpt["normalize"])),
            include_len_hist=bool(int(ckpt["include_len_hist"]))
            if "include_len_hist" in ckpt.files
            else False,
            pmax_for_hist=int(ckpt["pmax_for_hist"]) if "pmax_for_hist" in ckpt.files else 12,
            include_price_shape=bool(int(ckpt["include_price_shape"]))
            if "include_price_shape" in ckpt.files
            else False,
            include_meta=bool(int(ckpt["include_meta"]))
            if "include_meta" in ckpt.files
            else False,
        )
        return PolyRidgeValueModel(
            weights=np.asarray(ckpt["weights"], dtype=np.float64),
            spec=spec,
            powers_=np.asarray(ckpt["powers"], dtype=np.int32),
        )


def _poly_expand_single(x: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Expand a single feature vector using precomputed power matrix.

    powers: shape (n_output_features, n_input_features), each row is exponents.
    Returns: shape (n_output_features,)
    """
    # x^powers[i] for each output feature i, then product across input features
    # For degree-2, exponents are 0, 1, or 2 — so this is just multiplications
    return np.prod(x**powers, axis=1)


def _poly_expand_batch(X: np.ndarray, powers: np.ndarray) -> np.ndarray:
    """Expand a batch of feature vectors. X: (N, d_in) -> (N, d_out).

    This is optimized for the degree-2, include_bias=True powers matrix produced
    by this module (and compatible with sklearn's PolynomialFeatures ordering).
    """

    X = np.asarray(X, dtype=np.float64)
    powers = np.asarray(powers, dtype=np.int32)
    N, d_in = X.shape
    d_out = int(powers.shape[0])
    out = np.empty((N, d_out), dtype=np.float64)

    row_sums = powers.sum(axis=1)
    row_max = powers.max(axis=1)
    nnz = (powers != 0).sum(axis=1)

    # bias
    bias_rows = np.where(row_sums == 0)[0]
    if bias_rows.size:
        out[:, bias_rows] = 1.0

    # linear: exactly one 1
    lin_rows = np.where((row_sums == 1) & (row_max == 1) & (nnz == 1))[0]
    for r in lin_rows.tolist():
        j = int(np.argmax(powers[r]))
        out[:, r] = X[:, j]

    # squares: exactly one 2
    sq_rows = np.where((row_sums == 2) & (row_max == 2) & (nnz == 1))[0]
    for r in sq_rows.tolist():
        j = int(np.argmax(powers[r]))
        col = X[:, j]
        out[:, r] = col * col

    # interactions: exactly two 1s
    inter_rows = np.where((row_sums == 2) & (row_max == 1) & (nnz == 2))[0]
    for r in inter_rows.tolist():
        js = np.flatnonzero(powers[r])
        i0 = int(js[0])
        i1 = int(js[1])
        out[:, r] = X[:, i0] * X[:, i1]

    return out


def _poly_powers_degree2(d_in: int) -> np.ndarray:
    """Create sklearn-compatible powers_ for degree-2 PolynomialFeatures.

    Ordering matches sklearn.preprocessing.PolynomialFeatures(degree=2,
    include_bias=True, interaction_only=False).
    """
    d = int(d_in)
    powers: List[np.ndarray] = []

    # bias
    powers.append(np.zeros(d, dtype=np.int32))

    # linear
    for i in range(d):
        e = np.zeros(d, dtype=np.int32)
        e[i] = 1
        powers.append(e)

    # degree-2: squares and interactions in sklearn order
    for i in range(d):
        e2 = np.zeros(d, dtype=np.int32)
        e2[i] = 2
        powers.append(e2)
        for j in range(i + 1, d):
            e = np.zeros(d, dtype=np.int32)
            e[i] = 1
            e[j] = 1
            powers.append(e)

    return np.stack(powers, axis=0)


def fit_poly_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1e-3,
    degree: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit polynomial ridge. Returns (weights, powers_matrix).

    Notes:
    - We implement degree-2 PolynomialFeatures internally to avoid requiring
      scikit-learn as a dependency.
    """
    if int(degree) != 2:
        raise ValueError("Only degree=2 is supported without scikit-learn")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    powers = _poly_powers_degree2(int(X.shape[1]))
    X_poly = _poly_expand_batch(X, powers)

    D = X_poly.shape[1]
    A = X_poly.T @ X_poly + l2 * np.eye(D, dtype=np.float64)
    b = X_poly.T @ y
    w = np.linalg.solve(A, b).astype(np.float64)

    return w, powers


# ============================================================================
#  2. Small MLP (numpy inference)
# ============================================================================


@dataclass
class MLPValueModel:
    """Small MLP value function. Trained with PyTorch, inference with numpy.

    Architecture: input → 64 → 32 → 1 (ReLU activations).
    Inference is two matrix multiplies + ReLU, ~2μs per call.
    """

    W1: np.ndarray  # (d_in, 64)
    b1: np.ndarray  # (64,)
    W2: np.ndarray  # (64, 32)
    b2: np.ndarray  # (32,)
    W3: np.ndarray  # (32, 1)
    b3: np.ndarray  # (1,)
    spec: FeatureSpec
    H: int = 20

    def predict_from_used(
        self,
        *,
        t: int,
        used: Sequence[int],
        totals: np.ndarray,
        lengths: Sequence[int],
        ctx: TOUFeatureContext,
    ) -> float:
        x = phi_for_state(
            t=t, used=used, totals=totals, lengths=lengths, ctx=ctx, spec=self.spec
        )
        # Forward pass with numpy (no PyTorch overhead)
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        y = (h2 @ self.W3) + self.b3
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        return float(y[0])

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            include_per_class_counts=int(self.spec.include_per_class_counts),
            include_per_class_now_cost=int(self.spec.include_per_class_now_cost),
            include_bins=int(self.spec.include_bins),
            normalize=int(self.spec.normalize),
            include_len_hist=int(self.spec.include_len_hist),
            pmax_for_hist=int(self.spec.pmax_for_hist),
            include_price_shape=int(self.spec.include_price_shape),
            include_meta=int(self.spec.include_meta),
            model_type="mlp",
        )

    @staticmethod
    def load(path: str) -> "MLPValueModel":
        ckpt = np.load(path, allow_pickle=True)
        spec = FeatureSpec(
            include_per_class_counts=bool(int(ckpt["include_per_class_counts"])),
            include_per_class_now_cost=bool(int(ckpt["include_per_class_now_cost"])),
            include_bins=bool(int(ckpt["include_bins"])),
            normalize=bool(int(ckpt["normalize"])),
            include_len_hist=bool(int(ckpt["include_len_hist"]))
            if "include_len_hist" in ckpt.files
            else False,
            pmax_for_hist=int(ckpt["pmax_for_hist"]) if "pmax_for_hist" in ckpt.files else 12,
            include_price_shape=bool(int(ckpt["include_price_shape"]))
            if "include_price_shape" in ckpt.files
            else False,
            include_meta=bool(int(ckpt["include_meta"]))
            if "include_meta" in ckpt.files
            else False,
        )
        W3 = np.asarray(ckpt["W3"], dtype=np.float64)
        b3 = np.asarray(ckpt["b3"], dtype=np.float64)
        if W3.ndim == 1:
            W3 = W3.reshape(-1, 1)
        b3 = b3.reshape(-1)
        if b3.size == 0:
            b3 = np.asarray([0.0], dtype=np.float64)

        return MLPValueModel(
            W1=np.asarray(ckpt["W1"], dtype=np.float64),
            b1=np.asarray(ckpt["b1"], dtype=np.float64),
            W2=np.asarray(ckpt["W2"], dtype=np.float64),
            b2=np.asarray(ckpt["b2"], dtype=np.float64),
            W3=W3,
            b3=b3,
            spec=spec,
        )


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden1: int = 64,
    hidden2: int = 32,
    lr: float = 1e-3,
    batch_size: int = 2048,
    max_epochs: int = 200,
    patience: int = 15,
    device: str = "auto",
) -> MLPValueModel:
    """Train a small MLP with PyTorch, return numpy-inference model."""
    import torch
    import torch.nn as nn

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"  [mlp] Training on device={device}")

    d_in = X_train.shape[1]

    # Standardize features for better training
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    X_t = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_v = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
    y_v = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    model = nn.Sequential(
        nn.Linear(d_in, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    N = X_t.shape[0]
    n_batches = max(1, N // batch_size)

    for epoch in range(max_epochs):
        # Shuffle
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        model.train()
        for bi in range(n_batches):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            pred = model(X_t[idx])
            loss = nn.functional.mse_loss(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = nn.functional.mse_loss(val_pred, y_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0 or wait == 0:
            print(
                f"  [mlp] epoch={epoch+1:3d}  train_loss={epoch_loss:.6f}  "
                f"val_loss={val_loss:.6f}  best={best_val_loss:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

        if wait >= patience:
            print(f"  [mlp] Early stopping at epoch {epoch+1}")
            break

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    # Extract to numpy — bake input standardization into first layer
    with torch.no_grad():
        sd = model.state_dict()
        W1_raw = sd["0.weight"].cpu().numpy().astype(np.float64)  # (hidden1, d_in)
        b1_raw = sd["0.bias"].cpu().numpy().astype(np.float64)  # (hidden1,)
        W2_raw = sd["2.weight"].cpu().numpy().astype(np.float64)  # (hidden2, hidden1)
        b2_raw = sd["2.bias"].cpu().numpy().astype(np.float64)  # (hidden2,)
        W3_raw = sd["4.weight"].cpu().numpy().astype(np.float64)  # (1, hidden2)
        b3_raw = sd["4.bias"].cpu().numpy().astype(np.float64)  # (1,)

    # Bake standardization into W1, b1:
    #   h1 = W1_raw @ ((x - mean) / std) + b1_raw
    #      = (W1_raw / std) @ x + (b1_raw - W1_raw @ (mean / std))
    X_std_64 = X_std.astype(np.float64)
    X_mean_64 = X_mean.astype(np.float64)
    W1_baked = W1_raw / X_std_64[None, :]  # (hidden1, d_in)
    b1_baked = b1_raw - W1_raw @ (X_mean_64 / X_std_64)  # (hidden1,)

    # Transpose for x @ W format (d_in, hidden1) instead of (hidden1, d_in)
    return MLPValueModel(
        W1=W1_baked.T,  # (d_in, hidden1)
        b1=b1_baked,  # (hidden1,)
        W2=W2_raw.T,  # (hidden1, hidden2)
        b2=b2_raw,  # (hidden2,)
        W3=W3_raw.T,  # (hidden2, 1)
        b3=b3_raw,  # (1,)
        spec=FeatureSpec(),  # will be set by caller
    )


# ============================================================================
#  3. LightGBM
# ============================================================================


@dataclass
class LGBMValueModel:
    """LightGBM gradient boosted trees for value function approximation.

    Uses 100 trees, max_depth=5, n_jobs=-1 for parallel training.
    Inference: ~10μs per call (tree traversal).
    """

    booster: Any  # lightgbm.Booster
    spec: FeatureSpec
    H: int = 20

    def predict_from_used(
        self,
        *,
        t: int,
        used: Sequence[int],
        totals: np.ndarray,
        lengths: Sequence[int],
        ctx: TOUFeatureContext,
    ) -> float:
        x = phi_for_state(
            t=t, used=used, totals=totals, lengths=lengths, ctx=ctx, spec=self.spec
        )
        pred = self.booster.predict(x.reshape(1, -1))
        return float(pred[0])

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Save booster as txt and metadata as json
        booster_path = str(p) + ".lgbm"
        self.booster.save_model(booster_path)
        meta = {
            "include_per_class_counts": int(self.spec.include_per_class_counts),
            "include_per_class_now_cost": int(self.spec.include_per_class_now_cost),
            "include_bins": int(self.spec.include_bins),
            "normalize": int(self.spec.normalize),
            "model_type": "lgbm",
            "booster_path": booster_path,
        }
        meta_path = str(p) + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        # Also save as npz for compatibility with load detection
        np.savez(
            p,
            model_type="lgbm",
            include_per_class_counts=int(self.spec.include_per_class_counts),
            include_per_class_now_cost=int(self.spec.include_per_class_now_cost),
            include_bins=int(self.spec.include_bins),
            normalize=int(self.spec.normalize),
        )

    @staticmethod
    def load(path: str) -> "LGBMValueModel":
        import lightgbm as lgb

        meta_path = str(path) + ".meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        spec = FeatureSpec(
            include_per_class_counts=bool(int(meta["include_per_class_counts"])),
            include_per_class_now_cost=bool(int(meta["include_per_class_now_cost"])),
            include_bins=bool(int(meta["include_bins"])),
            normalize=bool(int(meta["normalize"])),
        )
        booster = lgb.Booster(model_file=meta["booster_path"])
        return LGBMValueModel(booster=booster, spec=spec)


def fit_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    n_jobs: int = -1,
    verbose: int = 10,
) -> Any:
    """Fit a LightGBM regressor. Returns the booster."""
    import lightgbm as lgb

    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, free_raw_data=False
    )

    params = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 2**max_depth - 1,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_jobs": n_jobs,
        "verbose": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[val_data],
        callbacks=[
            lgb.log_evaluation(period=verbose),
            lgb.early_stopping(stopping_rounds=20, verbose=True),
        ],
    )

    return booster
