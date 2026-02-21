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
#  2b. Poly-MLP (numpy inference)
# ============================================================================


@dataclass
class PolyMLPValueModel:
    """Poly-MLP value function. Trained with PyTorch, inference with numpy.

    Expands inputs to standard O(D^2) polynomial cross-terms, then feeds to MLP.
    Combines explicit interaction inductive bias with universal approximation.
    Architecture: input → poly_expand → 64 → 32 → 1 (ReLU activations).
    """

    powers: np.ndarray  # shape (d_poly, d_in)
    W1: np.ndarray  # (d_poly, 64)
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
        x_poly = _poly_expand_batch(x[None, :], self.powers)[0]
        # Forward pass with numpy (no PyTorch overhead)
        h1 = np.maximum(0, x_poly @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        y = (h2 @ self.W3) + self.b3
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        return float(y[0])

    def save(self, path: str) -> None:
        np.savez(
            path,
            powers=self.powers,
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
            model_type="poly_mlp",
        )

    @staticmethod
    def load(path: str) -> "PolyMLPValueModel":
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

        return PolyMLPValueModel(
            powers=np.asarray(ckpt["powers"], dtype=np.int32),
            W1=np.asarray(ckpt["W1"], dtype=np.float64),
            b1=np.asarray(ckpt["b1"], dtype=np.float64),
            W2=np.asarray(ckpt["W2"], dtype=np.float64),
            b2=np.asarray(ckpt["b2"], dtype=np.float64),
            W3=W3,
            b3=b3,
            spec=spec,
        )


def fit_poly_mlp(
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
) -> PolyMLPValueModel:
    """Train a Poly-MLP model with PyTorch, return numpy-inference model."""
    import torch
    import torch.nn as nn

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    d_in_raw = X_train.shape[1]
    powers = _poly_powers_degree2(d_in_raw)
    
    print(f"  [poly_mlp] Expanding features ...")
    X_train_poly = _poly_expand_batch(X_train, powers)
    X_val_poly = _poly_expand_batch(X_val, powers)

    print(f"  [poly_mlp] Training on device={device}, {d_in_raw} -> {X_train_poly.shape[1]} dims")

    d_in = X_train_poly.shape[1]

    # Standardize features for better training
    X_mean = X_train_poly.mean(axis=0)
    X_std = X_train_poly.std(axis=0) + 1e-8
    
    X_train_norm = (X_train_poly - X_mean) / X_std
    X_val_norm = (X_val_poly - X_mean) / X_std

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

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = nn.functional.mse_loss(val_pred, y_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            if epoch % 5 == 0 or epoch == max_epochs - 1:
                print(
                    f"  [poly_mlp] epoch={epoch+1:3d}  train_loss={epoch_loss:.6f}  "
                    f"val_loss={val_loss:.6f}  best={best_val_loss:.6f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.1e}"
                )
        else:
            wait += 1
            if wait >= patience:
                print(f"  [poly_mlp] Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)

    W1_raw = model[0].weight.detach().cpu().numpy()
    b1_raw = model[0].bias.detach().cpu().numpy()
    W2_raw = model[2].weight.detach().cpu().numpy()
    b2_raw = model[2].bias.detach().cpu().numpy()
    W3_raw = model[4].weight.detach().cpu().numpy()
    b3_raw = model[4].bias.detach().cpu().numpy()

    # Bake standardization into W1, b1
    X_std_64 = X_std.astype(np.float64)
    X_mean_64 = X_mean.astype(np.float64)
    W1_baked = W1_raw / X_std_64[None, :]
    b1_baked = b1_raw - W1_raw @ (X_mean_64 / X_std_64)

    return PolyMLPValueModel(
        powers=powers,
        W1=W1_baked.T,
        b1=b1_baked,
        W2=W2_raw.T,
        b2=b2_raw,
        W3=W3_raw.T,
        b3=b3_raw,
        spec=FeatureSpec(),
    )


# ============================================================================
#  2c. Factored-Interaction MLP (numpy inference)
# ============================================================================


# Default feature group indices for the standard 18-feature phi_for_state
# with spec=(bins=True, normalize=True, per_class_counts=False,
#            per_class_now_cost=False, include_len_hist=False,
#            include_price_shape=False, include_meta=False).
#
# Features layout (0-indexed):
#   0: bias (1.0)
#   1-3: regime one-hot (off, shoulder, peak)
#   4: dist_to_next_off
#   5: dist_to_next_cheap
#   6: N/T
#   7: W/T
#   8: R/T
#   9: S_pos/T
#  10: S_pos * reg_off   (slack × regime interaction)
#  11: S_pos * reg_peak   (slack × regime interaction)
#  12: c_off/T
#  13: c_peak/T
#  14: pressure_off
#  15: pressure_cheap
#  16: short/T
#  17: long/T

_DEFAULT_WORK_IDX = np.array([6, 7, 8, 9, 10, 11, 16, 17], dtype=np.int32)
_DEFAULT_PRICE_IDX = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15], dtype=np.int32)


@dataclass
class FactoredMLPValueModel:
    """Factored-interaction MLP for value function approximation.

    Splits features into workload and price-regime groups, encodes each
    through a small dense layer, then combines via:
        combined = [h_work ; h_price ; h_work ⊙ h_price]
    where ⊙ is element-wise (Hadamard) product.

    This explicitly encodes the multiplicative work × price structure
    of the cost-to-go without extra learnable parameters for the
    interaction, while still allowing nonlinear within-group encodings.

    Inference: ~2-3μs per call (numpy only, no PyTorch).
    """

    # Work encoder: x_work → h_work
    W_work: np.ndarray   # (d_work, h_dim)
    b_work: np.ndarray   # (h_dim,)
    # Price encoder: x_price → h_price
    W_price: np.ndarray  # (d_price, h_dim)
    b_price: np.ndarray  # (h_dim,)
    # Final layers: combined → output
    W_final1: np.ndarray  # (3 * h_dim, d_final)
    b_final1: np.ndarray  # (d_final,)
    W_final2: np.ndarray  # (d_final, 1)
    b_final2: np.ndarray  # (1,)
    # Feature group indices
    work_idx: np.ndarray   # indices into phi vector for work features
    price_idx: np.ndarray  # indices into phi vector for price features

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
        # Split into groups
        x_work = x[self.work_idx]
        x_price = x[self.price_idx]

        # Encode each group
        h_work = np.maximum(0, x_work @ self.W_work + self.b_work)
        h_price = np.maximum(0, x_price @ self.W_price + self.b_price)

        # Combine: concat + Hadamard product
        combined = np.concatenate([h_work, h_price, h_work * h_price])

        # Final prediction
        h = np.maximum(0, combined @ self.W_final1 + self.b_final1)
        y = (h @ self.W_final2) + self.b_final2
        return float(np.asarray(y, dtype=np.float64).reshape(-1)[0])

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            W_work=self.W_work, b_work=self.b_work,
            W_price=self.W_price, b_price=self.b_price,
            W_final1=self.W_final1, b_final1=self.b_final1,
            W_final2=self.W_final2, b_final2=self.b_final2,
            work_idx=self.work_idx, price_idx=self.price_idx,
            include_per_class_counts=int(self.spec.include_per_class_counts),
            include_per_class_now_cost=int(self.spec.include_per_class_now_cost),
            include_bins=int(self.spec.include_bins),
            normalize=int(self.spec.normalize),
            include_len_hist=int(self.spec.include_len_hist),
            pmax_for_hist=int(self.spec.pmax_for_hist),
            include_price_shape=int(self.spec.include_price_shape),
            include_meta=int(self.spec.include_meta),
            model_type="factored_mlp",
        )

    @staticmethod
    def load(path: str) -> "FactoredMLPValueModel":
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
        W_final2 = np.asarray(ckpt["W_final2"], dtype=np.float64)
        b_final2 = np.asarray(ckpt["b_final2"], dtype=np.float64)
        if W_final2.ndim == 1:
            W_final2 = W_final2.reshape(-1, 1)
        b_final2 = b_final2.reshape(-1)
        if b_final2.size == 0:
            b_final2 = np.asarray([0.0], dtype=np.float64)

        return FactoredMLPValueModel(
            W_work=np.asarray(ckpt["W_work"], dtype=np.float64),
            b_work=np.asarray(ckpt["b_work"], dtype=np.float64),
            W_price=np.asarray(ckpt["W_price"], dtype=np.float64),
            b_price=np.asarray(ckpt["b_price"], dtype=np.float64),
            W_final1=np.asarray(ckpt["W_final1"], dtype=np.float64),
            b_final1=np.asarray(ckpt["b_final1"], dtype=np.float64),
            W_final2=W_final2,
            b_final2=b_final2,
            work_idx=np.asarray(ckpt["work_idx"], dtype=np.int32),
            price_idx=np.asarray(ckpt["price_idx"], dtype=np.int32),
            spec=spec,
        )


def fit_factored_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    work_idx: np.ndarray | None = None,
    price_idx: np.ndarray | None = None,
    h_dim: int = 16,
    d_final: int = 24,
    lr: float = 1e-3,
    batch_size: int = 2048,
    max_epochs: int = 600,
    patience: int = 20,
    device: str = "auto",
) -> FactoredMLPValueModel:
    """Train factored-interaction MLP with PyTorch, return numpy-inference model.

    The architecture splits input features into workload and price-regime groups,
    encodes each through a dense layer, then combines via Hadamard interaction:
        h_work = ReLU(x_work @ W_work + b_work)
        h_price = ReLU(x_price @ W_price + b_price)
        combined = [h_work ; h_price ; h_work ⊙ h_price]
        output = W_final2 @ ReLU(W_final1 @ combined + b1) + b2
    """
    import torch
    import torch.nn as nn

    if work_idx is None:
        work_idx = _DEFAULT_WORK_IDX.copy()
    if price_idx is None:
        price_idx = _DEFAULT_PRICE_IDX.copy()

    work_idx = np.asarray(work_idx, dtype=np.int32)
    price_idx = np.asarray(price_idx, dtype=np.int32)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"  [factored_mlp] Training on device={device}")

    d_work = len(work_idx)
    d_price = len(price_idx)
    d_combined = 3 * h_dim  # [h_work ; h_price ; h_work ⊙ h_price]

    # Extract feature groups
    X_work_train = X_train[:, work_idx]
    X_price_train = X_train[:, price_idx]
    X_work_val = X_val[:, work_idx]
    X_price_val = X_val[:, price_idx]

    # Per-group standardization
    work_mean = X_work_train.mean(axis=0)
    work_std = X_work_train.std(axis=0) + 1e-8
    price_mean = X_price_train.mean(axis=0)
    price_std = X_price_train.std(axis=0) + 1e-8

    X_work_train_n = (X_work_train - work_mean) / work_std
    X_price_train_n = (X_price_train - price_mean) / price_std
    X_work_val_n = (X_work_val - work_mean) / work_std
    X_price_val_n = (X_price_val - price_mean) / price_std

    # Convert to tensors
    Xw_t = torch.tensor(X_work_train_n, dtype=torch.float32, device=device)
    Xp_t = torch.tensor(X_price_train_n, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    Xw_v = torch.tensor(X_work_val_n, dtype=torch.float32, device=device)
    Xp_v = torch.tensor(X_price_val_n, dtype=torch.float32, device=device)
    y_v = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    # Build model using nn.Module for clarity
    class FactoredNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.work_enc = nn.Linear(d_work, h_dim)
            self.price_enc = nn.Linear(d_price, h_dim)
            self.final1 = nn.Linear(d_combined, d_final)
            self.final2 = nn.Linear(d_final, 1)

        def forward(self, x_work, x_price):
            h_work = torch.relu(self.work_enc(x_work))
            h_price = torch.relu(self.price_enc(x_price))
            combined = torch.cat([h_work, h_price, h_work * h_price], dim=-1)
            h = torch.relu(self.final1(combined))
            return self.final2(h)

    model = FactoredNet().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [factored_mlp] Parameters: {n_params}")
    print(f"  [factored_mlp] d_work={d_work} d_price={d_price} h_dim={h_dim} d_final={d_final}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    N = Xw_t.shape[0]
    n_batches = max(1, N // batch_size)

    for epoch in range(max_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        model.train()
        for bi in range(n_batches):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            pred = model(Xw_t[idx], Xp_t[idx])
            loss = nn.functional.mse_loss(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches

        model.eval()
        with torch.no_grad():
            val_pred = model(Xw_v, Xp_v)
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
                f"  [factored_mlp] epoch={epoch+1:3d}  train_loss={epoch_loss:.6f}  "
                f"val_loss={val_loss:.6f}  best={best_val_loss:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

        if wait >= patience:
            print(f"  [factored_mlp] Early stopping at epoch {epoch+1}")
            break

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    # Extract to numpy — bake input standardization into encoder layers
    with torch.no_grad():
        sd = model.state_dict()
        # Work encoder
        Ww_raw = sd["work_enc.weight"].cpu().numpy().astype(np.float64)   # (h_dim, d_work)
        bw_raw = sd["work_enc.bias"].cpu().numpy().astype(np.float64)    # (h_dim,)
        # Price encoder
        Wp_raw = sd["price_enc.weight"].cpu().numpy().astype(np.float64)  # (h_dim, d_price)
        bp_raw = sd["price_enc.bias"].cpu().numpy().astype(np.float64)   # (h_dim,)
        # Final layers
        Wf1_raw = sd["final1.weight"].cpu().numpy().astype(np.float64)   # (d_final, d_combined)
        bf1_raw = sd["final1.bias"].cpu().numpy().astype(np.float64)     # (d_final,)
        Wf2_raw = sd["final2.weight"].cpu().numpy().astype(np.float64)   # (1, d_final)
        bf2_raw = sd["final2.bias"].cpu().numpy().astype(np.float64)     # (1,)

    # Bake standardization into encoder weights:
    #   h = W_raw @ ((x - mean) / std) + b_raw
    #     = (W_raw / std) @ x + (b_raw - W_raw @ (mean / std))
    work_std_64 = work_std.astype(np.float64)
    work_mean_64 = work_mean.astype(np.float64)
    Ww_baked = Ww_raw / work_std_64[None, :]                     # (h_dim, d_work)
    bw_baked = bw_raw - Ww_raw @ (work_mean_64 / work_std_64)   # (h_dim,)

    price_std_64 = price_std.astype(np.float64)
    price_mean_64 = price_mean.astype(np.float64)
    Wp_baked = Wp_raw / price_std_64[None, :]                      # (h_dim, d_price)
    bp_baked = bp_raw - Wp_raw @ (price_mean_64 / price_std_64)  # (h_dim,)

    # Transpose for x @ W format
    return FactoredMLPValueModel(
        W_work=Ww_baked.T,      # (d_work, h_dim)
        b_work=bw_baked,         # (h_dim,)
        W_price=Wp_baked.T,     # (d_price, h_dim)
        b_price=bp_baked,        # (h_dim,)
        W_final1=Wf1_raw.T,    # (d_combined, d_final)
        b_final1=bf1_raw,        # (d_final,)
        W_final2=Wf2_raw.T,    # (d_final, 1)
        b_final2=bf2_raw,        # (1,)
        work_idx=work_idx,
        price_idx=price_idx,
        spec=FeatureSpec(),      # will be set by caller
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
