import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


_NAMED_PROFILES: Dict[str, List[float]] = {
    "flat": [3.0] * 20,
    "two_block": [1.0] * 10 + [6.0] * 10,
    "ramp": [float(1 + i * 0.5) for i in range(20)],
    "double_peak": (
        [1.0, 1.0, 2.0, 4.0, 6.0, 4.0, 2.0, 1.0, 1.0, 1.0]
        + [1.0, 1.0, 2.0, 4.0, 6.0, 5.0, 3.0, 2.0, 1.0, 1.0]
    ),
    "weekend_weekday": [1.0] * 6 + [4.0] * 8 + [1.0] * 6,
}


def _make_generate_data_daily_prices(
    *,
    seed: int,
    T: int = 20,
    Tk_choices: Sequence[int] = (2, 3, 5),
    ck_low: int = 1,
    ck_high: int = 8,
) -> List[float]:
    import random

    if T <= 0:
        raise ValueError("T must be positive")
    if ck_low > ck_high:
        raise ValueError("ck_low must be <= ck_high")

    rng = random.Random(int(seed))

    while True:
        remaining = int(T)
        Tk: List[int] = []
        while remaining > 0:
            feasible = [int(x) for x in Tk_choices if int(x) <= remaining]
            if not feasible:
                break
            dur = int(rng.choice(feasible))
            Tk.append(dur)
            remaining -= dur
        if remaining == 0 and Tk:
            break

    ck = [int(rng.randint(int(ck_low), int(ck_high))) for _ in range(len(Tk))]
    ct: List[float] = []
    for dur, price in zip(Tk, ck):
        ct.extend([float(price)] * int(dur))
    if len(ct) != int(T):
        raise RuntimeError("Internal error: generated daily profile has wrong length")
    return ct


def _make_non_repeating_daily_prices(
    *,
    seed: int,
    D: int,
    H: int = 20,
    Tk_choices: Sequence[int] = (2, 3, 5),
    ck_low: int = 1,
    ck_high: int = 8,
) -> List[float]:
    import random

    rng = random.Random(int(seed))
    ct: List[float] = []
    for _d in range(int(D)):
        while True:
            remaining = int(H)
            Tk: List[int] = []
            while remaining > 0:
                feasible = [int(x) for x in Tk_choices if int(x) <= remaining]
                if not feasible:
                    break
                dur = int(rng.choice(feasible))
                Tk.append(dur)
                remaining -= dur
            if remaining == 0 and Tk:
                break
        ck = [int(rng.randint(int(ck_low), int(ck_high))) for _ in range(len(Tk))]
        for dur, price in zip(Tk, ck):
            ct.extend([float(price)] * int(dur))
    return ct


def _make_weekly_repeating_daily_prices(
    *,
    seed: int,
    n_days_per_week: int = 7,
    H_day: int = 20,
    Tk_choices: Sequence[int] = (2, 3, 5),
    ck_low: int = 1,
    ck_high: int = 8,
) -> List[float]:
    import random

    rng = random.Random(int(seed))
    ct: List[float] = []
    for _d in range(int(n_days_per_week)):
        while True:
            remaining = int(H_day)
            Tk: List[int] = []
            while remaining > 0:
                feasible = [int(x) for x in Tk_choices if int(x) <= remaining]
                if not feasible:
                    break
                dur = int(rng.choice(feasible))
                Tk.append(dur)
                remaining -= dur
            if remaining == 0 and Tk:
                break
        ck = [int(rng.randint(int(ck_low), int(ck_high))) for _ in range(len(Tk))]
        for dur, price in zip(Tk, ck):
            ct.extend([float(price)] * int(dur))
    return ct


def _make_realized_prices_from_forecast(
    forecast_prices: np.ndarray,
    *,
    seed: int,
    sigma: float,
    rho: float,
    spike_prob: float,
    spike_mag: float,
    spike_dur: int,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    fc = np.asarray(forecast_prices, dtype=np.float64)
    T = int(fc.shape[0])
    if T <= 0:
        raise ValueError("forecast_prices must be non-empty")

    rng = np.random.default_rng(int(seed))

    sig = float(sigma)
    r = float(rho)
    r = max(-0.999, min(0.999, r))

    eps = np.zeros(T, dtype=np.float64)
    if sig > 0.0:
        z = rng.standard_normal(T).astype(np.float64)
        for t in range(1, T):
            eps[t] = r * eps[t - 1] + sig * z[t]

    spikes = np.zeros(T, dtype=np.float64)
    p = float(spike_prob)
    if p > 0.0 and float(spike_mag) != 0.0 and int(spike_dur) > 0:
        dur = int(spike_dur)
        for t in range(T):
            if float(rng.random()) < p:
                end = min(T, t + dur)
                spikes[t:end] += float(spike_mag)

    realized = fc + eps + spikes
    realized = np.clip(realized, float(clip_low), float(clip_high))
    return realized.astype(np.float64)


def _make_drifting_amplitude_prices(
    forecast_prices: np.ndarray,
    *,
    seed: int,
    drift_sigma: float,
    drift_rho: float = 0.9,
    H: int = 20,
    clip_low: float = 0.1,
) -> np.ndarray:
    fc = np.asarray(forecast_prices, dtype=np.float64)
    T = int(len(fc))
    D = T // int(H)
    if D * int(H) != T:
        raise ValueError(f"T={T} not divisible by H={H}")

    rng = np.random.default_rng(int(seed))
    r = float(max(-0.999, min(0.999, float(drift_rho))))

    alphas = np.zeros(D, dtype=np.float64)
    z = rng.standard_normal(D).astype(np.float64)
    for d in range(1, D):
        alphas[d] = r * alphas[d - 1] + float(drift_sigma) * z[d]

    realized = fc.copy()
    for d in range(D):
        start = d * int(H)
        end = start + int(H)
        factor = max(float(clip_low), 1.0 + float(alphas[d]))
        realized[start:end] *= factor

    return np.maximum(realized, float(clip_low))


def _make_biased_forecast_prices(
    true_prices: np.ndarray,
    *,
    bias_factor: float = 0.0,
    bias_shift: int = 0,
    H: int = 20,
) -> np.ndarray:
    true = np.asarray(true_prices, dtype=np.float64)
    T = int(len(true))
    D = T // int(H)
    if D * int(H) != T:
        raise ValueError(f"T={T} not divisible by H={H}")

    forecast = true.copy()

    if int(bias_shift) != 0:
        for d in range(D):
            start = d * int(H)
            end = start + int(H)
            day = forecast[start:end].copy()
            forecast[start:end] = np.roll(day, int(bias_shift))

    if float(bias_factor) > 0.0:
        day0 = forecast[: int(H)]
        threshold = float(np.percentile(day0, 66.7))
        for t in range(T):
            if float(forecast[t]) >= threshold:
                forecast[t] *= 1.0 - float(bias_factor)

    return forecast


def _repeat_day(day20: Sequence[float], *, D: int) -> np.ndarray:
    day = np.asarray(list(day20), dtype=np.float64)
    return np.tile(day, int(D)).astype(np.float64)


def _weekly_repeat(template140: Sequence[float], *, D: int) -> np.ndarray:
    week = np.asarray(list(template140), dtype=np.float64)
    H_week = int(len(week))
    T = int(D) * 20
    if H_week <= 0:
        raise ValueError("weekly template is empty")
    reps = int(np.ceil(T / H_week))
    out = np.tile(week, reps)[:T]
    return out.astype(np.float64)


def _build_prices(profile: str, *, seed: int, D: int, ck_low: int, ck_high: int) -> Tuple[np.ndarray, int]:
    p = str(profile).strip().lower()
    if p in _NAMED_PROFILES:
        return _repeat_day(_NAMED_PROFILES[p], D=D), 20
    if p == "generate_data":
        day = _make_generate_data_daily_prices(seed=seed, T=20, ck_low=ck_low, ck_high=ck_high)
        return _repeat_day(day, D=D), 20
    if p == "non_repeating":
        out = _make_non_repeating_daily_prices(seed=seed, D=D, H=20, ck_low=ck_low, ck_high=ck_high)
        return np.asarray(out, dtype=np.float64), 20
    if p == "weekly_repeating":
        week = _make_weekly_repeating_daily_prices(seed=seed, n_days_per_week=7, H_day=20, ck_low=ck_low, ck_high=ck_high)
        return _weekly_repeat(week, D=D), 140
    raise ValueError(f"Unknown profile: {profile!r}")


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "matplotlib is required. Install it (e.g. pip install matplotlib)."
        ) from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profiles",
        type=str,
        default="flat,two_block,ramp,double_peak,weekend_weekday,generate_data,non_repeating,weekly_repeating",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D", type=int, default=6)
    ap.add_argument("--ck-low", type=int, default=1)
    ap.add_argument("--ck-high", type=int, default=8)

    ap.add_argument(
        "--noise-mode",
        type=str,
        default="none",
        choices=["none", "forecast_realized", "drifting_amplitude", "forecast_bias"],
    )

    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--spike-prob", type=float, default=0.0)
    ap.add_argument("--spike-mag", type=float, default=0.0)
    ap.add_argument("--spike-dur", type=int, default=1)
    ap.add_argument("--clip-low", type=float, default=0.1)
    ap.add_argument("--clip-high", type=float, default=20.0)

    ap.add_argument("--drift-sigma", type=float, default=0.0)
    ap.add_argument("--drift-rho", type=float, default=0.9)

    ap.add_argument("--bias-factor", type=float, default=0.0)
    ap.add_argument("--bias-shift", type=int, default=0)

    ap.add_argument("--save-dir", type=str, default="")
    ap.add_argument("--show", action="store_true")

    args = ap.parse_args()

    plt = _ensure_matplotlib()

    profiles = [p.strip() for p in str(args.profiles).split(",") if p.strip()]

    n = len(profiles)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(2.2 * n, 3.5)), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, profiles):
        base, H_cycle = _build_prices(
            p, seed=int(args.seed), D=int(args.D), ck_low=int(args.ck_low), ck_high=int(args.ck_high)
        )

        if args.noise_mode == "none":
            forecast = base
            realized = base
        elif args.noise_mode == "forecast_realized":
            forecast = base
            realized = _make_realized_prices_from_forecast(
                forecast,
                seed=int(args.seed) + 123,
                sigma=float(args.sigma),
                rho=float(args.rho),
                spike_prob=float(args.spike_prob),
                spike_mag=float(args.spike_mag),
                spike_dur=int(args.spike_dur),
                clip_low=float(args.clip_low),
                clip_high=float(args.clip_high),
            )
        elif args.noise_mode == "drifting_amplitude":
            forecast = base
            realized = _make_drifting_amplitude_prices(
                forecast,
                seed=int(args.seed) + 123,
                drift_sigma=float(args.drift_sigma),
                drift_rho=float(args.drift_rho),
                H=20,
                clip_low=float(args.clip_low),
            )
        elif args.noise_mode == "forecast_bias":
            true_prices = base
            forecast = _make_biased_forecast_prices(
                true_prices,
                bias_factor=float(args.bias_factor),
                bias_shift=int(args.bias_shift),
                H=20,
            )
            realized = true_prices
        else:
            raise ValueError(f"Unknown noise mode: {args.noise_mode!r}")

        t = np.arange(len(base), dtype=np.int32)
        ax.plot(t, realized, label="realized", linewidth=2)
        if not np.allclose(forecast, realized):
            ax.plot(t, forecast, label="forecast", linewidth=2, alpha=0.8)
        for d in range(1, int(args.D)):
            ax.axvline(d * 20, color="k", alpha=0.10, linewidth=1)

        ax.set_title(f"{p}  (D={int(args.D)}, H_cycle={H_cycle})")
        ax.set_ylabel("price")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time slot")
    fig.tight_layout()

    if str(args.save_dir).strip():
        out_dir = Path(str(args.save_dir)).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"price_profiles_seed{int(args.seed)}_D{int(args.D)}_{args.noise_mode}.png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=180)
        print(str(out_path))

    if bool(args.show):
        plt.show()


if __name__ == "__main__":
    main()
