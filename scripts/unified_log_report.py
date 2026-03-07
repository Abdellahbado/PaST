#!/usr/bin/env python3
"""Unified parser/report for PaST/ADP experiment logs.

This script is intended to replace one-off parsers by ingesting multiple log
formats and producing consistent summaries.

Usage:
  python scripts/unified_log_report.py \
    --logs ADP/logs/logs/rigourous/complete_new.log ADP/logs/logs/rigourous/round_2.log ADP/logs/logs/rigourous/round_3.log \
    --out-dir ADP/logs/analysis_unified

It prints summaries to stdout and also writes CSVs to --out-dir.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -------------------------
# Regexes (format-tolerant)
# -------------------------

# Seed-level line (rigorous/v2/v1 style)
SEED_LINE_RE = re.compile(
    r"seed=(?P<seed>\d+)\s+beam=(?P<beam>\d+)\s+"
    r"exact=(?P<exact>[\d.eE+-]+)\s+L=(?P<L>[\d.eE+-]+)\s+Z=(?P<Z>[\d.eE+-]+)\s+P=(?P<P>[\d.eE+-]+)\s+"
    r"gapL=(?P<gapL>[\d.eE+-]+)%\s+gapZ=(?P<gapZ>[\d.eE+-]+)%\s+gapP=(?P<gapP>[\d.eE+-]+)%"
    r"(?:\s+t_exact=(?P<t_exact>[\d.eE+-]+)s\s+tL=(?P<t_L>[\d.eE+-]+)s)?"
)

# Run headers (v2 style)
RUN_HEADER_RE = re.compile(r"\[(?P<idx>\d+)(?:/(?P<total>\d+))?\]\s+(?P<desc>.*?):\s+(?P<config>\S+)")

# Generic phase/experiment markers
PHASE_MARKER_RE = re.compile(r"Starting\s+Experiment\s+(?P<name>[^—\-]+)")

EXPERIMENT_TITLE_RE = re.compile(r"^\s*Experiment\s+(?P<id>[A-Za-z0-9\-]+)\s*[—\-]\s*(?P<title>.+?)\s*$")

# Round-2/3 style experiment tags
TAG_RE = re.compile(r"(?:RUN|CROSS|EVAL)\s+(?P<tag>\S+)")

PHASE_LINE_RE = re.compile(r"^\s*Phase\s+(?P<phase>[A-Za-z0-9\-]+):")

# Aggregated evaluation summary line (round_2 parser style)
AGG_EVAL_RE = re.compile(
    r"beam=\s*(?P<beam>\d+)\s+n=\s*(?P<n>\d+)\s+"
    r"gapL=\s*(?P<gapL_mean>[\d.]+)%/\s*(?P<gapL_median>[\d.]+)%\s+"
    r"gapZ=\s*(?P<gapZ_mean>[\d.]+)%/\s*(?P<gapZ_median>[\d.]+)%\s+"
    r"gapP=\s*(?P<gapP_mean>[\d.]+)%/\s*(?P<gapP_median>[\d.]+)%\s+"
    r"speedL=(?P<speedL>[\d.]+)x"
)

# Common parameter lines
MODEL_TYPE_RE = re.compile(r"Model type:\s*(?P<model>\w+)")
INSTANCE_PARAMS_RE = re.compile(
    r"Instance params:\s*D=(?P<D>\d+).*?N=(?P<N>\d+).*?pmax=(?P<pmax>\d+)"
)
TARGET_UTIL_RE = re.compile(r"target_util\s*=\s*(?P<util>[\d.]+)")
SAMPLES_PER_INSTANCE_RE = re.compile(r"Samples per instance:\s*(?P<spi>\d+)")
POOLED_SAMPLES_RE = re.compile(r"Pooled\s+samples:\s*(?P<count>[\d,]+)")
R2_RE = re.compile(r"R2_train=(?P<r2_train>[\d.]+)\s+R2_test=(?P<r2_test>[\d.]+)")
NOISE_RE = re.compile(r"sigma=(?P<sigma>[\d.]+).*?rho=(?P<rho>[\d.]+).*?spike_prob=(?P<spike_prob>[\d.]+)")
EVAL_PARAMS_RE = re.compile(r"Eval params:\s*D=(?P<D>\d+).*?N=(?P<N>\d+).*?pmax=(?P<pmax>\d+)")
PRICE_MODE_RE = re.compile(r"Eval prices:\s*(?P<mode>forecast_realized|deterministic)")

PROFILES_RE = re.compile(r"^\s*Profiles:\s*(?P<profiles>.+?)\s*$")
MODELS_RE = re.compile(r"^\s*Models:\s*(?P<models>.+?)\s*$")
TRAIN_RANGE_RE = re.compile(r"^\s*Train:\s*(?P<text>.+?)\s*$")
EVAL_SMALL_RANGE_RE = re.compile(r"^\s*Eval small:\s*(?P<text>.+?)\s*$")
EVAL_MEDIUM_RANGE_RE = re.compile(r"^\s*Eval medium:\s*(?P<text>.+?)\s*$")
TOTAL_RUNS_RE = re.compile(r"^\s*Total runs:\s*(?P<n>\d+)\s*$")

# Special-case patterns for some shell-script logs that don't have explicit "Experiment X:" headers.
HARD_PROFILE_HEADER_RE = re.compile(
    r"Hard profile:\s*(?P<profile>[A-Za-z0-9_\-]+)\s*\((?P<size>small|medium) size\)", re.IGNORECASE
)
RUN_ARROW_RE = re.compile(
    r"\[run\]\s*(?P<model>[A-Za-z0-9_\-]+)\s*→\s*(?P<path>.+\.csv)", re.IGNORECASE
)

# -------------------------
# Data structures
# -------------------------


def _clean_tag(tag: str) -> str:
    return re.sub(r"[=:].*", "", tag).strip()


@dataclass
class RunContext:
    source_log: str
    phase: str = "unknown"
    experiment: str = "unknown"
    run_id: Optional[int] = None
    run_total: Optional[int] = None
    desc: str = ""
    config: str = ""
    tag: str = ""

    model_type: str = "unknown"
    target_util: Optional[float] = None
    train_D: Optional[int] = None
    train_N: Optional[int] = None
    train_pmax: Optional[int] = None
    eval_D: Optional[int] = None
    eval_N: Optional[int] = None
    eval_pmax: Optional[int] = None
    price_mode: str = "unknown"

    samples_per_instance: Optional[int] = None
    pooled_samples: Optional[int] = None
    r2_train: Optional[float] = None
    r2_test: Optional[float] = None

    sigma: Optional[float] = None
    rho: Optional[float] = None
    spike_prob: Optional[float] = None


@dataclass
class SeedResult:
    source_log: str
    phase: str
    experiment: str
    run_key: str

    config: str
    desc: str
    tag: str

    model_type: str
    target_util: Optional[float]

    train_D: Optional[int]
    train_N: Optional[int]
    train_pmax: Optional[int]
    eval_D: Optional[int]
    eval_N: Optional[int]
    eval_pmax: Optional[int]
    price_mode: str

    samples_per_instance: Optional[int]
    pooled_samples: Optional[int]
    r2_train: Optional[float]
    r2_test: Optional[float]

    sigma: Optional[float]
    rho: Optional[float]
    spike_prob: Optional[float]

    seed: int
    beam: int
    exact: float
    L: float
    Z: float
    P: float
    gapL: float
    gapZ: float
    gapP: float
    t_exact: Optional[float] = None
    t_L: Optional[float] = None


@dataclass
class AggregatedEval:
    source_log: str
    phase: str
    experiment: str
    run_key: str

    tag: str

    beam: int
    n: int
    gapL_mean: float
    gapL_median: float
    gapZ_mean: float
    gapZ_median: float
    gapP_mean: float
    gapP_median: float
    speedL: float


@dataclass
class ExperimentMeta:
    key: str
    source_log: str

    experiment_id: str = "unknown"
    title: str = ""
    phase: str = "unknown"

    profiles: str = ""
    models: str = ""
    target_util: Optional[float] = None
    train_range: str = ""
    eval_small_range: str = ""
    eval_medium_range: str = ""
    total_runs: Optional[int] = None


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    title: str
    purpose: str
    setup_notes: List[str]


EXPERIMENT_DEFINITIONS: Dict[str, ExperimentDefinition] = {
    # V1 orchestrator (scripts/run_all_rigorous_experiments.sh + exp_*.sh)
    "B": ExperimentDefinition(
        experiment_id="B",
        title="Profile Complexity Sweep",
        purpose=(
            "Quantify how pricing-profile complexity affects approximation quality and cross-profile generalization."
        ),
        setup_notes=[
            "Runs sandbox/eval_pooled_vhat.py.",
            "Trains on small instances and evaluates on small and medium.",
            "Uses all named profiles plus generate_data.",
            "Uses beams 2,5,10.",
        ],
    ),
    "C": ExperimentDefinition(
        experiment_id="C",
        title="Regularization & Feature Enrichment Comparison",
        purpose=(
            "Compare regularization choices (ridge/poly vs LASSO vs ElasticNet) under varying feature richness, and measure cross-size generalization."
        ),
        setup_notes=[
            "Runs sandbox/eval_pooled_vhat.py.",
            "Profile fixed to daily_tou to isolate modeling effects.",
            "Trains on small, evaluates on small and medium.",
            "Uses beams 2,5,10.",
        ],
    ),
    "A": ExperimentDefinition(
        experiment_id="A",
        title="Noise Stress Test",
        purpose=(
            "Identify noise regimes where guided beam DP breaks down (gap thresholds), applying noise at evaluation time only without retraining."
        ),
        setup_notes=[
            "Eval-only: loads checkpoints from Experiment B.",
            "Applies forecast_realized price mode noise with sigma/rho and optional spikes.",
            "Uses beams 2,5,10.",
        ],
    ),
    "D": ExperimentDefinition(
        experiment_id="D",
        title="Combined Noise + Profile Stress",
        purpose=(
            "Map the operating envelope over (profile, noise) where guided beam DP stays within a target gap band."
        ),
        setup_notes=[
            "Eval-only: loads checkpoints from Experiment B.",
            "Uses forecast_realized noise with high rho and extreme spikes.",
            "Uses beams 2,5,10.",
        ],
    ),
    # V2 orchestrator (scripts/run_all_v2_experiments.sh + exp_*.sh)
    "B-hard": ExperimentDefinition(
        experiment_id="B-hard",
        title="Profile Sweep (Hard Instances)",
        purpose=(
            "Re-train and evaluate on calibrated hard instances to avoid trivial zero-gap regimes, and establish the core checkpoints used by E/F/G/H2."
        ),
        setup_notes=[
            "Runs sandbox/eval_pooled_vhat.py.",
            "Higher utilization (target_util=0.95) and matched N ranges to avoid p_j collapsing to 1.",
            "Trains on hard-small and evaluates on hard-small and hard-medium (cross-size).",
            "Uses beams 2,5,10 in the script (comment mentions larger beams; keep script as authority).",
        ],
    ),
    "H": ExperimentDefinition(
        experiment_id="H",
        title="Non-Repeating Price Profile",
        purpose=(
            "Test generalization when the repeating-profile assumption is broken, separating adaptation (train+eval on non-repeating) from zero-shot transfer (evaluate B-hard models on non-repeating)."
        ),
        setup_notes=[
            "H1: train on non_repeating and eval on non_repeating.",
            "H2: eval repeating-trained (B-hard) models on non_repeating (zero-shot).",
            "Uses target_util=0.95, pmax=12, beams 5,10,20 in the script.",
        ],
    ),
    "E": ExperimentDefinition(
        experiment_id="E",
        title="Epsilon-Constraint Boundary (Energy + Makespan)",
        purpose=(
            "Evaluate learned Vhat guidance in an epsilon-constraint parallel-machine decomposition as deadlines tighten, where guided search is expected to add value versus price guidance."
        ),
        setup_notes=[
            "Runs sandbox/eval_epsilon_constraint_sim.py (not eval_pooled_vhat.py).",
            "Eval-only: loads checkpoints from B-hard.",
            "Sweeps target_util in {0.85,0.90,0.95} and epsilon boundary via simulation.",
            "Uses beams 20 and 50.",
        ],
    ),
    "F": ExperimentDefinition(
        experiment_id="F",
        title="Weakly Repeating Price Stress (Drifting Amplitude)",
        purpose=(
            "Measure degradation as the repeating-profile assumption weakens via day-to-day multiplicative amplitude drift between forecast features and realized costs."
        ),
        setup_notes=[
            "Eval-only: loads checkpoints from B-hard.",
            "Uses eval-price-mode drifting_amplitude with drift_sigma grid and drift_rho in {0.0,0.9}.",
            "Uses beams 5,10,20.",
        ],
    ),
    "G": ExperimentDefinition(
        experiment_id="G",
        title="Forecast Bias Stress",
        purpose=(
            "Test robustness to systematic forecast errors (peak underprediction and timing shift) where features are computed on biased forecasts but costs use true prices."
        ),
        setup_notes=[
            "Eval-only: loads checkpoints from B-hard.",
            "Uses eval-price-mode forecast_bias with bias_factor grid and bias_shift in {0,1,2}.",
            "Uses beams 5,10,20.",
        ],
    ),
    # Round 2 corrected experiments (scripts/run_round2_all.sh + exp_*.sh)
    "B2": ExperimentDefinition(
        experiment_id="B2",
        title="Corrected Daily-Repeating Baseline (Small + Medium + Cross-Size)",
        purpose=(
            "Provide a matched repeating baseline at small and medium scales (with corrected labels/features), serving as a reference for I (NR-honest) and J (weekly)."
        ),
        setup_notes=[
            "Runs sandbox/eval_pooled_vhat.py with cached pooled data per (profile,size).",
            "Profiles: ramp, double_peak, daily_tou, generate_data.",
            "Models: mlp, poly, poly_mlp, factored_mlp.",
            "Includes cross-size evaluation (train small -> eval medium).",
            "Uses beams 2,5,10 and target_util=0.95.",
        ],
    ),
    "I": ExperimentDefinition(
        experiment_id="I",
        title="Non-Repeating with NR-Honest Features",
        purpose=(
            "Correct the non-repeating setting by constructing features from the true full-trajectory prices rather than a day-1 copy proxy; quantify gains and remaining transfer gaps."
        ),
        setup_notes=[
            "I1: train non_repeating and eval non_repeating at small and medium.",
            "I2: zero-shot transfer: evaluate repeating-trained models on non_repeating.",
            "I3: cross-size: train NR-small -> eval NR-medium.",
            "Uses beams 2,5,10 and target_util=0.95.",
        ],
    ),
    "J": ExperimentDefinition(
        experiment_id="J",
        title="Weekly-Repeating (H_cycle=140)",
        purpose=(
            "Test periodicity mismatch when the cycle is a week (7 unique days) rather than a single day; evaluate whether guidance scales to longer horizons and harder DP regimes."
        ),
        setup_notes=[
            "J1: train weekly_repeating and eval weekly_repeating at small and medium.",
            "J2: zero-shot transfer: evaluate daily-repeating-trained models on weekly.",
            "J3: cross-size: train weekly-small -> eval weekly-medium.",
            "Uses beams 2,5,10 and target_util=0.95.",
        ],
    ),
}


def _mk_experiment_key(source_log: str, phase: str, experiment_id: str, title: str) -> str:
    core = experiment_id if experiment_id and experiment_id != "unknown" else (phase or "unknown")
    t = title.strip() or "unknown"
    return f"{source_log}|{core}|{t}"


def _purpose_from_experiment_id_and_title(experiment_id: str, title: str) -> str:
    t = (title or "").lower()
    eid = (experiment_id or "").lower()

    if "profile sweep" in t or "sweep" in t:
        return "Measure sensitivity to pricing profile structure and compare model/heuristic robustness across profiles."
    if "noise" in t or "sigma" in t:
        return "Evaluate robustness to stochastic price noise (noise level, correlation, spikes) while keeping instance generation fixed."
    if "drift" in t or "weak repeat" in t:
        return "Evaluate performance under distribution shift (temporal drift / reduced repeatability) relative to training distribution."
    if "forecast bias" in t or "bias" in t:
        return "Evaluate robustness to systematic forecast bias between forecasted and realized prices."
    if "non-repeating" in t or eid.startswith("h"):
        return "Evaluate generalization to non-repeating price patterns (out-of-distribution relative to repeating training)."
    if eid.startswith("b"):
        return "Baseline comparison across profiles/models; establishes reference performance and gap structure."
    if eid.startswith("c"):
        return "Ablation/comparison of modeling choices (e.g., regularization, features) under a fixed evaluation protocol."
    if eid.startswith("a"):
        return "Stress test under stochastic price perturbations (noise)."
    if eid.startswith("d"):
        return "Combined stress test across multiple sources of difficulty (e.g., noise × profile)."
    return "Summarize performance under the experiment's configured training/evaluation conditions."


# -------------------------
# Parsing
# -------------------------


def _derive_phase_experiment_from_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    m = PHASE_MARKER_RE.search(line)
    if m:
        name = m.group("name").strip()
        return name, None
    return None, None


def _make_run_key(ctx: RunContext) -> str:
    # A stable key that can survive future experiments.
    parts = [ctx.source_log, ctx.phase or "unknown", ctx.experiment or "unknown"]
    if ctx.config:
        parts.append(ctx.config)
    elif ctx.tag:
        parts.append(ctx.tag)
    if ctx.run_id is not None:
        parts.append(f"run{ctx.run_id}")
    return "|".join(parts)


def _infer_experiment_id_from_tag_or_config(tag: str, config: str, experiment_fallback: str) -> str:
    s = tag or config or ""
    s = s.strip()
    if not s:
        return experiment_fallback or "unknown"

    # Common encodings: "B2_ramp_small_mlp" or "I_xxx" or "J_xxx".
    m = re.match(r"^(?P<eid>[A-Za-z]+\d+)[_\-]", s)
    if m:
        return m.group("eid")

    # v2 configs sometimes imply the experiment, while phase marker may be broad.
    if "hard" in s.lower() and "b" in (experiment_fallback or "").lower():
        return experiment_fallback

    # If we already have something like "B-hard" / "H" / "E" / "F" / "G".
    if experiment_fallback and experiment_fallback != "unknown":
        return experiment_fallback
    return "unknown"


def parse_logs(log_paths: Iterable[Path]) -> Tuple[List[SeedResult], List[AggregatedEval], Dict[str, ExperimentMeta]]:
    seed_rows: List[SeedResult] = []
    agg_rows: List[AggregatedEval] = []
    metas: Dict[str, ExperimentMeta] = {}

    for log_path in log_paths:
        ctx = RunContext(source_log=str(log_path))
        current_tag: str = ""
        current_meta_key: Optional[str] = None

        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for raw_line in f:
                    line = raw_line.rstrip("\n")

                    mpl = PHASE_LINE_RE.search(line)
                    if mpl:
                        ctx.phase = mpl.group("phase").strip()

                    # Phase/experiment marker
                    phase, exp = _derive_phase_experiment_from_line(line)
                    if phase:
                        ctx.phase = phase
                    if exp:
                        ctx.experiment = exp

                    mtit = EXPERIMENT_TITLE_RE.search(line)
                    if mtit:
                        exp_id = mtit.group("id").strip()
                        title = mtit.group("title").strip()

                        # Handle completion markers and minor format variants.
                        # Example seen in logs: "Experiment B: hard complete." (this belongs to B-hard).
                        title_l = title.lower()
                        if "complete" in title_l and exp_id in {"B", "B-hard", "H", "E", "F", "G"}:
                            current_meta_key = None
                        else:
                            if exp_id == "B" and "hard" in title_l:
                                exp_id = "B-hard"

                            ctx.experiment = exp_id
                            key = _mk_experiment_key(str(log_path), ctx.phase, exp_id, title)
                            current_meta_key = key
                            if key not in metas:
                                metas[key] = ExperimentMeta(
                                    key=key,
                                    source_log=str(log_path),
                                    experiment_id=exp_id,
                                    title=title,
                                    phase=ctx.phase,
                                )

                    # Handle hard-profile script logs (e.g., non_repeating.log)
                    mhp = HARD_PROFILE_HEADER_RE.search(line)
                    if mhp:
                        prof = mhp.group("profile").strip()
                        size = mhp.group("size").strip().lower()
                        # This comes from scripts/run_hard_profile.sh and corresponds to B-hard.
                        ctx.phase = "B"
                        ctx.experiment = "B-hard"
                        ctx.config = "hard-small" if size == "small" else "medium-hard"
                        ctx.tag = f"{prof}_{size}"
                        # Also store a minimal meta entry so the report has clear setup fields.
                        key = _mk_experiment_key(str(log_path), ctx.phase, ctx.experiment, "Hard profile sweep")
                        current_meta_key = key
                        if key not in metas:
                            metas[key] = ExperimentMeta(
                                key=key,
                                source_log=str(log_path),
                                experiment_id=ctx.experiment,
                                title="Hard profile sweep",
                                phase=ctx.phase,
                                profiles=prof,
                            )

                    mrun = RUN_ARROW_RE.search(line)
                    if mrun:
                        model = mrun.group("model").strip()
                        csv_path = mrun.group("path").strip()
                        ctx.model_type = model
                        # Try to refine tag based on filename when possible.
                        base = Path(csv_path).name
                        # Example: non_repeating_mlp_hard_small.csv
                        m = re.search(r"(?P<profile>[A-Za-z0-9_\-]+)_(?P<model>[A-Za-z0-9_\-]+)_hard_(?P<size>small|medium)", base)
                        if m:
                            prof = m.group("profile")
                            size = m.group("size")
                            ctx.phase = "B"
                            ctx.experiment = "B-hard"
                            ctx.config = "hard-small" if size == "small" else "medium-hard"
                            ctx.tag = f"{prof}_{model}_hard_{size}"

                    if current_meta_key is not None:
                        meta = metas[current_meta_key]
                        m = PROFILES_RE.search(line)
                        if m:
                            meta.profiles = m.group("profiles").strip()
                        m = MODELS_RE.search(line)
                        if m:
                            meta.models = m.group("models").strip()
                        m = TRAIN_RANGE_RE.search(line)
                        if m:
                            meta.train_range = m.group("text").strip()
                        m = EVAL_SMALL_RANGE_RE.search(line)
                        if m:
                            meta.eval_small_range = m.group("text").strip()
                        m = EVAL_MEDIUM_RANGE_RE.search(line)
                        if m:
                            meta.eval_medium_range = m.group("text").strip()
                        m = TOTAL_RUNS_RE.search(line)
                        if m:
                            try:
                                meta.total_runs = int(m.group("n"))
                            except ValueError:
                                pass
                        m = TARGET_UTIL_RE.search(line)
                        if m and meta.target_util is None:
                            try:
                                meta.target_util = float(m.group("util"))
                            except ValueError:
                                pass

                    # v2-style run header
                    mh = RUN_HEADER_RE.search(line)
                    if mh:
                        ctx.run_id = int(mh.group("idx"))
                        ctx.run_total = int(mh.group("total")) if mh.group("total") else None
                        ctx.desc = mh.group("desc").strip()
                        ctx.config = mh.group("config").strip()
                        ctx.tag = ""
                        current_tag = ""

                        # Reset per-run measurements that are frequently redefined
                        ctx.model_type = "unknown"
                        ctx.price_mode = "unknown"
                        ctx.eval_D = None
                        ctx.eval_N = None
                        ctx.eval_pmax = None
                        ctx.train_D = None
                        ctx.train_N = None
                        ctx.train_pmax = None
                        ctx.target_util = None
                        ctx.samples_per_instance = None
                        ctx.pooled_samples = None
                        ctx.r2_train = None
                        ctx.r2_test = None
                        ctx.sigma = None
                        ctx.rho = None
                        ctx.spike_prob = None

                    # round_2/3 tag
                    mt = TAG_RE.search(line)
                    if mt:
                        current_tag = _clean_tag(mt.group("tag"))
                        if current_tag:
                            ctx.tag = current_tag
                            ctx.config = ""
                            ctx.desc = ""

                    # Keep experiment id in sync with tag/config when possible.
                    inferred = _infer_experiment_id_from_tag_or_config(ctx.tag, ctx.config, ctx.experiment)
                    if inferred and inferred != "unknown":
                        ctx.experiment = inferred

                    # Params (may appear multiple times)
                    m = MODEL_TYPE_RE.search(line)
                    if m:
                        ctx.model_type = m.group("model")

                    m = TARGET_UTIL_RE.search(line)
                    if m:
                        try:
                            ctx.target_util = float(m.group("util"))
                        except ValueError:
                            pass

                    m = INSTANCE_PARAMS_RE.search(line)
                    if m:
                        ctx.train_D = int(m.group("D"))
                        ctx.train_N = int(m.group("N"))
                        ctx.train_pmax = int(m.group("pmax"))

                    m = EVAL_PARAMS_RE.search(line)
                    if m:
                        ctx.eval_D = int(m.group("D"))
                        ctx.eval_N = int(m.group("N"))
                        ctx.eval_pmax = int(m.group("pmax"))

                    m = PRICE_MODE_RE.search(line)
                    if m:
                        ctx.price_mode = m.group("mode")

                    m = SAMPLES_PER_INSTANCE_RE.search(line)
                    if m:
                        ctx.samples_per_instance = int(m.group("spi"))

                    m = POOLED_SAMPLES_RE.search(line)
                    if m:
                        ctx.pooled_samples = int(m.group("count").replace(",", ""))

                    m = R2_RE.search(line)
                    if m:
                        ctx.r2_train = float(m.group("r2_train"))
                        ctx.r2_test = float(m.group("r2_test"))

                    m = NOISE_RE.search(line)
                    if m:
                        ctx.sigma = float(m.group("sigma"))
                        ctx.rho = float(m.group("rho"))
                        ctx.spike_prob = float(m.group("spike_prob"))

                    # Aggregated eval lines
                    ma = AGG_EVAL_RE.search(line)
                    if ma and ctx.tag:
                        run_key = _make_run_key(ctx)
                        agg_rows.append(
                            AggregatedEval(
                                source_log=str(log_path),
                                phase=ctx.phase,
                                experiment=ctx.experiment,
                                run_key=run_key,
                                tag=ctx.tag,
                                beam=int(ma.group("beam")),
                                n=int(ma.group("n")),
                                gapL_mean=float(ma.group("gapL_mean")),
                                gapL_median=float(ma.group("gapL_median")),
                                gapZ_mean=float(ma.group("gapZ_mean")),
                                gapZ_median=float(ma.group("gapZ_median")),
                                gapP_mean=float(ma.group("gapP_mean")),
                                gapP_median=float(ma.group("gapP_median")),
                                speedL=float(ma.group("speedL")),
                            )
                        )

                    # Seed-level eval lines
                    ms = SEED_LINE_RE.search(line)
                    if ms:
                        run_key = _make_run_key(ctx)
                        seed_rows.append(
                            SeedResult(
                                source_log=str(log_path),
                                phase=ctx.phase,
                                experiment=ctx.experiment,
                                run_key=run_key,
                                config=ctx.config,
                                desc=ctx.desc,
                                tag=ctx.tag,
                                model_type=ctx.model_type,
                                target_util=ctx.target_util,
                                train_D=ctx.train_D,
                                train_N=ctx.train_N,
                                train_pmax=ctx.train_pmax,
                                eval_D=ctx.eval_D,
                                eval_N=ctx.eval_N,
                                eval_pmax=ctx.eval_pmax,
                                price_mode=ctx.price_mode,
                                samples_per_instance=ctx.samples_per_instance,
                                pooled_samples=ctx.pooled_samples,
                                r2_train=ctx.r2_train,
                                r2_test=ctx.r2_test,
                                sigma=ctx.sigma,
                                rho=ctx.rho,
                                spike_prob=ctx.spike_prob,
                                seed=int(ms.group("seed")),
                                beam=int(ms.group("beam")),
                                exact=float(ms.group("exact")),
                                L=float(ms.group("L")),
                                Z=float(ms.group("Z")),
                                P=float(ms.group("P")),
                                gapL=float(ms.group("gapL")),
                                gapZ=float(ms.group("gapZ")),
                                gapP=float(ms.group("gapP")),
                                t_exact=float(ms.group("t_exact")) if ms.group("t_exact") else None,
                                t_L=float(ms.group("t_L")) if ms.group("t_L") else None,
                            )
                        )

        except FileNotFoundError:
            print(f"[error] log not found: {log_path}", file=sys.stderr)
        except Exception as e:
            print(f"[error] failed reading {log_path}: {e}", file=sys.stderr)

    return seed_rows, agg_rows, metas


def _md_escape(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def _best_beam_rows(seed_summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Pick best beam per (experiment, config/tag) by min gapL_mean.
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in seed_summary:
        exp = str(r.get("experiment") or "unknown")
        name = str(r.get("config") or r.get("tag") or "unknown")
        key = (exp, name)
        gap = r.get("gapL_mean")
        if gap is None:
            continue
        if key not in best or (best[key].get("gapL_mean") is not None and float(gap) < float(best[key]["gapL_mean"])):
            best[key] = r
    return list(best.values())


def generate_markdown_report(
    *,
    log_paths: List[Path],
    seed_summary: List[Dict[str, Any]],
    agg_summary: List[Dict[str, Any]],
    metas: Dict[str, ExperimentMeta],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Index seed summaries by experiment id.
    by_exp: Dict[str, List[Dict[str, Any]]] = {}
    for r in seed_summary:
        by_exp.setdefault(str(r.get("experiment") or "unknown"), []).append(r)

    # Index aggregated summaries similarly.
    by_exp_agg: Dict[str, List[Dict[str, Any]]] = {}
    for r in agg_summary:
        by_exp_agg.setdefault(str(r.get("experiment") or "unknown"), []).append(r)

    # De-duplicate aggregated rows across logs: group by (experiment, tag, beam)
    dedup_agg: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in agg_summary:
        exp = str(r.get("experiment") or "unknown")
        tag = str(r.get("tag") or "unknown")
        beam = int(r.get("beam") or 0)
        dedup_agg.setdefault((exp, tag, beam), []).append(r)

    dedup_agg_rows: List[Dict[str, Any]] = []
    for (exp, tag, beam), rows in dedup_agg.items():
        # Average means/medians/speed across sources.
        def _avg_num(k: str) -> Optional[float]:
            vals = [float(x[k]) for x in rows if x.get(k) is not None]
            return _mean(vals) if vals else None

        dedup_agg_rows.append(
            {
                "experiment": exp,
                "tag": tag,
                "beam": beam,
                "n": int(_mean([float(x.get("n") or 0) for x in rows]) or 0),
                "gapL_mean": _avg_num("gapL_mean"),
                "gapL_median": _avg_num("gapL_median"),
                "gapP_mean": _avg_num("gapP_mean"),
                "gapP_median": _avg_num("gapP_median"),
                "gapZ_mean": _avg_num("gapZ_mean"),
                "gapZ_median": _avg_num("gapZ_median"),
                "speedL": _avg_num("speedL"),
                "sources": len(rows),
            }
        )

    by_exp_agg_dedup: Dict[str, List[Dict[str, Any]]] = {}
    for r in dedup_agg_rows:
        by_exp_agg_dedup.setdefault(str(r.get("experiment") or "unknown"), []).append(r)

    # Best-beam view for seed-level.
    best_rows = _best_beam_rows(seed_summary)
    best_by_exp: Dict[str, List[Dict[str, Any]]] = {}
    for r in best_rows:
        best_by_exp.setdefault(str(r.get("experiment") or "unknown"), []).append(r)

    # Pre-compute interpretable rollups per experiment and per (experiment, config/tag).
    # Seed-summary rows are already aggregated per run_key+beam, so weight by n.
    per_exp_pairs: Dict[str, Dict[str, List[Tuple[float, int]]]] = {}
    per_exp_group_pairs: Dict[str, Dict[str, Dict[str, List[Tuple[float, int]]]]] = {}
    for r in seed_summary:
        eid = str(r.get("experiment") or "unknown")
        name = str(r.get("config") or r.get("tag") or "")
        n = int(r.get("n") or 0)
        gL = r.get("gapL_mean")
        gP = r.get("gapP_mean")
        if gL is None or gP is None or n <= 0:
            continue
        gL = float(gL)
        gP = float(gP)
        d = gP - gL  # positive => learned better (lower gap)
        per_exp_pairs.setdefault(eid, {}).setdefault("gapL", []).append((gL, n))
        per_exp_pairs.setdefault(eid, {}).setdefault("gapP", []).append((gP, n))
        per_exp_pairs.setdefault(eid, {}).setdefault("delta", []).append((d, n))

        per_exp_group_pairs.setdefault(eid, {}).setdefault(name, {}).setdefault("delta", []).append((d, n))
        per_exp_group_pairs.setdefault(eid, {}).setdefault(name, {}).setdefault("gapL", []).append((gL, n))
        per_exp_group_pairs.setdefault(eid, {}).setdefault(name, {}).setdefault("gapP", []).append((gP, n))

    # Prefer ordering by experiments relevant to the provided rigorous logs, then log headers, then remaining.
    ordered_exp_ids: List[str] = []
    for eid in [
        "B-hard",
        "H",
        "E",
        "F",
        "G",
        "B2",
        "I",
        "J",
    ]:
        if eid not in ordered_exp_ids:
            ordered_exp_ids.append(eid)

    # Include any other definitions only if they actually occur in the parsed inputs.
    for eid in EXPERIMENT_DEFINITIONS.keys():
        if eid in ordered_exp_ids:
            continue
        if eid in by_exp or eid in by_exp_agg_dedup:
            ordered_exp_ids.append(eid)
    for m in metas.values():
        if m.experiment_id and m.experiment_id not in ordered_exp_ids:
            ordered_exp_ids.append(m.experiment_id)
    for eid in sorted(set(list(by_exp.keys()) + list(by_exp_agg.keys()) + list(by_exp_agg_dedup.keys()))):
        if eid not in ordered_exp_ids:
            ordered_exp_ids.append(eid)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Unified Experiment Report\n\n")
        f.write("## Inputs\n\n")
        for p in log_paths:
            f.write(f"- `{p}`\n")
        f.write("\n")

        f.write("## Executive summary (computed from parsed results)\n\n")
        total_seed = sum(int(r.get("n") or 0) for r in seed_summary)
        total_agg = sum(int(r.get("n") or 0) for r in agg_summary)
        f.write(f"- **Seed-level measurements**: {total_seed}\n")
        f.write(f"- **Aggregated measurements**: {total_agg}\n\n")

        f.write("## Executive insights (what works best / worst)\n\n")

        # Overall: where learned helps most / hurts most.
        all_groups: List[Tuple[float, int, str, str]] = []
        for eid, groups in per_exp_group_pairs.items():
            for name, stats in groups.items():
                d = _weighted_mean(stats.get("delta", []))
                w = sum(int(x[1]) for x in stats.get("delta", []))
                if d is None or w <= 0 or not name:
                    continue
                all_groups.append((float(d), int(w), eid, name))

        all_groups.sort(key=lambda x: x[0], reverse=True)
        top_wins = all_groups[:8]
        top_losses = list(reversed(all_groups[-8:])) if len(all_groups) >= 8 else []

        overall_delta = _weighted_mean([
            (float(r.get("gapP_mean")) - float(r.get("gapL_mean")), int(r.get("n") or 0))
            for r in seed_summary
            if r.get("gapL_mean") is not None and r.get("gapP_mean") is not None and int(r.get("n") or 0) > 0
        ])
        if overall_delta is not None:
            direction = "better" if overall_delta > 0 else "worse"
            f.write(
                f"Across all provided logs, the learned method is on average **{direction} than price guidance** by "
                f"`Δ = gapP - gapL = {_fmt(overall_delta, 3)}%` (positive favors learned).\n\n"
            )

        f.write("### Where the learned method helps most (largest Δ = gapP - gapL)\n\n")
        if not top_wins:
            f.write("No comparable (gapL_mean, gapP_mean) rows were found to compute deltas.\n\n")
        else:
            f.write("| experiment | config/tag | Δ (P-L) | weight (n) | hints |\n")
            f.write("|---|---|---:|---:|---|\n")
            for d, w, eid, name in top_wins:
                facets = _infer_facets_from_name(name)
                hints = ", ".join([f"{k}={v}" for k, v in facets.items()])
                f.write(f"| `{eid}` | {_md_escape(name)} | {d:.3f}% | {w} | {_md_escape(hints)} |\n")
            f.write("\n")

        f.write("### Where the learned method hurts most (most negative Δ = gapP - gapL)\n\n")
        if not top_losses:
            f.write("Not enough rows to compute worst-case deltas.\n\n")
        else:
            f.write("| experiment | config/tag | Δ (P-L) | weight (n) | hints |\n")
            f.write("|---|---|---:|---:|---|\n")
            for d, w, eid, name in top_losses:
                facets = _infer_facets_from_name(name)
                hints = ", ".join([f"{k}={v}" for k, v in facets.items()])
                f.write(f"| `{eid}` | {_md_escape(name)} | {d:.3f}% | {w} | {_md_escape(hints)} |\n")
            f.write("\n")

        for eid in ordered_exp_ids:
            meta = next((m for m in metas.values() if m.experiment_id == eid), None)
            definition = EXPERIMENT_DEFINITIONS.get(eid)

            title = (definition.title if definition else "") or (meta.title if meta else "")
            purpose = (definition.purpose if definition else "")
            if not purpose:
                purpose = _purpose_from_experiment_id_and_title(eid, title)

            f.write(f"## Experiment {eid}")
            if title:
                f.write(f": {_md_escape(title)}")
            f.write("\n\n")

            f.write("### Point of the experiment\n\n")
            f.write(f"{purpose}\n\n")

            f.write("### Setup\n\n")
            if definition is not None:
                for note in definition.setup_notes:
                    f.write(f"- {_md_escape(note)}\n")
            if meta is not None:
                # Only print fields if present; avoid placeholders.
                if meta.phase and meta.phase != "unknown":
                    f.write(f"- **Phase (from logs)**: `{meta.phase}`\n")
                if meta.profiles:
                    f.write(f"- **Profiles (from logs)**: `{_md_escape(meta.profiles)}`\n")
                if meta.models:
                    f.write(f"- **Models (from logs)**: `{_md_escape(meta.models)}`\n")
                if meta.target_util is not None:
                    f.write(f"- **target_util (from logs)**: `{meta.target_util}`\n")
                if meta.train_range:
                    f.write(f"- **Train range (from logs)**: `{_md_escape(meta.train_range)}`\n")
                if meta.eval_small_range:
                    f.write(f"- **Eval small range (from logs)**: `{_md_escape(meta.eval_small_range)}`\n")
                if meta.eval_medium_range:
                    f.write(f"- **Eval medium range (from logs)**: `{_md_escape(meta.eval_medium_range)}`\n")
                if meta.total_runs is not None:
                    f.write(f"- **Total runs (declared, from logs)**: `{meta.total_runs}`\n")

            # Observed beams and model types in this experiment.
            exp_rows = by_exp.get(eid, [])
            beams = sorted({int(r["beam"]) for r in exp_rows if r.get("beam") is not None})
            model_types = sorted({str(r.get("model_type") or "") for r in exp_rows if r.get("model_type")})
            if beams:
                f.write(f"- **Beams observed**: `{beams}`\n")
            if model_types:
                f.write(f"- **Model types observed**: `{model_types}`\n")
            f.write("\n")

            f.write("### Key findings (from parsed results)\n\n")
            if not exp_rows:
                f.write("No seed-level result lines matched this experiment id in the provided logs.\n\n")
            else:
                total_n = sum(int(r.get("n") or 0) for r in exp_rows)
                l_better = sum(int(r.get("L_better_P_count") or 0) for r in exp_rows)
                p_better = sum(int(r.get("P_better_L_count") or 0) for r in exp_rows)
                ties = sum(int(r.get("ties_count") or 0) for r in exp_rows)
                all_opt = sum(int(r.get("all_methods_optimal_count") or 0) for r in exp_rows)
                f.write(f"- **L better than P (count-based)**: {l_better}/{total_n} ({(100.0*l_better/total_n) if total_n else 0.0:.1f}%)\n")
                f.write(f"- **P better than L (count-based)**: {p_better}/{total_n} ({(100.0*p_better/total_n) if total_n else 0.0:.1f}%)\n")
                f.write(f"- **Ties (gapL == gapP)**: {ties}/{total_n} ({(100.0*ties/total_n) if total_n else 0.0:.1f}%)\n")
                f.write(f"- **All methods optimal (thresholded)**: {all_opt}/{total_n} ({(100.0*all_opt/total_n) if total_n else 0.0:.1f}%)\n")

                # Best beam overall by averaging run-level gapL_mean.
                by_beam_all: Dict[int, List[float]] = {}
                for r in exp_rows:
                    if r.get("gapL_mean") is None:
                        continue
                    by_beam_all.setdefault(int(r["beam"]), []).append(float(r["gapL_mean"]))
                if by_beam_all:
                    beam_stats = [(b, _mean(v)) for b, v in by_beam_all.items() if _mean(v) is not None]
                    beam_stats.sort(key=lambda x: float(x[1]))
                    best_beam, best_val = beam_stats[0]
                    f.write(f"- **Best beam (by avg gapL_mean across runs)**: `beam={best_beam}` (avg={best_val:.3f}%)\n")
            f.write("\n")

            # Make the key findings actionable: where learned is best/worst vs price.
            if exp_rows:
                exp_gapL = _weighted_mean(per_exp_pairs.get(eid, {}).get("gapL", []))
                exp_gapP = _weighted_mean(per_exp_pairs.get(eid, {}).get("gapP", []))
                exp_delta = _weighted_mean(per_exp_pairs.get(eid, {}).get("delta", []))

                f.write("### Interpretation (when does learned work best / worst?)\n\n")
                if exp_delta is None or exp_gapL is None or exp_gapP is None:
                    f.write("Not enough comparable rows to compute an average advantage of learned vs price.\n\n")
                else:
                    direction = "better" if exp_delta > 0 else "worse"
                    f.write(
                        f"On average within this experiment, learned is **{direction} than price guidance** by "
                        f"`Δ = gapP - gapL = {_fmt(exp_delta, 3)}%` (weighted by the number of seeds per run).\n\n"
                    )
                    f.write(f"- **Avg learned gap** `gapL`: `{_fmt(exp_gapL, 3)}%`\n")
                    f.write(f"- **Avg price gap** `gapP`: `{_fmt(exp_gapP, 3)}%`\n\n")

                groups = per_exp_group_pairs.get(eid, {})
                scored: List[Tuple[float, int, str]] = []
                for name, stats in groups.items():
                    d = _weighted_mean(stats.get("delta", []))
                    w = sum(int(x[1]) for x in stats.get("delta", []))
                    if d is None or w <= 0 or not name:
                        continue
                    scored.append((float(d), int(w), name))
                scored.sort(key=lambda x: x[0], reverse=True)

                if scored:
                    f.write("#### Learned helps most (top configs/tags by Δ)\n\n")
                    f.write("| config/tag | Δ (P-L) | weight (n) | hints |\n")
                    f.write("|---|---:|---:|---|\n")
                    for d, w, name in scored[:6]:
                        facets = _infer_facets_from_name(name)
                        hints = ", ".join([f"{k}={v}" for k, v in facets.items()])
                        f.write(f"| {_md_escape(name)} | {d:.3f}% | {w} | {_md_escape(hints)} |\n")
                    f.write("\n")

                    f.write("#### Learned fails most (bottom configs/tags by Δ)\n\n")
                    f.write("| config/tag | Δ (P-L) | weight (n) | hints |\n")
                    f.write("|---|---:|---:|---|\n")
                    for d, w, name in list(reversed(scored[-6:])):
                        facets = _infer_facets_from_name(name)
                        hints = ", ".join([f"{k}={v}" for k, v in facets.items()])
                        f.write(f"| {_md_escape(name)} | {d:.3f}% | {w} | {_md_escape(hints)} |\n")
                    f.write("\n")
                else:
                    f.write("No per-config/tag rows were available to localize wins/losses.\n\n")

            f.write("### Results (seed-level, grouped by config/tag and beam)\n\n")
            if not exp_rows:
                f.write("No seed-level result lines matched this experiment id in the provided logs.\n\n")
            else:
                # Best beam per config/tag
                f.write("#### Best beam per config/tag (min mean gapL)\n\n")
                f.write("| config/tag | beam | n | gapL mean | gapP mean | gapZ mean | L better than P | P better than L | all optimal |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                for r in sorted(best_by_exp.get(eid, []), key=lambda x: (str(x.get("config") or x.get("tag") or ""))):
                    name = str(r.get("config") or r.get("tag") or "unknown")
                    f.write(
                        "| "
                        + _md_escape(name)
                        + f" | {int(r.get('beam') or 0)} | {int(r.get('n') or 0)}"
                        + f" | {_fmt(r.get('gapL_mean'), 3)}% | {_fmt(r.get('gapP_mean'), 3)}% | {_fmt(r.get('gapZ_mean'), 3)}%"
                        + f" | {int(r.get('L_better_P_count') or 0)} | {int(r.get('P_better_L_count') or 0)} | {int(r.get('all_methods_optimal_count') or 0)}"
                        + " |\n"
                    )
                f.write("\n")

                # Beam-wise summary across runs
                f.write("#### Beam-level summary (mean over run-level gapL_mean)\n\n")
                by_beam: Dict[int, List[float]] = {}
                for r in exp_rows:
                    if r.get("gapL_mean") is None:
                        continue
                    by_beam.setdefault(int(r["beam"]), []).append(float(r["gapL_mean"]))
                f.write("| beam | avg(gapL_mean) | runs |\n")
                f.write("|---:|---:|---:|\n")
                for b in sorted(by_beam):
                    f.write(f"| {b} | {_fmt(_mean(by_beam[b]), 3)}% | {len(by_beam[b])} |\n")
                f.write("\n")

                # Non-trivial runs
                f.write("#### Non-trivial runs (|gapL_mean| >= 0.10%)\n\n")
                nontrivial = [r for r in exp_rows if r.get("gapL_mean") is not None and abs(float(r["gapL_mean"])) >= 0.10]
                if not nontrivial:
                    f.write("None under this threshold.\n\n")
                else:
                    f.write("| config/tag | beam | n | gapL mean | gapP mean | gapZ mean |\n")
                    f.write("|---|---:|---:|---:|---:|---:|\n")
                    for r in sorted(nontrivial, key=lambda x: (abs(float(x.get("gapL_mean") or 0.0)) * -1.0))[:40]:
                        name = str(r.get("config") or r.get("tag") or "unknown")
                        f.write(
                            "| "
                            + _md_escape(name)
                            + f" | {int(r.get('beam') or 0)} | {int(r.get('n') or 0)}"
                            + f" | {_fmt(r.get('gapL_mean'), 3)}% | {_fmt(r.get('gapP_mean'), 3)}% | {_fmt(r.get('gapZ_mean'), 3)}%"
                            + " |\n"
                        )
                    f.write("\n")

            f.write("### Results (aggregated summaries, if present in logs)\n\n")
            agg_rows_exp = by_exp_agg_dedup.get(eid, [])
            if not agg_rows_exp:
                f.write("No aggregated beam-summary lines matched this experiment id in the provided logs.\n\n")
            else:
                f.write("| tag | beam | n | gapL mean/median | gapP mean/median | gapZ mean/median | speedL | sources |\n")
                f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
                for r in sorted(agg_rows_exp, key=lambda x: (str(x.get("tag") or ""), int(x.get("beam") or 0))):
                    f.write(
                        "| "
                        + _md_escape(str(r.get("tag") or "unknown"))
                        + f" | {int(r.get('beam') or 0)} | {int(r.get('n') or 0)}"
                        + f" | {_fmt(r.get('gapL_mean'), 2)}%/{_fmt(r.get('gapL_median'), 2)}%"
                        + f" | {_fmt(r.get('gapP_mean'), 2)}%/{_fmt(r.get('gapP_median'), 2)}%"
                        + f" | {_fmt(r.get('gapZ_mean'), 2)}%/{_fmt(r.get('gapZ_median'), 2)}%"
                        + f" | {_fmt(r.get('speedL'), 2)}x"
                        + f" | {int(r.get('sources') or 1)}"
                        + " |\n"
                    )
                f.write("\n")


# -------------------------
# Reporting
# -------------------------


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def _weighted_mean(pairs: List[Tuple[float, int]]) -> Optional[float]:
    if not pairs:
        return None
    num = 0.0
    den = 0
    for v, w in pairs:
        num += float(v) * int(w)
        den += int(w)
    if den <= 0:
        return None
    return num / den


def _infer_facets_from_name(name: str) -> Dict[str, str]:
    # Best-effort extraction of interpretable “conditions” from tag/config naming.
    # This is intentionally conservative: only emit a facet if the name strongly suggests it.
    n = (name or "")
    nl = n.lower()
    out: Dict[str, str] = {}

    for size in ["small", "medium", "cross", "small_to_med", "small-to-med"]:
        if size in nl:
            out["size"] = size.replace("small_to_med", "cross").replace("small-to-med", "cross")
            break

    for profile in [
        "ramp",
        "double_peak",
        "daily_tou",
        "generate_data",
        "non_repeating",
        "weekly_repeating",
    ]:
        if profile in nl:
            out["profile"] = profile
            break

    if nl.startswith("i1_"):
        out["subexp"] = "I1 (train NR -> eval NR)"
    elif nl.startswith("i2_"):
        out["subexp"] = "I2 (zero-shot repeating -> NR)"
    elif nl.startswith("i3_"):
        out["subexp"] = "I3 (cross-size NR)"
    elif nl.startswith("j1_"):
        out["subexp"] = "J1 (train weekly -> eval weekly)"
    elif nl.startswith("j2_"):
        out["subexp"] = "J2 (zero-shot daily -> weekly)"
    elif nl.startswith("j3_"):
        out["subexp"] = "J3 (cross-size weekly)"
    elif nl.startswith("b2_"):
        out["subexp"] = "B2 (repeating baseline)"

    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def summarize_seed_results(seed_rows: List[SeedResult]) -> Tuple[List[Dict[str, Any]], List[str]]:
    # Group by run_key and beam.
    groups: Dict[Tuple[str, int], List[SeedResult]] = {}
    for r in seed_rows:
        groups.setdefault((r.run_key, r.beam), []).append(r)

    out: List[Dict[str, Any]] = []
    for (run_key, beam), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        gapL = [x.gapL for x in rows]
        gapP = [x.gapP for x in rows]
        gapZ = [x.gapZ for x in rows]

        l_better = sum(1 for x in rows if x.gapL < x.gapP)
        p_better = sum(1 for x in rows if x.gapP < x.gapL)
        ties = len(rows) - l_better - p_better
        zero_all = sum(
            1
            for x in rows
            if abs(x.gapL) < 0.005 and abs(x.gapP) < 0.005 and abs(x.gapZ) < 0.005
        )

        t_exact = [x.t_exact for x in rows if x.t_exact is not None and x.t_exact > 0]
        t_L = [x.t_L for x in rows if x.t_L is not None and x.t_L > 0]
        speed = None
        if t_exact and t_L and len(t_exact) == len(t_L):
            speed = _mean([te / tl for te, tl in zip(t_exact, t_L)])

        any_row = rows[0]
        out.append(
            {
                "source_log": any_row.source_log,
                "phase": any_row.phase,
                "experiment": any_row.experiment,
                "run_key": run_key,
                "tag": any_row.tag,
                "config": any_row.config,
                "desc": any_row.desc,
                "model_type": any_row.model_type,
                "target_util": any_row.target_util,
                "train_D": any_row.train_D,
                "train_N": any_row.train_N,
                "train_pmax": any_row.train_pmax,
                "eval_D": any_row.eval_D,
                "eval_N": any_row.eval_N,
                "eval_pmax": any_row.eval_pmax,
                "price_mode": any_row.price_mode,
                "samples_per_instance": any_row.samples_per_instance,
                "pooled_samples": any_row.pooled_samples,
                "r2_train": any_row.r2_train,
                "r2_test": any_row.r2_test,
                "sigma": any_row.sigma,
                "rho": any_row.rho,
                "spike_prob": any_row.spike_prob,
                "beam": beam,
                "n": len(rows),
                "gapL_mean": _mean(gapL),
                "gapL_median": _median(gapL),
                "gapP_mean": _mean(gapP),
                "gapP_median": _median(gapP),
                "gapZ_mean": _mean(gapZ),
                "gapZ_median": _median(gapZ),
                "L_better_P_count": l_better,
                "P_better_L_count": p_better,
                "ties_count": ties,
                "all_methods_optimal_count": zero_all,
                "speedup_exact_over_L_mean": speed,
            }
        )

    columns = [
        "source_log",
        "phase",
        "experiment",
        "tag",
        "config",
        "desc",
        "model_type",
        "target_util",
        "train_D",
        "train_N",
        "train_pmax",
        "eval_D",
        "eval_N",
        "eval_pmax",
        "price_mode",
        "samples_per_instance",
        "pooled_samples",
        "r2_train",
        "r2_test",
        "sigma",
        "rho",
        "spike_prob",
        "beam",
        "n",
        "gapL_mean",
        "gapL_median",
        "gapP_mean",
        "gapP_median",
        "gapZ_mean",
        "gapZ_median",
        "L_better_P_count",
        "P_better_L_count",
        "ties_count",
        "all_methods_optimal_count",
        "speedup_exact_over_L_mean",
        "run_key",
    ]
    return out, columns


def summarize_aggregated(agg_rows: List[AggregatedEval]) -> Tuple[List[Dict[str, Any]], List[str]]:
    out: List[Dict[str, Any]] = []
    for r in agg_rows:
        out.append(
            {
                "source_log": r.source_log,
                "phase": r.phase,
                "experiment": r.experiment,
                "tag": r.tag,
                "beam": r.beam,
                "n": r.n,
                "gapL_mean": r.gapL_mean,
                "gapL_median": r.gapL_median,
                "gapP_mean": r.gapP_mean,
                "gapP_median": r.gapP_median,
                "gapZ_mean": r.gapZ_mean,
                "gapZ_median": r.gapZ_median,
                "speedL": r.speedL,
                "run_key": r.run_key,
            }
        )

    columns = [
        "source_log",
        "phase",
        "experiment",
        "tag",
        "beam",
        "n",
        "gapL_mean",
        "gapL_median",
        "gapP_mean",
        "gapP_median",
        "gapZ_mean",
        "gapZ_median",
        "speedL",
        "run_key",
    ]
    return out, columns


def print_console_report(seed_summary: List[Dict[str, Any]], agg_summary: List[Dict[str, Any]]) -> None:
    # Keep output compact and stable.
    total_seed = sum(r.get("n", 0) for r in seed_summary)
    total_agg = sum(r.get("n", 0) for r in agg_summary)

    print("=" * 100)
    print("UNIFIED LOG REPORT")
    print("=" * 100)
    print(f"Seed-level measurements: {total_seed}")
    print(f"Aggregated measurements: {total_agg}")

    # Beam coverage
    beams_seed = sorted({r["beam"] for r in seed_summary if r.get("beam") is not None})
    beams_agg = sorted({r["beam"] for r in agg_summary if r.get("beam") is not None})
    if beams_seed:
        print(f"Beams (seed-level): {beams_seed}")
    if beams_agg:
        print(f"Beams (aggregated): {beams_agg}")

    # Top-level signal: head-to-head and zero-gap prevalence (seed-level only)
    if seed_summary:
        total_pairs = sum(r["n"] for r in seed_summary)
        l_better = sum(r["L_better_P_count"] for r in seed_summary)
        p_better = sum(r["P_better_L_count"] for r in seed_summary)
        ties = sum(r["ties_count"] for r in seed_summary)
        all_opt = sum(r["all_methods_optimal_count"] for r in seed_summary)

        print("-" * 100)
        print("Seed-level overall")
        print("-" * 100)
        if total_pairs > 0:
            print(f"L better than P: {l_better}/{total_pairs} ({100.0*l_better/total_pairs:.1f}%)")
            print(f"P better than L: {p_better}/{total_pairs} ({100.0*p_better/total_pairs:.1f}%)")
            print(f"Ties:           {ties}/{total_pairs} ({100.0*ties/total_pairs:.1f}%)")
            print(
                f"All methods optimal (|gap|<0.005% for L,P,Z): {all_opt}/{total_pairs} ({100.0*all_opt/total_pairs:.1f}%)"
            )

        # Beam sensitivity quick check (gapL_mean by beam)
        by_beam: Dict[int, List[float]] = {}
        for r in seed_summary:
            if r.get("gapL_mean") is None:
                continue
            by_beam.setdefault(int(r["beam"]), []).append(float(r["gapL_mean"]))
        if by_beam:
            print("-" * 100)
            print("Beam sensitivity (seed-summary gapL_mean averaged across runs)")
            print("-" * 100)
            for b in sorted(by_beam):
                print(f"beam={b:<3d}  avg(gapL_mean)={_fmt(_mean(by_beam[b]), nd=3)}%")

    # A small, rule-based insights section (no speculative claims)
    print("-" * 100)
    print("Notes")
    print("-" * 100)
    print("- If many rows have all methods optimal, gaps may be dominated by instance difficulty rather than model quality.")
    print("- If beam coverage differs across experiments, compare methods at matched beam values only.")
    print("- For fairness, compare L/P/Z against exact only when exact is present in the log.")


# -------------------------
# CLI
# -------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Unified parser/report for PaST experiment logs")
    parser.add_argument(
        "--logs",
        nargs="+",
        default=[
            "ADP/logs/logs/rigourous/complete_new.log",
            "ADP/logs/logs/rigourous/round_2.log",
            "ADP/logs/logs/rigourous/round_3.log",
        ],
        help="One or more log file paths",
    )
    parser.add_argument(
        "--out-dir",
        default="ADP/logs/analysis_unified",
        help="Directory to write CSV summaries",
    )
    parser.add_argument(
        "--md-out",
        default=None,
        help="Markdown report output path (default: <out-dir>/report.md)",
    )
    args = parser.parse_args(argv)

    log_paths = [Path(p) for p in args.logs]
    out_dir = Path(args.out_dir)

    seed_rows, agg_rows, metas = parse_logs(log_paths)

    seed_summary, seed_cols = summarize_seed_results(seed_rows)
    agg_summary, agg_cols = summarize_aggregated(agg_rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "seed_level_summary.csv", seed_summary, seed_cols)
    write_csv(out_dir / "aggregated_summary.csv", agg_summary, agg_cols)

    md_out = Path(args.md_out) if args.md_out else (out_dir / "report.md")
    generate_markdown_report(
        log_paths=log_paths,
        seed_summary=seed_summary,
        agg_summary=agg_summary,
        metas=metas,
        out_path=md_out,
    )

    print_console_report(seed_summary, agg_summary)

    print(f"[saved] {out_dir / 'seed_level_summary.csv'}")
    print(f"[saved] {out_dir / 'aggregated_summary.csv'}")
    print(f"[saved] {md_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
