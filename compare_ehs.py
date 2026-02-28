#!/usr/bin/env python3
"""Compare G-LNS Pareto fronts against the EHS baseline (Wang 2018 benchmark).

Usage
-----
    # Standard mode (archive_full.json from instances_90.json evolution):
    python compare_ehs.py --glns-archive <path/to/archive_full.json> \
                          [--ehs-dir Benchmark/results/EHS]

    # Benchmark-adaptation mode (also include held-out eval archive):
    python compare_ehs.py --glns-archive results/glns/archive_full.json \
                          --glns-eval-archive results/glns/archive_benchmark_eval.json

The script:
  1. Loads EHS results (10 runs × 90 instances) and merges each instance's
     runs into a single aggregated non-dominated front.
  2. Loads the G-LNS archive (archive_full.json produced by runner.py).
     Optionally also loads archive_benchmark_eval.json (held-out eval set from
     benchmark-adaptation mode) and merges it into the same comparison.
  3. For every instance that appears in both, computes:
       • Hypervolume (HV) of each front using the same reference point.
       • HV ratio  (G-LNS / EHS).
       • Number of Pareto-optimal points.
       • Coverage metric C(A,B) = fraction of B dominated by at least one A.
  4. Prints a per-instance and per-scale summary table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Pareto helpers ──────────────────────────────────────────────────────────


def _dominates(a: Tuple[int, float], b: Tuple[int, float]) -> bool:
    """True if a weakly dominates b (≤ both, < at least one)."""
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def pareto_filter(pts: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Return non-dominated subset sorted by makespan ascending."""
    nd: List[Tuple[int, float]] = []
    for p in pts:
        if not any(_dominates(q, p) for q in nd):
            nd = [q for q in nd if not _dominates(p, q)]
            nd.append(p)
    return sorted(nd, key=lambda x: x[0])


def hypervolume_2d(front: List[Tuple[int, float]], ref: Tuple[int, float]) -> float:
    """Exact 2-D HV contribution (sweep-line)."""
    pts = sorted(front, key=lambda p: p[0])
    hv = 0.0
    prev_energy = ref[1]
    for cmax, tec in pts:
        if cmax >= ref[0]:
            break
        if tec < prev_energy:
            hv += float(ref[0] - cmax) * (prev_energy - tec)
            prev_energy = tec
    return hv


def coverage(A: List[Tuple[int, float]], B: List[Tuple[int, float]]) -> float:
    """C(A,B) = fraction of points in B dominated by at least one point in A."""
    if not B:
        return 0.0
    dom_count = sum(1 for b in B if any(_dominates(a, b) for a in A))
    return dom_count / len(B)


# ── Loaders ─────────────────────────────────────────────────────────────────


def load_ehs_results(ehs_dir: Path) -> Dict[int, List[Tuple[int, float]]]:
    """Load all EHS runs and merge into one non-dominated front per instance.

    Returns {instance_id (1..90): [(makespan, energy), ...]}.
    """
    all_pts: Dict[int, List[Tuple[int, float]]] = {}

    for run_dir in sorted(ehs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for csv_file in run_dir.glob("res_*.csv"):
            inst_id = int(csv_file.stem.split("_")[1])
            pts = all_pts.setdefault(inst_id, [])
            with open(csv_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        cmax = int(parts[0])
                        tec = float(parts[1])
                        pts.append((cmax, tec))

    # Filter to non-dominated per instance.
    for iid in all_pts:
        all_pts[iid] = pareto_filter(all_pts[iid])

    return all_pts


def load_glns_archive(path: Path) -> Dict[int, List[Tuple[int, float]]]:
    """Load archive_full.json and group by instance_id.

    G-LNS uses instance_id = 1000 + benchmark_id for Wang2018 instances.
    This function returns results keyed by the *original* benchmark ID (1..90).
    """
    with open(path) as f:
        data = json.load(f)

    raw: Dict[int, List[Tuple[int, float]]] = {}
    for entry in data:
        iid = int(entry["instance_id"])
        # Map G-LNS offset IDs back to benchmark IDs.
        if iid > 1000:
            bm_id = iid - 1000
        else:
            bm_id = iid
        raw.setdefault(bm_id, []).append(
            (int(entry["makespan"]), float(entry["energy"]))
        )

    for iid in raw:
        raw[iid] = pareto_filter(raw[iid])

    return raw


# ── Reference point ─────────────────────────────────────────────────────────


def compute_ref_point(
    ehs_front: List[Tuple[int, float]],
    glns_front: List[Tuple[int, float]],
    margin: float = 1.1,
) -> Tuple[int, float]:
    """Reference point = max of both fronts × margin (per objective)."""
    all_pts = ehs_front + glns_front
    if not all_pts:
        return (1, 1.0)
    max_cmax = max(c for c, _ in all_pts)
    max_tec = max(t for _, t in all_pts)
    return (int(max_cmax * margin) + 1, max_tec * margin)


# ── Scale classification ───────────────────────────────────────────────────


def scale_for_id(iid: int) -> str:
    if 1 <= iid <= 30:
        return "small"
    elif 31 <= iid <= 60:
        return "medium"
    elif 61 <= iid <= 90:
        return "large"
    return "?"


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare G-LNS vs EHS on Wang2018 benchmark"
    )
    parser.add_argument(
        "--glns-archive",
        type=str,
        required=True,
        help="Path to G-LNS archive_full.json",
    )
    parser.add_argument(
        "--glns-eval-archive",
        type=str,
        default=None,
        help=(
            "Optional: path to archive_benchmark_eval.json produced in "
            "benchmark-adaptation mode. Its instances are merged with the "
            "main archive before comparison."
        ),
    )
    parser.add_argument(
        "--ehs-dir",
        type=str,
        default="Benchmark/results/EHS",
        help="Path to EHS results folder (contains run dirs 1..10)",
    )
    args = parser.parse_args()

    ehs_dir = Path(args.ehs_dir)
    glns_path = Path(args.glns_archive)

    if not ehs_dir.is_dir():
        print(f"ERROR: EHS directory not found: {ehs_dir}", file=sys.stderr)
        sys.exit(1)
    if not glns_path.is_file():
        print(f"ERROR: G-LNS archive not found: {glns_path}", file=sys.stderr)
        sys.exit(1)

    # Load data.
    ehs = load_ehs_results(ehs_dir)
    glns = load_glns_archive(glns_path)

    # Optionally merge the benchmark eval archive (held-out set from benchmark-adaptation).
    if args.glns_eval_archive:
        eval_path = Path(args.glns_eval_archive)
        if not eval_path.is_file():
            print(
                f"WARNING: eval archive not found: {eval_path} — skipping",
                file=sys.stderr,
            )
        else:
            eval_glns = load_glns_archive(eval_path)
            for iid, pts in eval_glns.items():
                if iid in glns:
                    merged = pareto_filter(glns[iid] + pts)
                    glns[iid] = merged
                else:
                    glns[iid] = pts
            print(f"Merged eval archive: +{len(eval_glns)} instances")

    print(f"EHS : {len(ehs)} instances loaded (10 runs merged)")
    print(f"G-LNS: {len(glns)} instances in archive")
    print()

    # Find common instances.
    common = sorted(set(ehs.keys()) & set(glns.keys()))
    if not common:
        ehs_only = sorted(ehs.keys())
        glns_only = sorted(glns.keys())
        print("No common instances found!")
        print(f"  EHS instance IDs : {ehs_only[:10]}...")
        print(f"  G-LNS instance IDs: {glns_only[:10]}...")
        print("\nG-LNS uses offset IDs (1001-1090 → mapped back to 1-90).")
        print("Check that your G-LNS archive contains Wang2018 benchmark results.")
        sys.exit(1)

    print(f"Comparing {len(common)} common instances\n")

    # ── Per-instance comparison ──────────────────────────────────────────
    header = f"{'Inst':>5} {'Scale':>6} │ {'EHS_pts':>7} {'GLNS_pts':>8} │ {'HV_EHS':>12} {'HV_GLNS':>12} {'Ratio':>7} │ {'C(G,E)':>7} {'C(E,G)':>7} │ {'Winner':>6}"
    sep = "─" * len(header)
    print(header)
    print(sep)

    results = []
    for iid in common:
        e_front = ehs[iid]
        g_front = glns[iid]

        ref = compute_ref_point(e_front, g_front)
        hv_e = hypervolume_2d(e_front, ref)
        hv_g = hypervolume_2d(g_front, ref)
        ratio = hv_g / hv_e if hv_e > 0 else float("inf")

        c_ge = coverage(g_front, e_front)  # fraction of EHS dominated by G-LNS
        c_eg = coverage(e_front, g_front)  # fraction of G-LNS dominated by EHS

        if hv_g > hv_e * 1.001:
            winner = "G-LNS"
        elif hv_e > hv_g * 1.001:
            winner = "EHS"
        else:
            winner = "TIE"

        scale = scale_for_id(iid)
        results.append(
            {
                "id": iid,
                "scale": scale,
                "ehs_pts": len(e_front),
                "glns_pts": len(g_front),
                "hv_ehs": hv_e,
                "hv_glns": hv_g,
                "ratio": ratio,
                "c_ge": c_ge,
                "c_eg": c_eg,
                "winner": winner,
            }
        )

        print(
            f"{iid:>5} {scale:>6} │ {len(e_front):>7} {len(g_front):>8} │ "
            f"{hv_e:>12.1f} {hv_g:>12.1f} {ratio:>7.3f} │ "
            f"{c_ge:>7.3f} {c_eg:>7.3f} │ {winner:>6}"
        )

    print(sep)

    # ── Scale-level summary ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY BY SCALE")
    print("=" * 70)

    scales = ["small", "medium", "large"]
    for scale in scales:
        sr = [r for r in results if r["scale"] == scale]
        if not sr:
            continue
        n = len(sr)
        wins_glns = sum(1 for r in sr if r["winner"] == "G-LNS")
        wins_ehs = sum(1 for r in sr if r["winner"] == "EHS")
        ties = sum(1 for r in sr if r["winner"] == "TIE")
        avg_ratio = sum(r["ratio"] for r in sr) / n
        avg_c_ge = sum(r["c_ge"] for r in sr) / n
        avg_c_eg = sum(r["c_eg"] for r in sr) / n

        print(f"\n  {scale.upper()} ({n} instances)")
        print(f"    Wins:  G-LNS={wins_glns}  EHS={wins_ehs}  Tie={ties}")
        print(f"    Avg HV ratio (G-LNS/EHS): {avg_ratio:.4f}")
        print(
            f"    Avg C(G-LNS,EHS):         {avg_c_ge:.4f}  ← fraction of EHS dominated by G-LNS"
        )
        print(
            f"    Avg C(EHS,G-LNS):         {avg_c_eg:.4f}  ← fraction of G-LNS dominated by EHS"
        )

    # ── Overall ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    n_total = len(results)
    wins_glns = sum(1 for r in results if r["winner"] == "G-LNS")
    wins_ehs = sum(1 for r in results if r["winner"] == "EHS")
    ties = sum(1 for r in results if r["winner"] == "TIE")
    avg_ratio = sum(r["ratio"] for r in results) / n_total
    avg_c_ge = sum(r["c_ge"] for r in results) / n_total
    avg_c_eg = sum(r["c_eg"] for r in results) / n_total

    print(f"  Instances compared: {n_total}")
    print(f"  Wins:  G-LNS={wins_glns}  EHS={wins_ehs}  Tie={ties}")
    print(f"  Avg HV ratio (G-LNS/EHS): {avg_ratio:.4f}")
    print(f"  Avg C(G-LNS,EHS):         {avg_c_ge:.4f}")
    print(f"  Avg C(EHS,G-LNS):         {avg_c_eg:.4f}")

    if avg_ratio > 1.0:
        print("\n  ✓ G-LNS produces BETTER fronts on average (HV ratio > 1)")
    elif avg_ratio < 1.0:
        print("\n  ✗ EHS produces better fronts on average (HV ratio < 1)")
    else:
        print("\n  ≈ Methods are comparable on average")

    print()


if __name__ == "__main__":
    main()
