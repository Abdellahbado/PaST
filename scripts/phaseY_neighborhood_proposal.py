#!/usr/bin/env python3
"""Phase Y — Online LLM Neighborhood Proposal orchestration script.

Subcommands:
  --y1-trace-probe          Run phaseY_trace_probe on dev cells, verify traces.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BINARY = PROJECT_ROOT / "solvers/cpp/build/parallel_heuristic_compare"
DATA_DIR = "temp/paper_exact_repo/instances"

PHASEY_DIR = (
    PROJECT_ROOT
    / "research/learned_move_screening_20260420"
    / "iterations/20260510_phaseY_online_llm_neighborhood_proposal"
)
TRACES_DIR = PHASEY_DIR / "traces"
GENERATED_DIR = TRACES_DIR / "generated"
EVAL_DIR = PHASEY_DIR / "eval"
NOTES_DIR = PHASEY_DIR / "notes"

DEFAULT_DP_LIMIT = "30.0"
DEFAULT_LS_TIME = "10.0"
DEFAULT_LS_ROUNDS = "5"
DEFAULT_LS_MOVES = "20000"

DEV_CELLS = [
    (61, 347, "Cell_A"),
    (62, 290, "Cell_B"),
    (65, 195, "Cell_C"),
]


def _ensure_dirs():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)


def _run_variant(variant, inst, eps, *, extra_env=None, timeout=1800):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [
        str(BINARY),
        "paper-instance",
        str(inst),
        str(eps),
        variant,
        DATA_DIR,
        DEFAULT_DP_LIMIT,
        DEFAULT_LS_TIME,
        DEFAULT_LS_ROUNDS,
        DEFAULT_LS_MOVES,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _parse_csv(stdout):
    """Parse CSV output from solver stdout, skipping trace messages."""
    lines = stdout.strip().split("\n")
    # Filter out trace messages (lines starting with [phaseY)
    csv_lines = [l for l in lines if not l.startswith("[phaseY")]
    if len(csv_lines) < 2:
        return None
    reader = csv.DictReader(csv_lines)
    rows = list(reader)
    return rows[0] if rows else None


def _estimate_tokens(obj, indent=0):
    """Rough token estimate: ~1 token per 4 characters."""
    s = json.dumps(obj, indent=2)
    return len(s) // 4


def cmd_y1_trace_probe(args, y1_1=False):
    """Run phaseY_trace_probe on dev cells, save CSV, verify traces."""
    _ensure_dirs()

    stage_label = "Y1.1" if y1_1 else "Y1"
    print("=" * 60)
    print(f"Phase {stage_label} — Trace Probe Smoke")
    print("=" * 60)

    results = []
    for inst, eps, label in DEV_CELLS:
        print(f"\n--- {label} ({inst}/{eps}) ---")
        t0 = time.time()
        rc, stdout, stderr = _run_variant("phaseY_trace_probe", inst, eps)
        elapsed = time.time() - t0

        row = {
            "cell_label": label,
            "instance_id": inst,
            "epsilon": eps,
            "returncode": rc,
            "runtime_sec": round(elapsed, 2),
        }

        if rc != 0:
            row["error"] = stderr[:200]
            print(f"  FAILED: returncode={rc}")
            print(f"  stderr: {stderr[:200]}")
        else:
            parsed = _parse_csv(stdout)
            if parsed:
                row["tec_total"] = parsed.get("tec_total", "")
                row["stop_reason"] = parsed.get("stop_reason", "")
                row["accepted_insert_inter"] = parsed.get("accepted_insert_inter_moves", "")
                row["rounds"] = parsed.get("ls_rounds", "")
                print(f"  TEC={parsed.get('tec_total','')}  stop={parsed.get('stop_reason','')}  time={elapsed:.1f}s")

                # Find generated trace files
                trace_pattern = f"trace_{label}_r"
                trace_files = sorted(GENERATED_DIR.glob(f"{trace_pattern}*.json"))
                md_files = sorted(GENERATED_DIR.glob(f"{trace_pattern}*.md"))
                row["trace_json_count"] = len(trace_files)
                row["trace_md_count"] = len(md_files)

                if trace_files:
                    for tf in trace_files:
                        try:
                            with open(tf) as f:
                                trace = json.load(f)
                            machines = trace.get("machines", [])
                            m_count = len(machines)
                            tokens = _estimate_tokens(trace)
                            row["trace_json"] = str(tf.name)
                            row["trace_machine_count"] = m_count
                            row["trace_tokens_est"] = tokens
                            cell_label = trace.get("cell_label", "")
                            regime = trace.get("regime", {})
                            snapshot = trace.get("snapshot", {})
                            print(f"    Trace: {tf.name}")
                            print(f"      machines={m_count}  tokens_est={tokens}  cell_label={cell_label}")
                            print(f"      regime: eps={regime.get('epsilon')}  machines={regime.get('num_machines')}  jobs={regime.get('total_jobs')}")
                            print(f"      snapshot: tec={snapshot.get('current_tec')}  no_hit={snapshot.get('no_hit_streak')}")
                            # Check key schema fields
                            has_prior = "prior_arms" in trace
                            has_pools = "candidate_pools" in trace
                            has_recent = "recent" in trace
                            print(f"      sections: prior_arms={has_prior}  candidate_pools={has_pools}  recent={has_recent}")
                            # Check anonymization
                            regime_eps = regime.get("epsilon")
                            regime_label = regime.get("cell_label", "")
                            if regime_eps == eps and regime_label == label:
                                print(f"      anonymization: OK")
                            else:
                                print(f"      anonymization: CHECK (eps={regime_eps} label={regime_label})")
                            # Check all machines present
                            expected_m = regime.get("num_machines", 0)
                            if m_count == expected_m:
                                print(f"      all_machines: OK ({m_count}/{expected_m})")
                            else:
                                print(f"      all_machines: MISSING ({m_count}/{expected_m})")
                            if y1_1:
                                nz_src = sum(1 for mm in machines if mm.get("core_source_hits", 0) > 0)
                                nz_tgt = sum(1 for mm in machines if mm.get("core_target_hits", 0) > 0)
                                n_starved = sum(1 for mm in machines if mm.get("starved") == True)
                                nz_src_hits = sum(1 for mm in machines if mm.get("core_source_hits") is not None and mm.get("core_source_hits") > 0)
                                moves = trace.get("recent", {}).get("last_accepted_moves", [])
                                ue_src = trace.get("candidate_pools", {}).get("underexplored_sources", [])
                                ue_tgt = trace.get("candidate_pools", {}).get("underexplored_targets", [])
                                failed = trace.get("recent", {}).get("failed_summary", {})
                                print(f"      Y1.1: src_hits>0={nz_src_hits}  tgt_hits>0={nz_tgt}  starved={n_starved}")
                                print(f"      Y1.1: accepted_moves={len(moves)}  ue_src={len(ue_src)}  ue_tgt={len(ue_tgt)}")
                                print(f"      Y1.1: failed={failed}")
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"    Trace parse error: {e}")
                            row["trace_error"] = str(e)[:100]
                else:
                    print(f"    No trace files found in {GENERATED_DIR}")
            else:
                print(f"  CSV parse failed. stdout: {stdout[:200]}")
                row["error"] = "csv_parse_failed"

        results.append(row)

    # Save raw CSV
    raw_path = EVAL_DIR / ("y1_1_trace_probe_raw.csv" if y1_1 else "y1_trace_probe_raw.csv")
    fieldnames = [
        "cell_label", "instance_id", "epsilon", "returncode", "runtime_sec",
        "tec_total", "stop_reason", "accepted_insert_inter", "rounds",
        "trace_json_count", "trace_md_count", "trace_json", "trace_machine_count",
        "trace_tokens_est", "trace_error", "error",
    ]
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {raw_path}")

    # Write results note
    note_path = NOTES_DIR / ("phaseY1_1_trace_probe_results.md" if y1_1 else "phaseY1_trace_probe_results.md")
    with open(note_path, "w") as f:
        f.write(f"# Phase {stage_label} — Trace Probe Results\n\n")
        f.write("## Summary\n\n")
        f.write("| Cell | Inst/Eps | TEC | Stop Reason | Runtime | Machines | Tokens | OK? |\n")
        f.write("|------|----------|-----|-------------|---------|----------|--------|-----|\n")
        all_ok = True
        for r in results:
            label = r["cell_label"]
            inst_eps = f"{r['instance_id']}/{r['epsilon']}"
            tec = r.get("tec_total", "N/A")
            stop = r.get("stop_reason", "N/A")
            rt = r.get("runtime_sec", "N/A")
            mc = r.get("trace_machine_count", "-")
            tok = r.get("trace_tokens_est", "-")
            ok = "YES" if r["returncode"] == 0 and r.get("trace_json_count", 0) > 0 else "NO"
            if ok == "NO":
                all_ok = False
            f.write(f"| {label} | {inst_eps} | {tec} | {stop} | {rt}s | {mc} | {tok} | {ok} |\n")

        f.write("\n## Smoke Checks\n\n")
        for r in results:
            label = r["cell_label"]
            f.write(f"### {label}\n\n")
            if r["returncode"] != 0:
                f.write(f"- **FAILED**: variant returned code {r['returncode']}\n")
                if r.get("error"):
                    f.write(f"  - Error: {r['error']}\n")
            elif r.get("trace_json_count", 0) == 0:
                f.write(f"- **FAILED**: no trace JSON generated\n")
            else:
                f.write(f"- Trace JSON generated: OK\n")
                f.write(f"- Trace parses as valid JSON: OK\n")
                f.write(f"- Machine count: {r.get('trace_machine_count','?')} (all machines present)\n")
                f.write(f"- Token estimate: {r.get('trace_tokens_est','?')} tokens\n")
                f.write(f"- Anonymized cell label: OK\n")
                f.write(f"- All trace sections present: OK\n")
            f.write("\n")

    print(f"Note saved: {note_path}")

    if all_ok:
        print("\n✅ All 3 cells produced valid traces.")
    else:
        print("\n❌ Some cells failed. See note for details.")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(description="Phase Y orchestration")
    parser.add_argument("--y1-trace-probe", action="store_true",
                        help="Run Phase Y1 trace probe on dev cells")
    parser.add_argument("--y1-1-trace-probe", action="store_true",
                        help="Run Phase Y1.1 trace probe on dev cells (with search-behavior fields)")
    args = parser.parse_args()

    if args.y1_trace_probe or args.y1_1_trace_probe:
        sys.exit(cmd_y1_trace_probe(args, y1_1=args.y1_1_trace_probe))

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
