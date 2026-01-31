from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

import torch


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Watches a PPO checkpoint (e.g. *_latest.pt) and prints progress whenever it changes. "
            "Useful for monitoring nohup runs without stopping them."
        )
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint .pt (typically <out_dir>/<run_name>_latest.pt).",
    )
    p.add_argument(
        "--poll",
        type=float,
        default=5.0,
        help="Poll interval in seconds.",
    )
    p.add_argument(
        "--jsonl_out",
        type=str,
        default="",
        help="Optional JSONL output path (appends a record each time checkpoint changes).",
    )
    args = p.parse_args()

    ckpt_path = os.path.expanduser(str(args.ckpt))
    poll_s = max(0.2, float(args.poll))
    jsonl_out = str(args.jsonl_out).strip()

    last_mtime: float | None = None

    while True:
        try:
            st = os.stat(ckpt_path)
        except FileNotFoundError:
            print(f"Waiting for checkpoint: {ckpt_path}", flush=True)
            time.sleep(poll_s)
            continue

        mtime = float(st.st_mtime)
        if last_mtime is not None and mtime <= last_mtime:
            time.sleep(poll_s)
            continue

        last_mtime = mtime
        try:
            ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"Failed to load {ckpt_path}: {e}", flush=True)
            time.sleep(poll_s)
            continue

        upd = ckpt.get("update", None)
        lm = ckpt.get("last_metrics", None)

        # Minimal always-available line.
        msg = f"ckpt_update={upd} mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}"

        # If trainer is new enough, last_metrics gives the rich fields.
        if isinstance(lm, dict):
            best_p50 = _safe_float(lm.get("best_energy_p50"))
            best_mean = _safe_float(lm.get("best_energy_mean"))
            accept = _safe_float(lm.get("accept_rate"))
            feasible = _safe_float(lm.get("feasible_rate"))
            sps = _safe_float(lm.get("steps_per_sec"))

            parts = []
            if best_p50 is not None:
                parts.append(f"bestE_p50={best_p50:.1f}")
            if best_mean is not None:
                parts.append(f"bestE_mean={best_mean:.1f}")
            if accept is not None:
                parts.append(f"accept={accept:.3f}")
            if feasible is not None:
                parts.append(f"feasible={feasible:.3f}")
            if sps is not None:
                parts.append(f"steps/s={sps:.1f}")

            if parts:
                msg += " " + " ".join(parts)

        print(msg, flush=True)

        if jsonl_out:
            try:
                os.makedirs(os.path.dirname(jsonl_out) or ".", exist_ok=True)
                with open(jsonl_out, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "time": time.time(),
                                "ckpt": ckpt_path,
                                "update": upd,
                                "mtime": mtime,
                                "last_metrics": lm if isinstance(lm, dict) else None,
                            }
                        )
                        + "\n"
                    )
            except Exception as e:
                print(f"Failed to write jsonl_out: {e}", flush=True)

        time.sleep(poll_s)


if __name__ == "__main__":
    main()
