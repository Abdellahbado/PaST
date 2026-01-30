from __future__ import annotations

import argparse
import random

from PaST.solvers.alns_parallel import ALNSConfig
from PaST.alns_rl.pm_alns_env import PMALNSEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=str, default="medium")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--instance_id", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--slack_ratio", type=float, default=0.25)
    args = ap.parse_args()

    env = PMALNSEnv(
        scale=str(args.scale),
        seed=int(args.seed),
        instance_id=int(args.instance_id),
        slack_ratio=float(args.slack_ratio),
        alns_cfg=ALNSConfig(
            max_iters=int(args.steps), no_improve_limit=int(args.steps)
        ),
    )

    obs = env.reset()
    print("obs_dim=", obs.shape[0], "action_dim=", env.action_dim)

    total_r = 0.0
    rng = random.Random(int(args.seed) + 1)
    done = False
    t = 0
    while not done and t < int(args.steps):
        a = rng.randrange(env.action_dim)
        obs, r, done, info = env.step(a)
        total_r += float(r)
        if (t + 1) % 10 == 0 or done:
            s = info["step"]
            print(
                f"t={t+1} r={r:+.3f} total_r={total_r:+.3f} accepted={s['accepted']} bestE={s['best_energy']:.2f}"
            )
        t += 1


if __name__ == "__main__":
    main()
