# PaST on Docker (A100 / long runs)

This folder contains everything needed to run PaST (PPO + REINFORCE variants) inside Docker with NVIDIA GPUs.

## Prereqs (host)

- Docker installed
- NVIDIA driver installed
- NVIDIA Container Toolkit installed (`nvidia-container-toolkit`)

Quick check:
- `nvidia-smi` works on the host
- `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` works

## Build (from GitHub clone)

From the repo root:

- `docker build -t past:latest -f PaST/Dockerfile PaST`

This uses `PaST/` as the build context (so other folders aren’t copied).

## Run all variants (recommended)

From `PaST/`:

- `docker compose up --build`

Outputs persist to:
- `PaST/runs/` (mounted to `/workspace/runs` inside the container)

Stop:
- `docker compose down`

## Custom runs

Run a single variant (PPO example):

- `docker run --rm --gpus all -v "$PWD/runs:/workspace/runs" past:latest \
  python -m PaST.train_ppo --variant_id ppo_full_global --config PaST/configs/a100_full.yaml \
  --output_dir runs --eval_seed 1337 --eval_seed_mode per_update`

Run the full suite with multiple seeds:

- `docker run --rm --gpus all -v "$PWD/runs:/workspace/runs" past:latest \
  python -m PaST.run_experiments --variants all --seeds 0 1 2 \
  --config PaST/configs/a100_full.yaml --output_dir runs \
  --eval_seed 1337 --eval_seed_mode per_update`

## Parallel runs (multi-GPU nodes)

If the server has multiple GPUs, `PaST.run_experiments` can schedule jobs across them:

- `python -m PaST.run_experiments --variants all --seeds 0 1 \
  --config PaST/configs/a100_full.yaml --output_dir runs \
  --gpus 0,1 --max_parallel 2`

Notes:
- With a single A100, keep `--max_parallel 1`.
- Parallel mode sets `CUDA_VISIBLE_DEVICES` per job.

## Fair comparison: same eval instances across all methods

Pass `--eval_seed` to force the *same evaluation instances* for every variant.

- `--eval_seed_mode fixed`: always evaluate on the exact same instances.
- `--eval_seed_mode per_update`: evaluate on a different (but deterministic) batch each eval step.

This is implemented in:
- `PaST/train_ppo.py`
- `PaST/train_reinforce.py`

## Monitoring

- Watch metrics: `tail -f PaST/runs/<variant>/seed_<seed>/metrics.jsonl`
- Best checkpoints: `PaST/runs/<variant>/seed_<seed>/checkpoints/best.pt`
