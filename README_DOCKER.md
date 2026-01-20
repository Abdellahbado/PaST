# Docker Training (Artifacts Persisted to Host)

This guide is for supervisors who need repeatable training runs and **persistent artifacts**.

## What a “mount” means (plain language)

The container writes training results to **/outputs**. We “mount” your local folder **./artifacts** into that location, so anything the container writes to **/outputs** appears on your host in **./artifacts** and stays there after the container exits.

---

## Quick start (Linux/macOS)

Run these commands from the repo root.

1. Create a host folder for artifacts:

given command:
mkdir -p artifacts

1. Build the Docker image:

given command:
docker build -t past-train -f PaST/Dockerfile .

1. Run training with a bind mount:

given command:
docker run --rm \
  --mount type=bind,source="$(pwd)/artifacts",target=/outputs \
  past-train

1. Find results on your host:

- Look in ./artifacts/
- Each run creates a unique subfolder (for example: q_sequence_cnn_ctx13_s0_20260120_205658/)
- Inside each run folder you will find:
  - config.json
  - log.jsonl
  - checkpoint_*.pt
  - best_model.pt
  - final_model.pt

---

## Run using docker compose

Run from the PaST/ folder:

given command:
docker compose up --build

Results will appear in PaST/artifacts/ on the host.

---

## How to change the training command/flags

### Option A: docker run (one-off overrides)

Change the command after the image name:

given command:
docker run --rm \
  --mount type=bind,source="$(pwd)/artifacts",target=/outputs \
  past-train \
  python -m PaST.train_q_sequence --variant_id q_sequence_ctx13 --seed 1 --device cpu

### Option B: docker compose (edit command)

Edit the `command:` in docker-compose.yml, for example:

- Variant change: `q_sequence_ctx13` or `q_sequence_cnn_ctx13`
- Seed change: `--seed 1`
- Output dir change: `--output_dir /outputs` (default in Docker)

---

## Troubleshooting

- **Bind mount path does not exist:**
  - Docker will fail if the host path doesn’t exist.
  - Fix: create it first: `mkdir -p artifacts`

- **Files owned by root on the host:**
  - This can happen if Docker runs as root.
  - Optional fix: run the container as your user:
    - `--user "$(id -u):$(id -g)"`

---

## Notes

- Training scripts accept `--output_dir` and default to `/outputs` in Docker via `PAST_OUTPUT_DIR`.
- The container always writes to /outputs, which maps to ./artifacts on the host.
