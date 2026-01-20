# Docker Training (PaST)

This guide provides a clean, reproducible workflow for running PaST training in Docker with persisted artifacts.

---

## Clone and build from GitHub

git clone <https://github.com/Abdellahbado/PaST.git>
cd PaST

(All commands below assume you are in the PaST repository root: the folder that contains `Dockerfile`.)

docker build -t past-train .

---

## Run (Linux/macOS)

Create a host folder for artifacts and run training:

mkdir -p artifacts

docker run --rm \
  --mount type=bind,source="$(pwd)/artifacts",target=/outputs \
  past-train \
  python -m PaST.train_q_sequence --variant_id q_sequence_cnn_ctx13 --device cpu --output_dir /outputs

ls -lah artifacts/

Results appear in ./artifacts/ (one subfolder per run, containing config.json, log.jsonl, checkpoints, best_model.pt, final_model.pt).

---

## Run using docker compose

Run from the repo root:

docker compose -f docker-compose.yml up --build

ls -lah artifacts/

---

## Change the training command/flags

### Option A: docker run (one-off overrides)

docker run --rm \
  --mount type=bind,source="$(pwd)/artifacts",target=/outputs \
  past-train \
  python -m PaST.train_q_sequence --variant_id q_sequence_ctx13 --seed 1 --device cpu --output_dir /outputs

### Option B: docker compose (edit command)

Edit the `command:` in docker-compose.yml, for example:

- Variant change: `q_sequence_ctx13` or `q_sequence_cnn_ctx13`
- Seed change: `--seed 1`
- Output dir change: `--output_dir /outputs` (recommended; guarantees host persistence)

---

## Troubleshooting

- If the bind-mount path does not exist, create it first: `mkdir -p artifacts`
- If files are owned by root on the host, run with: `--user "$(id -u):$(id -g)"`
