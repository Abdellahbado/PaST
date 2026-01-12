# PaST container (GPU-enabled)
# Build from repo root (recommended: PaST-only context):
#   docker build -f PaST/Dockerfile -t past:latest PaST

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Copy only PaST (build context should be the PaST folder)
COPY . /workspace/PaST

# Install Python deps, but do NOT reinstall torch (already provided by base image)
RUN python -m pip install --upgrade pip

RUN python - <<'PY'
import pathlib

src = pathlib.Path('PaST/requirements.txt').read_text().splitlines()
lines = []
for raw in src:
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    if line == 'torch' or line.startswith('torch==') or line.startswith('torch>='):
        continue
    lines.append(line)
pathlib.Path('PaST/requirements.docker.txt').write_text('\n'.join(lines) + '\n')
print('Wrote PaST/requirements.docker.txt')
PY

RUN python -m pip install -r PaST/requirements.docker.txt

# Default entrypoint runs the experiment suite; override by passing a command.
ENTRYPOINT ["sh", "/workspace/PaST/docker/entrypoint.sh"]
