# Base image: CPU-friendly Python
FROM python:3.11-slim

# Workdir (we keep /app as the parent so that /app/PaST is importable)
WORKDIR /app

# Copy repo into /app/PaST so that `python -m PaST.*` works
COPY . /app/PaST

# Install dependencies
RUN pip install --no-cache-dir -r /app/PaST/requirements.txt

# Ensure outputs directory exists
RUN mkdir -p /outputs

# Default output directory inside container
ENV PAST_OUTPUT_DIR=/outputs

# Make sure the `PaST` package (copied into /app/PaST) is importable.
ENV PYTHONPATH=/app

# Default command (can be overridden at runtime)
CMD ["python", "-m", "PaST.train_q_sequence", "--variant_id", "q_sequence_cnn_ctx13", "--device", "cpu", "--output_dir", "/outputs"]
