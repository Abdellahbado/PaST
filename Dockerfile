# Base image: CPU-friendly Python
FROM python:3.11-slim

# Workdir
WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy repo
COPY . /app

# Ensure outputs directory exists
RUN mkdir -p /outputs

# Default output directory inside container
ENV PAST_OUTPUT_DIR=/outputs

# Default command (can be overridden at runtime)
CMD ["python", "-m", "PaST.train_q_sequence", "--variant_id", "q_sequence_cnn_ctx13", "--device", "cpu"]
