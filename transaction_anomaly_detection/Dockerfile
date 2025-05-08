# Stage 1: Build environment
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./transaction_anomaly_detection/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# Stage 2: Runtime image
FROM python:3.12-slim

WORKDIR /app

ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="."

# Copy only necessary files
COPY --from=builder /root/.local /root/.local
COPY ./transaction_anomaly_detection ./transaction_anomaly_detection
COPY ./libs ./libs

CMD ["python", "transaction_anomaly_detection/main.py"]
