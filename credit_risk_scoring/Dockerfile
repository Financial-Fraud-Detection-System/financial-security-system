# Stage 1: Build environment
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./credit_risk_scoring/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt


# Stage 2: Runtime image
FROM python:3.10-slim

WORKDIR /app

ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="."

# Copy only necessary files
COPY --from=builder /root/.local /root/.local
COPY ./credit_risk_scoring ./credit_risk_scoring
COPY ./libs ./libs

CMD ["python", "credit_risk_scoring/main.py"]
