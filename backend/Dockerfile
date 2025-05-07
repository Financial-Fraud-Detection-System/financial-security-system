# Stage 1: Build environment
FROM python:3.13-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./backend/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# Stage 2: Runtime image
FROM python:3.13-slim

WORKDIR /app

ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="."

# Copy only necessary files
COPY --from=builder /root/.local /root/.local
COPY ./backend ./backend
COPY ./libs ./libs

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
