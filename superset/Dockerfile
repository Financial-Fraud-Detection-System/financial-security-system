FROM apache/superset:latest

# Switch to root user to install dependencies
USER root

# Install build system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    build-essential \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir pillow psycopg2 clickhouse-connect

# Copy init script
COPY ./superset/superset-init.sh /superset-init.sh

RUN dos2unix /superset-init.sh \
    && chmod +x /superset-init.sh \
    && apt-get remove -y dos2unix \
    && apt-get autoremove -y

# Copy Superset config
COPY ./superset/superset_config.py /app/superset_config.py
ENV SUPERSET_CONFIG_PATH=/app/superset_config.py

# Optional environment variables for admin user and DB URI
ENV ADMIN_USERNAME=${ADMIN_USERNAME}
ENV ADMIN_EMAIL=${ADMIN_EMAIL}
ENV ADMIN_PASSWORD=${ADMIN_PASSWORD}
ENV SUPERSET_DATABASE_URI=${SUPERSET_DATABASE_URI}

# Create a non-root user and switch to it
USER superset

# Shell entrypoint
ENTRYPOINT ["sh", "/superset-init.sh"]
