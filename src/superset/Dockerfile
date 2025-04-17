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
 && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir pillow psycopg2

# Copy init script
COPY ./superset-init.sh /superset-init.sh
RUN chmod +x /superset-init.sh

# Copy Superset config
COPY superset_config.py /app/superset_config.py
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
