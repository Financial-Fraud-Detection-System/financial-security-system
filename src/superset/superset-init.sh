#!/bin/bash
# Adapted from: https://gist.github.com/malihasameen/218c4c81290ab431f6cc0872f948168c

# Upgrading Superset metastore
superset db upgrade

# Create admin user if not already created
superset fab create-admin \
  --username "$ADMIN_USERNAME" \
  --firstname Admin \
  --lastname User \
  --email "$ADMIN_EMAIL" \
  --password "$ADMIN_PASSWORD" || true

# setup roles and permissions
superset superset init

# Starting server
/bin/sh -c /usr/bin/run-server.sh
