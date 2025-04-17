#!/bin/bash

# This script exports Superset dashboards and datasources from the running Docker container from the host.

set -e  # Exit on error

# Variables
CONTAINER_NAME=financial-security-system-superset
EXPORT_PATH_IN_CONTAINER=/tmp/superset_exports
EXPORT_PATH_ON_HOST=src/superset/exports

echo "🧹 Pre clean-up temporary files in container..."
docker exec "$CONTAINER_NAME" bash -c "
  rm -rf $EXPORT_PATH_IN_CONTAINER
"

echo "📦 Exporting Superset dashboards from container '$CONTAINER_NAME'..."

docker exec "$CONTAINER_NAME" bash -c "
  mkdir -p $EXPORT_PATH_IN_CONTAINER &&
  cd $EXPORT_PATH_IN_CONTAINER &&
  superset export-dashboards
"

echo "📦 Exporting Superset datasources from container '$CONTAINER_NAME'..."

docker exec "$CONTAINER_NAME" bash -c "
  mkdir -p $EXPORT_PATH_IN_CONTAINER &&
  cd $EXPORT_PATH_IN_CONTAINER &&
  superset export-datasources
"

echo "📤 Copying exported files from container to host..."

docker cp "$CONTAINER_NAME:$EXPORT_PATH_IN_CONTAINER/." "$EXPORT_PATH_ON_HOST"

echo "🧹 Post clean-up temporary files in container..."

docker exec "$CONTAINER_NAME" bash -c "
  rm -rf $EXPORT_PATH_IN_CONTAINER
"

echo "✅ Export complete. File is available at: $EXPORT_PATH_ON_HOST"
