#!/bin/bash

# Sync all .venv environments with their respective requirements.txt files
# Reinstall packages only if requirements.txt has changed

set -e  # Exit on error

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
  echo "❌ pyenv is not installed. Please install it first."
  exit 1
fi

find . -type f -name "requirements.txt" | while read -r req_file; do
  dir="$(dirname "$req_file")"
  venv_path="$dir/.venv"
  hash_file="$dir/.requirements.hash"

  echo "📦 Checking: $req_file"

  # Calculate current hash
  current_hash=$(sha256sum "$req_file" | awk '{print $1}')

  # Read existing hash if present
  if [[ -f "$hash_file" ]]; then
    old_hash=$(cat "$hash_file")
  else
    old_hash=""
  fi

  # Compare hashes
  if [[ "$current_hash" != "$old_hash" || ! -d "$venv_path" ]]; then
    echo "🔄 Changes detected or venv missing. Syncing $venv_path..."

    # Check for .pythonversion file
    python_version_file="$dir/.pythonversion"
    if [[ -f "$python_version_file" ]]; then
      python_version=$(cat "$python_version_file" | tr -d '[:space:]')
      echo "🐍 Using Python version: $python_version"
      export PYENV_VERSION="$python_version"
    else
      echo "🐍 No .pythonversion file found. Using default Python 3 version."
      export PYENV_VERSION="3"
    fi

    # Create or recreate venv
    echo "🔨 Creating venv"
    pyenv exec python -m venv "$venv_path"
    "$venv_path/bin/pip" install --upgrade pip
    "$venv_path/bin/pip" install -r "$req_file"

    # Save new hash
    echo "$current_hash" > "$hash_file"

    echo "✅ Synced: $dir"
  else
    echo "✅ Already up-to-date: $dir"
  fi

  echo "-----------------------------------------"
done
