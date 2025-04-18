#!/bin/bash

# This script runs pytest on all the subdirectories with a requirements.txt file.
# Arguments are passed to pytest.
# You must setup corresponding virtualenvs before running this script.

set -e  # Exit on error

root_dir="$(pwd)"

echo "🔍 Verifying each test file has pytest in the corresponding requirements.txt..."

# Find all test files excluding .venv, __pycache__, and node_modules
find . -type d \( -name .venv -o -name __pycache__ -o -name node_modules \) -prune -false -o -type f -name 'test_*.py' | while read -r test_file; do
  dir=$(dirname "$test_file")
  found_pytest=0

  # Traverse up until root directory or until we find a requirements.txt file
  while [[ "$dir" != "." && "$dir" != "/" ]]; do
    req_file="$dir/requirements.txt"
    if [[ -f "$req_file" ]]; then
      if grep -qi '^pytest' "$req_file"; then
        found_pytest=1
      fi
      break
    fi
    dir=$(dirname "$dir")
  done

  if [[ $found_pytest -eq 0 ]]; then
    echo "❌ Missing pytest in requirements.txt for $test_file"
    exit 1
  fi
done

echo "✅ All test files are properly linked to a pytest requirement."

find . -type f -name "requirements.txt" | while read -r req_file; do
  dir="$(dirname "$req_file")"

  # Check if pytest is in the requirements.txt file
  if ! grep -q "pytest" "$req_file"; then
    echo "❌ Skipping: $req_file (no pytest)"
    continue
  fi

  venv_path="$dir/.venv"

  echo "📦 Checking: $dir"

  # Check if the virtualenv exists
  if [[ ! -d "$venv_path" ]]; then
    echo "❌ Virtualenv not found: $venv_path"
    exit 1
  fi

  # Check if pytest is installed in the virtualenv
  if ! "$venv_path/bin/pip" show pytest > /dev/null 2>&1; then
    echo "❌ pytest not found in virtualenv: $venv_path"
    exit 1
  fi

  # Activate the virtualenv
  source "$venv_path/bin/activate"

  # Change the directory
  cd "$dir"

  # Run pytest
  echo "🔄 Running pytest in $dir..."
  pytest -v "$@"

  # Deactivate the virtualenv
  deactivate

  # Change back to the root directory
  cd "$root_dir"
done
