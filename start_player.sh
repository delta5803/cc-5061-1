#!/usr/bin/env bash
# Start the DenseFlow motion-analysis player.
# First run? generate the built-in sequence results once:
#   python3 run_experiments.py
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
echo "Starting DenseFlow player on http://localhost:${PORT}/"
exec python3 -m denseflow.server
