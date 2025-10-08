#!/usr/bin/env bash
set -e

echo "📊 Running evaluation..."
python src/eval.py --model-dir $(ls -td src/artifacts/mcic-* | head -1) --data-dir src/data
