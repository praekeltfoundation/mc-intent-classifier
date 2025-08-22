#!/usr/bin/env bash
set -e

echo "🚀 Training model..."
python src/train.py --data-dir src/data --artifacts-dir src/artifacts --encoder BAAI/bge-m3
