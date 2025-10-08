#!/usr/bin/env bash
set -e

echo "🌐 Starting Flask service..."
gunicorn --bind 0.0.0.0:8000 src.application:app
