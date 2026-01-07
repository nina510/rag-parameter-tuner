#!/bin/bash
cd "$(dirname "$0")"
echo "Starting RAG Parameter Tuning Tool..."
echo ""
echo "Installing dependencies (first time only)..."
pip3 install -r requirements.txt
echo ""
echo "Starting server..."
python3 app.py

