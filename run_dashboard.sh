#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/dashboard"
python3 -m streamlit run streamlit_app.py
