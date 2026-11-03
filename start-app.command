#!/bin/bash
# Double-click in Finder to start the photo printer app.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  python3 setup_check.py
fi
exec .venv/bin/python thermal.py app
