"""DEPRECATED — superseded by ``dashboard/generate_predictions.py``.

The dashboard prediction table is now produced by the tuned-model script under
``dashboard/`` (walk-forward production fold, graph-hub features, tuned
hyper-parameters from ``PRT661_outputs/models/best_hyperparameters.json``).

This shim forwards so old commands / ``run_pipeline.sh`` keep working.
"""
import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "dashboard" / "generate_predictions.py"

if __name__ == "__main__":
    print(f"[deprecated] delegating to {TARGET}", flush=True)
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
