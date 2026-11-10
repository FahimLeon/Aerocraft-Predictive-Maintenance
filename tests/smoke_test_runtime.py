"""Artifact-load and inference smoke test. Synthetic values only; not an accuracy test."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aerolife_runtime import RULPredictor, RAW_COLS, SUBSETS

rng = np.random.default_rng(2026)
rows = []
for unit in (1, 2):
    for cycle in range(1, 41):
        row = {"unit": unit, "cycle": cycle}
        row.update({f"os{i}": float(rng.normal()) for i in range(1, 4)})
        row.update({f"s{i}": float(rng.normal(500 + i, 2)) for i in range(1, 22)})
        rows.append(row)
history = pd.DataFrame(rows, columns=RAW_COLS)

for dataset in SUBSETS:
    predictor = RULPredictor(dataset, root=ROOT)
    result = predictor.predict(history)
    assert len(result) == 2
    assert result["engine"].tolist() == [1, 2]
    assert result["last_cycle"].tolist() == [40, 40]
    assert np.isfinite(result[["predicted_rul", "rul_lower_90", "rul_upper_90"]].to_numpy()).all()
    assert (result["rul_lower_90"] <= result["predicted_rul"]).all()
    assert (result["predicted_rul"] <= result["rul_upper_90"]).all()
    print(f"{dataset}: PASS — bundle loaded; 2 synthetic engines predicted")
print("Runtime smoke test PASS. Synthetic data are not suitable for accuracy claims.")
