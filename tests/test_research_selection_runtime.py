from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aerolife_runtime import RULPredictor, RAW_COLS, SUBSETS
for ds in SUBSETS:
    raw=pd.read_csv(ROOT/'data'/'nasa_test'/f'test_{ds}.txt',sep=r'\s+',header=None)
    raw.columns=RAW_COLS
    predictor=RULPredictor(ds,root=ROOT)
    out=predictor.predict(raw)
    assert len(out)==raw.unit.nunique(), (ds,len(out),raw.unit.nunique())
    vals=out[['catboost_prediction','xgboost_prediction','lightgbm_prediction','predicted_rul','rul_lower_90','rul_upper_90']].to_numpy(float)
    assert np.isfinite(vals).all(), ds
    assert (out.rul_lower_90 <= out.predicted_rul).all(), ds
    assert (out.predicted_rul <= out.rul_upper_90).all(), ds
    assert out.ensemble.str.contains('Research Selection').all(), ds
    print(f'{ds}: PASS | engines={len(out)} | cap={predictor.bundle["cap"]} | weights={[round(x[0],4) for x in [(m["weight"],m["member_id"]) for m in predictor.bundle["members"]]]}')
print('All Research Selection runtime checks passed. No test-set accuracy metric was calculated in this smoke test.')
