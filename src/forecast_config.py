"""Load frozen forward-test dates; importing this module needs only the standard library."""
from pathlib import Path
from datetime import date,timedelta
import json
ROOT=Path(__file__).resolve().parents[1]

def load_config():
    cfg=json.loads((ROOT/'configs/forecast.json').read_text())
    keys=['train_end','validation_start','validation_end','test_start','test_end']
    if not all(cfg.get(k) for k in keys):
        raise ValueError('Freeze all forecast dates in configs/forecast.json after official data coverage audit.')
    d={k:date.fromisoformat(cfg[k]) for k in keys}
    if not (d['train_end']+timedelta(days=1)<d['validation_start']<=d['validation_end'] and
            d['validation_end']+timedelta(days=1)<d['test_start']<=d['test_end']):
        raise ValueError('Forward date ranges must be ordered with next-day boundary embargoes.')
    if cfg.get('seed')!=28:raise ValueError('This implementation uses fixed seed 28; update code and audit before changing it.')
    return cfg
