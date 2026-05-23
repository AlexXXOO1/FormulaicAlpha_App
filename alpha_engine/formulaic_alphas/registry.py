from __future__ import annotations

from alpha_engine.formulaic_alphas.alpha_001 import compute_alpha_001
from alpha_engine.formulaic_alphas.alpha_002 import compute_alpha_002
from alpha_engine.formulaic_alphas.alpha_003 import compute_alpha_003
from alpha_engine.formulaic_alphas.alpha_004 import compute_alpha_004
from alpha_engine.formulaic_alphas.alpha_005 import compute_alpha_005
from alpha_engine.formulaic_alphas.alpha_006 import compute_alpha_006
from alpha_engine.formulaic_alphas.alpha_007 import compute_alpha_007
from alpha_engine.formulaic_alphas.alpha_008 import compute_alpha_008
from alpha_engine.formulaic_alphas.alpha_009 import compute_alpha_009
from alpha_engine.formulaic_alphas.alpha_010 import compute_alpha_010

FORMULAIC_ALPHA_REGISTRY = {
    "alpha_001": compute_alpha_001,
    "alpha_002": compute_alpha_002,
    "alpha_003": compute_alpha_003,
    "alpha_004": compute_alpha_004,
    "alpha_005": compute_alpha_005,
    "alpha_006": compute_alpha_006,
    "alpha_007": compute_alpha_007,
    "alpha_008": compute_alpha_008,
    "alpha_009": compute_alpha_009,
    "alpha_010": compute_alpha_010,
}


def get_formulaic_alpha(name: str):
    key = name.strip().lower()
    if key not in FORMULAIC_ALPHA_REGISTRY:
        raise KeyError(f"Unknown alpha: {name}. Available: {sorted(FORMULAIC_ALPHA_REGISTRY)}")
    return FORMULAIC_ALPHA_REGISTRY[key]
