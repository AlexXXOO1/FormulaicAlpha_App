from __future__ import annotations

from alpha_engine.formulaic_alphas.alpha_001 import compute_alpha_001

FORMULAIC_ALPHA_REGISTRY = {
    "alpha_001": compute_alpha_001,
}


def get_formulaic_alpha(name: str):
    key = name.strip().lower()
    if key not in FORMULAIC_ALPHA_REGISTRY:
        raise KeyError(f"Unknown alpha: {name}. Available: {sorted(FORMULAIC_ALPHA_REGISTRY)}")
    return FORMULAIC_ALPHA_REGISTRY[key]
