from __future__ import annotations

import pytest

from alpha_engine.formulaic_alphas.registry import (
    FORMULAIC_ALPHA_REGISTRY,
    get_formulaic_alpha,
    list_formulaic_alphas,
)


def test_registry_auto_discovers_alpha_modules():
    assert "alpha_001" in FORMULAIC_ALPHA_REGISTRY
    assert "alpha_010" in FORMULAIC_ALPHA_REGISTRY
    assert "alpha_011" in FORMULAIC_ALPHA_REGISTRY

    assert get_formulaic_alpha("alpha_011") is FORMULAIC_ALPHA_REGISTRY["alpha_011"]
    assert get_formulaic_alpha(" ALPHA_011 ") is FORMULAIC_ALPHA_REGISTRY["alpha_011"]


def test_registry_functions_follow_naming_contract():
    for alpha_name, compute_func in FORMULAIC_ALPHA_REGISTRY.items():
        assert callable(compute_func)
        assert compute_func.__name__ == f"compute_{alpha_name}"


def test_list_formulaic_alphas_is_sorted_registry_key_list():
    assert list_formulaic_alphas() == sorted(FORMULAIC_ALPHA_REGISTRY)


def test_registry_unknown_alpha_error_lists_available_names():
    with pytest.raises(KeyError, match="Unknown alpha"):
        get_formulaic_alpha("alpha_999")
