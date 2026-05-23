from __future__ import annotations

import pkgutil
import re
from importlib import import_module
from typing import Callable

_ALPHA_MODULE_PATTERN = re.compile(r"^alpha_\d{3}$")


def _discover_formulaic_alphas() -> dict[str, Callable]:
    package_name = __package__ or "alpha_engine.formulaic_alphas"
    package = import_module(package_name)

    registry: dict[str, Callable] = {}

    for module_info in pkgutil.iter_modules(package.__path__):
        alpha_name = module_info.name

        if not _ALPHA_MODULE_PATTERN.fullmatch(alpha_name):
            continue

        module = import_module(f"{package_name}.{alpha_name}")
        function_name = f"compute_{alpha_name}"
        compute_func = getattr(module, function_name, None)

        if not callable(compute_func):
            raise AttributeError(
                f"{package_name}.{alpha_name} must define callable {function_name}"
            )

        registry[alpha_name] = compute_func

    return dict(sorted(registry.items()))


FORMULAIC_ALPHA_REGISTRY = _discover_formulaic_alphas()


def get_formulaic_alpha(name: str):
    key = name.strip().lower()
    if key not in FORMULAIC_ALPHA_REGISTRY:
        raise KeyError(
            f"Unknown alpha: {name}. Available: {sorted(FORMULAIC_ALPHA_REGISTRY)}"
        )
    return FORMULAIC_ALPHA_REGISTRY[key]


def list_formulaic_alphas() -> list[str]:
    return list(FORMULAIC_ALPHA_REGISTRY)
