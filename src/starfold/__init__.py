"""Starfold: UMAP, HDBSCAN and Optuna population recovery.

One Optuna search selects a UMAP and HDBSCAN configuration by a label-free
score, and a full-data fit of it is the emitted partition. Possible worlds
measure its reproducibility, a bounded co-association check tests it without
matching labels, and a matched Monte Carlo null calibrates the claim.

The public surface is loaded lazily so that ``import starfold`` does not
import umap-learn.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

_EXPORTS: dict[str, str] = {
    "recover": "starfold.api",
    "Config": "starfold.api",
    "Space": "starfold.api",
    "search": "starfold.search",
    "SearchResult": "starfold.search",
    "WorldEvidence": "starfold.worlds",
    "null_test": "starfold.api",
    "NullEvidence": "starfold.nulls",
    "generate_null": "starfold.nulls",
    "compare_backends": "starfold.backend",
    "BackendAgreement": "starfold.backend",
    "Params": "starfold.backend",
    "default_params": "starfold.backend",
    "Result": "starfold.result",
    "save": "starfold.result",
    "load": "starfold.result",
}

__all__ = [
    "BackendAgreement",
    "Config",
    "NullEvidence",
    "Params",
    "Result",
    "SearchResult",
    "Space",
    "WorldEvidence",
    "__version__",
    "compare_backends",
    "default_params",
    "generate_null",
    "load",
    "null_test",
    "recover",
    "save",
    "search",
]

if TYPE_CHECKING:
    from starfold.api import Config, Space, null_test, recover
    from starfold.backend import BackendAgreement, Params, compare_backends, default_params
    from starfold.nulls import NullEvidence, generate_null
    from starfold.result import Result, load, save
    from starfold.search import SearchResult, search
    from starfold.worlds import WorldEvidence


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy attribute dispatch
    """Resolve a public name from its defining module on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        msg = f"module 'starfold' has no attribute {name!r}"
        raise AttributeError(msg)
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public surface."""
    return sorted(set(globals()) | set(_EXPORTS))
