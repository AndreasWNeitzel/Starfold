"""Exercise an installed Starfold from outside the source tree.

The tests import from ``src/`` and cannot catch a file missing from the wheel.
Run this with only the built wheel installed (CI does). It exits 1 if
``starfold`` resolves inside the repository or its version differs from the
installed metadata, then runs one bounded recovery on a synthetic two-blob
fixture, so a missing dependency also fails here.
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

import starfold

EXPECTED_GROUPS = 2


def main() -> int:
    """Import the installed package and run one bounded recovery."""
    location = Path(starfold.__file__).resolve()
    repository = Path(__file__).resolve().parent.parent
    if repository in location.parents:
        sys.stdout.write(
            f"starfold imported from the repository ({location}), not from an installed wheel\n"
        )
        return 1
    installed = version("starfold")
    if starfold.__version__ != installed:
        sys.stdout.write(
            f"starfold.__version__ is {starfold.__version__}, the wheel metadata says {installed}\n"
        )
        return 1

    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0.0, 1.0, (120, 4)), rng.normal(8.0, 1.0, (120, 4))])
    config = starfold.Config(n_trials=3, n_explore=2, n_worlds=2, null_families=())
    result = starfold.recover(x, config=config, random_state=0, engine="cpu")

    if result.labels.shape != (x.shape[0],):
        sys.stdout.write(f"labels have shape {result.labels.shape}, expected {(x.shape[0],)}\n")
        return 1
    if result.n_groups != EXPECTED_GROUPS:
        sys.stdout.write(f"two separated blobs returned {result.n_groups} groups\n")
        return 1

    sys.stdout.write(
        f"wheel ok: starfold {starfold.__version__} from {location.parent}, "
        f"{result.n_groups} groups, flags {list(result.flags)}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
