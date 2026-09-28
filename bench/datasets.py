"""Labelled real datasets for the bench, loaded with a pinned schema.

Loaders fail closed on schema drift (feature or class count) so that a
silently changed upstream file cannot enter a result. Files are cached under
``STARFOLD_DATA_CACHE`` (default ``~/.cache/starfold``). Labels are returned
separately and are never passed to the procedure under test.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.datasets import fetch_openml, load_digits

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

CACHE = Path(os.environ.get("STARFOLD_DATA_CACHE", "~/.cache/starfold")).expanduser()


EXAMPLE_TABLE = Path(__file__).with_name("data") / "stellar_chemokinematics_apogee_dr19.npz"
EXAMPLE_SHA256 = "5bd5c78bd1eadaf5d38801fda962c8433922d67bc18aa20a44ea6c47023d411a"
EXAMPLE_FEATURES = ("fe_h", "alpha_m", "mg_fe", "J_R", "J_z", "L_z", "ecc", "E")
EXAMPLE_ROWS = 9242


@dataclass(frozen=True)
class Dataset:
    """One labelled dataset with its provenance."""

    name: str
    X: NDArray[np.float64]
    truth: NDArray[np.int64]
    n_classes: int
    source: str
    sha256: str | None


def example_table(*, verify: bool = True) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """The APOGEE DR19 chemokinematic table, 9,242 stars and 8 features.

    Real ASPCAP abundances and galpy-integrated orbital quantities, with no
    labels; see ``bench/data/stellar_chemokinematics_apogee_dr19.md``.

    Parameters
    ----------
    verify : bool, default True
        Check the file digest, the row count and the column names before
        returning, so a changed file cannot enter a result silently.

    Raises
    ------
    FileNotFoundError
        If the table is absent, which is the case for an installed wheel; it
        ships with the repository, not with the package.
    ValueError
        If ``verify`` and the file does not match what the provenance records.
    """
    if not EXAMPLE_TABLE.exists():
        msg = (
            f"{EXAMPLE_TABLE} is missing. The example table ships with the "
            "repository, not with the installed package; clone the repository "
            "to run the bench and the notebooks."
        )
        raise FileNotFoundError(msg)
    with np.load(EXAMPLE_TABLE, allow_pickle=False) as bundle:
        x = np.asarray(bundle["X"], dtype=np.float64)
        names = tuple(str(name) for name in bundle["feature_names"])
    if verify:
        digest = _file_sha256(EXAMPLE_TABLE)
        if (digest, x.shape, names) != (
            EXAMPLE_SHA256,
            (EXAMPLE_ROWS, len(EXAMPLE_FEATURES)),
            EXAMPLE_FEATURES,
        ):
            msg = (
                f"{EXAMPLE_TABLE} does not match its provenance: digest "
                f"{digest[:16]}, shape {x.shape}, names {names}."
            )
            raise ValueError(msg)
    return x, names


def _encode_labels(values: NDArray[np.generic]) -> NDArray[np.int64]:
    """Map source classes to compact integers."""
    _, encoded = np.unique(np.asarray(values), return_inverse=True)
    return np.asarray(encoded, dtype=np.int64)


def _file_sha256(path: Path) -> str:
    """SHA-256 of a cached file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(name: str, url: str) -> Path:
    """Return the cached archive, downloading it once if absent."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,  # noqa: S310 - pinned UCI URL
                temporary.open("wb") as sink,
            ):
                shutil.copyfileobj(response, sink)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _arff_archive(
    *,
    name: str,
    archive_name: str,
    url: str,
    member: str,
    expected_features: int,
    expected_classes: int,
) -> Dataset:
    """Read a pinned numeric UCI ARFF member from a zip archive."""
    archive = _download(archive_name, url)
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as raw:
        handle = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        in_data = False
        rows: list[list[str]] = []
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            if not in_data:
                in_data = stripped.lower() == "@data"
                continue
            rows.extend(csv.reader([stripped]))
    expected_columns = expected_features + 1
    if not rows or any(len(row) != expected_columns for row in rows):
        msg = f"{name}: ARFF schema drift, expected {expected_columns} columns."
        raise ValueError(msg)
    features = np.asarray(
        [[float(value) for value in row[:expected_features]] for row in rows], dtype=np.float64
    )
    truth = _encode_labels(np.asarray([row[-1] for row in rows]))
    if np.unique(truth).size != expected_classes:
        msg = f"{name}: label drift, received {np.unique(truth).size} classes."
        raise ValueError(msg)
    return Dataset(name, features, truth, expected_classes, url, _file_sha256(archive))


def _openml(*, name: str, data_id: int, expected_features: int, expected_classes: int) -> Dataset:
    """Load a pinned OpenML numeric dataset by identifier."""
    data = fetch_openml(
        data_id=data_id, as_frame=False, parser="auto", data_home=str(CACHE / "openml")
    )
    features = np.asarray(data.data, dtype=np.float64)
    truth = _encode_labels(np.asarray(data.target))
    if features.ndim != 2 or features.shape[1] != expected_features:
        msg = f"{name}: schema drift, expected {expected_features} features, got {features.shape}."
        raise ValueError(msg)
    if np.unique(truth).size != expected_classes:
        msg = f"{name}: label drift, received {np.unique(truth).size} classes."
        raise ValueError(msg)
    return Dataset(name, features, truth, expected_classes, f"openml:{data_id}", None)


def digits() -> Dataset:
    """scikit-learn Digits, 1,797 x 64, 10 classes."""
    data = load_digits()
    return Dataset(
        "digits",
        np.asarray(data.data, dtype=np.float64),
        np.asarray(data.target, dtype=np.int64),
        10,
        "sklearn:load_digits",
        None,
    )


def rice() -> Dataset:
    """UCI Rice Cammeo and Osmancik, doi:10.24432/C5MW4Z, 3,810 x 7, 2 classes."""
    return _arff_archive(
        name="rice",
        archive_name="rice_cammeo_osmancik.zip",
        url="https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip",
        member="Rice_Cammeo_Osmancik.arff",
        expected_features=7,
        expected_classes=2,
    )


def dry_bean() -> Dataset:
    """UCI Dry Bean, doi:10.24432/C50S4B, 13,611 x 16, 7 classes."""
    return _arff_archive(
        name="dry_bean",
        archive_name="dry_bean_dataset.zip",
        url="https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
        member="DryBeanDataset/Dry_Bean_Dataset.arff",
        expected_features=16,
        expected_classes=7,
    )


def pendigits() -> Dataset:
    """OpenML PenDigits (id 32), 10,992 x 16, 10 classes."""
    return _openml(name="pendigits", data_id=32, expected_features=16, expected_classes=10)


def landsat() -> Dataset:
    """OpenML Landsat satellite (id 182), 6,430 x 36, 6 classes."""
    return _openml(name="landsat", data_id=182, expected_features=36, expected_classes=6)


def waveform() -> Dataset:
    """OpenML Waveform (id 60), 5,000 x 40, 3 classes."""
    return _openml(name="waveform", data_id=60, expected_features=40, expected_classes=3)


def coil20() -> Dataset:
    """OpenML COIL-20 (id 46783), 1,440 x 1024, 20 classes."""
    return _openml(name="coil20", data_id=46_783, expected_features=1024, expected_classes=20)


def isolet() -> Dataset:
    """OpenML Isolet (id 300), 7,797 x 617, 26 classes."""
    return _openml(name="isolet", data_id=300, expected_features=617, expected_classes=26)


def gas_drift() -> Dataset:
    """OpenML gas sensor drift (id 1476), 13,910 x 128, 6 classes."""
    return _openml(name="gas_drift", data_id=1_476, expected_features=128, expected_classes=6)


def letter() -> Dataset:
    """OpenML Letter (id 6), 20,000 x 16, 26 classes."""
    return _openml(name="letter", data_id=6, expected_features=16, expected_classes=26)


DEVELOPMENT: dict[str, Callable[[], Dataset]] = {
    "rice": rice,
    "dry_bean": dry_bean,
    "landsat": landsat,
    "digits": digits,
    "pendigits": pendigits,
    "waveform": waveform,
}
SEALED: dict[str, Callable[[], Dataset]] = {
    "coil20": coil20,
    "isolet": isolet,
    "gas_drift": gas_drift,
    "letter": letter,
}
LOADERS: dict[str, Callable[[], Dataset]] = {**DEVELOPMENT, **SEALED}


def load(name: str) -> Dataset:
    """Load a dataset by name."""
    try:
        return LOADERS[name]()
    except KeyError as error:
        msg = f"Unknown dataset {name!r}; known: {sorted(LOADERS)}."
        raise ValueError(msg) from error


def rows(name: str) -> NDArray[np.float64]:
    """Feature matrix of one case by name; ``"apogee"`` is the example table."""
    if name == "apogee":
        return example_table()[0]
    return load(name).X
