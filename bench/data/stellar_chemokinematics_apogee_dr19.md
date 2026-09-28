# stellar_chemokinematics_apogee_dr19 (npz) provenance

Real-data example table, loaded by `bench.datasets.example_table()`. It lives
in the repository at `bench/data/stellar_chemokinematics_apogee_dr19.npz`
(arrays `X` float32 of shape 9242 x 8, `source_id` int64, `feature_names`)
and is deliberately not part of the installed package: the library holds only
the procedure, and this table would be 85 percent of an otherwise 62 kB
wheel. The loader verifies the file digest, row count and column names before
returning. SHA-256 of the npz:
`5bd5c78bd1eadaf5d38801fda962c8433922d67bc18aa20a44ea6c47023d411a`.

On 2026-09-05 the original zstd parquet was converted to this npz with an
exact round-trip check: every value and every `source_id` is identical
(SHA-256 of the concatenated `X` and `source_id` bytes:
`13521a2eb590ce993c7be37927fa6a7c0822fc26dd7eccc4d4ce9fa5664fb8fe`). The
parquet was removed so that the package does not depend on pyarrow. The
build description below refers to the parquet that was converted.

## What it is

9 242 Milky Way stars with eight chemo-dynamical features (real ASPCAP
abundances, `galpy`-integrated orbits) and a `source_id` for
cross-matching. Columns:

| column | dtype | derivation |
|---|---|---|
| `source_id` | int64 | Gaia DR3 source ID |
| `fe_h` | float32 | ASPCAP `FE_H_ATM`, dex; true `[Fe/H]` |
| `alpha_m` | float32 | ASPCAP `ALPHA_M_ATM`, dex; `[alpha/M]` |
| `mg_fe` | float32 | ASPCAP `MG_H_ATM` − `FE_H_ATM`, dex; true `[Mg/Fe]` |
| `J_R` | float32 | radial action from `galpy`, kpc km/s |
| `J_z` | float32 | vertical action from `galpy`, kpc km/s |
| `L_z` | float32 | azimuthal action from `galpy`, kpc km/s |
| `ecc` | float32 | orbital eccentricity from `galpy` |
| `E` | float32 | orbital energy from `galpy`, (km/s)^2 |

File size: 0.34 MB (compressed npz); the converted parquet was 0.47 MB.

## Scientific origin

**Chemistry** comes directly from the APOGEE ASPCAP atmospheric
abundance columns in SDSS-V Data Release 19
([Meszaros et al. 2025, AJ 170, 96](https://doi.org/10.3847/1538-3881/ade4b9);
release described in [SDSS Collaboration 2025](https://arxiv.org/abs/2507.07093);
the ASPCAP method itself is documented for APOGEE DR17 by
[Abdurro'uf et al. 2022, ApJS 259, 35](https://doi.org/10.3847/1538-4365/ac4414)).
No neural-network prediction layer is involved. The bracket `[Mg/Fe] = [Mg/H] − [Fe/H]` is derived
arithmetically from the two ASPCAP `_atm` columns.

**Orbital quantities** come from `galpy` ([Bovy 2015, ApJS 216, 29](https://doi.org/10.1088/0067-0049/216/2/29))
orbit integrations in the `McMillan17` Milky Way potential
([McMillan 2017, MNRAS 465, 76](https://doi.org/10.1093/mnras/stw2759))
using Gaia DR3 astrometry
([Gaia Collaboration et al. 2023, A&A 674, A1](https://doi.org/10.1051/0004-6361/202243940)).

The upstream pipeline belongs to the `ArqueoGal` project. Applied gates
at build time:

- `flag_bad == True` in APOGEE ASPCAP -> drop
- `snr <= 70` -> drop (spectroscopic SNR cut)
- missing `FE_H_ATM`, `MG_H_ATM`, or `ALPHA_M_ATM` -> drop
- missing `galpy` kinematics (integration failure or missing PM/RV) -> drop
- any NaN in the eight derived features -> drop
- duplicate `source_id` -> drop (keep first)

This yields 9 242 stars that have *both* ASPCAP chemistry and `galpy`
kinematics, which is the intersection of a spectroscopic survey
(APOGEE DR19, ~350 000 sources) with a volume-limited kinematic cut
(~249 000 sources).

## Why it is bundled

Downloading APOGEE DR19 (~1 GB) and re-running orbit integration takes
far longer than the bounded diagnostic. Bundling a reproducible subset keeps
that real-data check repeatable and offline.

This bundle is a convenience. Do not use it for scientific work:
rebuild from APOGEE and Gaia for anything that needs to be citable.

## Earlier version (removed 2026-04-21)

A previous `stellar_chemokinematics_100k.parquet` (100 000 rows)
sourced its chemistry from the ArqueoGal Pipeline-1 *predicted*
abundances (`mh_pred`, `mg_h_pred`, `alpha_m_pred`). That file has been
removed because:

- It labelled `mh_pred` (which is `[M/H]`) as `fe_h`, conflating
  overall metallicity with iron abundance.
- It labelled `mg_h_pred − mh_pred` (which is `[Mg/M]`) as `mg_fe`
  instead of computing the true `[Mg/Fe] = [Mg/H] − [Fe/H]`.
- The predictions collapsed at low metallicity: for stars with
  `[Fe/H] < -0.8` the predicted `[alpha/M]` had an inter-quartile
  spread of only 0.004 dex, far below astroNN's per-star precision.
  This created an artificial tight cluster at low predicted `[Fe/H]`
  with unphysically low predicted `[alpha/M]`.

Switching to direct ASPCAP columns eliminates both problems.

## How it was built

```python
import hashlib
import numpy as np
import pandas as pd

APOGEE = ".../ArqueoGal/data/interim/apogee_dr19_corrected.parquet"
KIN    = ".../ArqueoGal/data/processed/pipeline2_kinematics_stream3_volume.parquet"

ap = pd.read_parquet(APOGEE, columns=[
    "source_id", "fe_h_atm", "mg_h_atm", "alpha_m_atm",
    "flag_bad", "snr",
])
ap = ap[
    (~ap["flag_bad"].fillna(True).astype(bool)) &
    ap[["fe_h_atm", "mg_h_atm", "alpha_m_atm"]].notna().all(axis=1) &
    (ap["snr"] > 70)
].copy()
ap["fe_h"] = ap["fe_h_atm"].astype(np.float32)
ap["mg_fe"] = (ap["mg_h_atm"] - ap["fe_h_atm"]).astype(np.float32)
ap["alpha_m"] = ap["alpha_m_atm"].astype(np.float32)

kin = pd.read_parquet(KIN, columns=[
    "source_id", "J_R_kpc_kms", "J_z_kpc_kms", "L_z_kpc_kms", "ecc", "E_kms2",
]).rename(columns={
    "J_R_kpc_kms": "J_R", "J_z_kpc_kms": "J_z",
    "L_z_kpc_kms": "L_z", "E_kms2": "E",
})
for c in ("J_R", "J_z", "L_z", "ecc", "E"):
    kin[c] = kin[c].astype(np.float32)

FEATURES = ["fe_h", "alpha_m", "mg_fe", "J_R", "J_z", "L_z", "ecc", "E"]
merged = (
    ap[["source_id", "fe_h", "alpha_m", "mg_fe"]]
    .merge(kin, on="source_id", how="inner", validate="many_to_one")
    .dropna(subset=FEATURES)
    .drop_duplicates(subset=["source_id"])
    .sort_values("source_id")
    .reset_index(drop=True)
)
merged["source_id"] = merged["source_id"].astype(np.int64)
merged[["source_id", *FEATURES]].to_parquet(
    "docs/data/stellar_chemokinematics_apogee_dr19.parquet",
    index=False, compression="zstd", compression_level=9,
)
```

Stable `source_id` sort order. Rebuilding against the same upstream
parquet produces a byte-identical file.

## Integrity

```
sha256  38beae1d9aaf0a131a826a90e4bfa151d40cfdb215596c8c4107d2072c8a8b73
```

## Licensing

APOGEE and Gaia data products are public. Redistribution of a small
reproducible subset for tutorial purposes is permitted. If you publish
results using this tool on real data, cite the APOGEE, Gaia, and
`galpy` references above.

## Summary statistics (of the bundled 9 242 rows)

| feature | mean | std | min | median | max |
|---|---:|---:|---:|---:|---:|
| `fe_h` | -0.154 | 0.244 | -1.75 | -0.138 | +0.44 |
| `alpha_m` | +0.072 | 0.086 | -0.10 | +0.045 | +0.39 |
| `mg_fe` | +0.094 | 0.104 | -0.28 | +0.062 | +0.42 |
| `J_R` | 50.7 | 74.2 | ~0 | 29.4 | 1750 |
| `J_z` | 19.4 | 26.6 | ~0 | 11.3 | 466 |
| `L_z` | 1811 | 365 | -914 | 1857 | 2831 |
| `ecc` | 0.171 | 0.117 | ~0 | 0.148 | 1.00 |
| `E` | -156 971 | 8 748 | -199 415 | -155 907 | -123 824 |
