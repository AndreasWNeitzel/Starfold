# Run Starfold on a GPU

This guide installs {term}`cuML` (NVIDIA's GPU versions of UMAP and HDBSCAN), selects it with the `engine` option, and shows what happens when no GPU is usable.

```{eval-rst}
.. index::
   single: GPU
   single: cuML
   single: engine
   pair: engine; auto
   single: WSL 2
   single: CPU fallback
```

## Prerequisites

- A Linux machine with an NVIDIA GPU, or Windows with WSL 2 (Windows Subsystem for Linux, a Linux environment inside Windows). cuML does not run on native Windows. RAPIDS is NVIDIA's suite of GPU data libraries, and {term}`cuML` is its machine-learning part. CUDA is NVIDIA's GPU computing platform.
- Python 3.11 or 3.12.
- The output of `nvidia-smi` (a command that comes with the NVIDIA driver), which shows the highest CUDA version your driver supports.

## Steps

1. Pick the GPU extra that matches the CUDA version of `nvidia-smi`, and install one of them. The RAPIDS install selector lists the driver each CUDA version needs ([RAPIDS install](https://docs.nvidia.com/datascience/install/)). As of 2026-09-30 that page lists driver 525.60.13 or newer for CUDA 12 and 580.65.06 or newer for CUDA 13.

   ```console
   pip install "starfold[gpu-cu13] @ git+https://github.com/AndreasWNeitzel/starfold"
   pip install "starfold[gpu-cu12] @ git+https://github.com/AndreasWNeitzel/starfold"
   ```

   An extra is an optional group of dependencies that the install pulls in. Run only one of the two lines. They pin different CUDA toolkits, and `uv` (a Python package manager, see {doc}`../start/install`) refuses to install both. From a clone, use `uv sync --extra bench --extra gpu-cu13` (or `gpu-cu12`).

2. Ask for cuML and run.

   ```python
   import starfold

   result = starfold.recover(X, engine="cuml", random_state=0)
   ```

   If you leave out `engine`, the default `"auto"` tries cuML first and falls back to the CPU with a warning. With `engine="cuml"` the call raises `ImportError` before any stage runs if cuML cannot be imported or sees no CUDA device. The message holds the reason.

3. Confirm which backend ran.

   ```python
   prov = result.provenance
   print(f"engine: asked for {prov['engine_requested']}, used {prov['engine_resolved']}")
   ```

## Expected result

The recorded Digits run, started with the default `engine="auto"` on a laptop RTX 3060 under WSL 2, prints:

You see:

```text
engine: asked for auto, used cuml
```

Under cuML, UMAP, HDBSCAN and the scoring run on the GPU. Scaling, Optuna's sampling, the structureless copies and the final regrouping of the {term}`consensus check` run on the CPU. Peak GPU memory is not recorded, so it is not measured here. The recorded peak host memory of the Digits run is 1,428 MB.

## What "auto" does

`engine="auto"` is the default. It uses cuML when cuML imports and a CUDA device is visible. Otherwise it issues a `RuntimeWarning` that names the reason and runs on the CPU. `engine="cpu"` never touches the GPU, and `engine="cuml"` fails instead of falling back. The check runs once per Python process, because a failing cuML import is slow and its outcome cannot change within a run.

The CPU path is the reference. The two backends give different labels and group counts for the same settings, because their floating-point order differs, so compare them by partition with {doc}`compare-backends` and never by label number.

## Variations

- **Force the CPU.** Pass `engine="cpu"`. The digits notebook expects a CPU run to take several times longer than the GPU run. No CPU time is recorded, so it is not measured here.
- **GPU out of memory.** Lower `consensus_budget_bytes` and `consensus_check_rows` in `Config`. The check needs `16 * consensus_check_rows**2` bytes, which must not exceed `consensus_budget_bytes`. Or use `engine="cpu"`.
- **Parallel jobs.** On cuML, `n_jobs` is forced to 1 ({doc}`run-parallel`).
- **Check the agreement of the backends.** Set `Config(check_backends=True)`, which needs cuML.

## Related pages

- {doc}`run-parallel`
- {doc}`compare-backends`
- {doc}`../start/install`
- {doc}`../reference/api`
- [cuML documentation](https://docs.nvidia.com/cuml/latest/)
