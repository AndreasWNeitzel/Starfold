# Install Starfold

This page installs Starfold on the CPU, then adds GPU support if you have a supported NVIDIA card.

```{eval-rst}
.. index::
   single: install
   single: cuML
   single: engine
   pair: install; GPU
   single: Python version
```

## What you need

- Python 3.11 or 3.12. Starfold declares `requires-python >=3.11,<3.13`, so pip refuses other versions.
- A terminal with `pip`.
- For the GPU only: Linux or WSL 2 (Windows Subsystem for Linux), an NVIDIA GPU and a driver new enough for one CUDA major version. CUDA is NVIDIA's toolkit for running code on its GPUs (see below). You do not need a GPU for the first run: the CPU path gives the same kind of result and takes longer.

## Install on the CPU

1. Create and activate a virtual environment (a private folder of packages for one project), so that the install leaves your system Python alone:

   ```console
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows outside WSL, run `.venv\Scripts\activate` instead of the second line.

2. Install Starfold:

   ```console
   pip install "starfold @ git+https://github.com/AndreasWNeitzel/starfold"
   ```

   This installs Starfold with {term}`UMAP` (`umap-learn`), {term}`HDBSCAN` (`hdbscan`) and {term}`Optuna`. The CPU path is the reference path.

3. Check the install:

   ```console
   python -c "import starfold; print(starfold.__version__)"
   ```

   It prints the version:

   ```text
   0.2.0
   ```

## Add the GPU

The GPU extras install RAPIDS {term}`cuML`, NVIDIA's GPU versions of UMAP and HDBSCAN. cuML runs on Linux and WSL 2 only. An extra is an optional bundle of dependencies that pip installs on request. Choose the extra that matches your CUDA major version, and install one, never both, because each one installs a different CUDA toolkit.

| Your setup | Command |
|---|---|
| CUDA 13 | `pip install "starfold[gpu-cu13] @ git+https://github.com/AndreasWNeitzel/starfold"` |
| CUDA 12 | `pip install "starfold[gpu-cu12] @ git+https://github.com/AndreasWNeitzel/starfold"` |

Type `nvidia-smi` in a terminal. The header of its table shows the highest CUDA version your driver supports. The RAPIDS install page lists the minimum driver for each major version: 525.60.13 for CUDA 12 and 580.65.06 for CUDA 13 (see the [RAPIDS install page](https://docs.nvidia.com/datascience/install/), which moved from docs.rapids.ai). Both extras require `cuml>=25.12`.

The {term}`engine` argument of `starfold.recover` decides what runs. `engine="auto"` uses cuML when it imports and sees a GPU, and otherwise warns and runs on the CPU. `engine="cpu"` never touches the GPU. `engine="cuml"` raises an error when cuML is not usable.

## Clone for notebooks and the bundled data

The notebooks and the bundled APOGEE table are in the repository, not in the pip package. The commands below use `uv`, a Python package manager. Install it from its website ([uv documentation](https://docs.astral.sh/uv/)), or use pip as shown above and skip this section. Then clone the repository:

```console
git clone https://github.com/AndreasWNeitzel/starfold
cd starfold
uv sync --extra bench
```

`uv sync` creates the virtual environment for you. Add `--extra gpu-cu13` or `--extra gpu-cu12` to the last command for cuML. The `bench` extra adds pandas, matplotlib and ipykernel, which the notebooks need.

## Common install errors

**pip reports no matching distribution, or a Python version that does not satisfy the requirement.**
Your Python is older than 3.11 or newer than 3.12. Create an environment with Python 3.11 or 3.12 and install again.

**The installer reports conflicting requirements when you ask for both GPU extras.**
`gpu-cu12` and `gpu-cu13` pin mutually exclusive CUDA toolkits, and `pyproject.toml` declares them as conflicting for `uv`. Install one.

**`ImportError: engine='cuml' requires an operational RAPIDS/cuML CUDA backend.`**
You asked for `engine="cuml"` and cuML cannot import or find a device. The message carries the reason. Install a GPU extra on Linux or WSL 2, update the driver, or use `engine="auto"` or `engine="cpu"`.

## Next

Run it: {doc}`first-run`. Platform and method limits are on the {doc}`../limits` page.
