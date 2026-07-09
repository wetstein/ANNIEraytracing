# Linux Installation

This project uses [Taichi](https://taichi-lang.org/) for ray tracing.  Taichi
auto-detects CUDA (NVIDIA GPU), Vulkan, or Metal backends, and falls back to
CPU if none are found — no code changes needed either way.  All launch code
defaults to `ti.cpu`; with a GPU you can switch to `ti.cuda` for roughly
5-10× faster batch simulations (see step 6).

The interactive 3D viewer needs a browser for WebGL rendering but no GPU on
the server for compute.

## Prerequisites

- **Python ≥ 3.11** — check with `python3 --version`.  If older, install
  via your package manager (e.g. `apt install python3.11 python3.11-venv`
  on Debian/Ubuntu) or from [python.org](https://python.org).
- **git** — `apt install git` / `dnf install git`
- **C++ toolchain** (for Taichi's JIT compiler):
  - Debian/Ubuntu: `apt install build-essential cmake`
  - Fedora: `dnf install gcc-c++ cmake`
  - Arch: `pacman -S base-devel cmake`
- **NVIDIA GPU owners only — CUDA toolkit:**
  - Debian/Ubuntu: `sudo apt install nvidia-cuda-toolkit`
  - Fedora: `sudo dnf install cuda-toolkit`
  - Arch: `sudo pacman -S cuda`
  - Or download from <https://developer.nvidia.com/cuda-downloads>
  - Verify with: `nvcc --version`

## 1. Clone and initialise submodules

```bash
git clone https://github.com/wetstein/ANNIEraytracing.git
cd ANNIEraytracing
git submodule update --init
```

## 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Upgrade pip: `pip install --upgrade pip`

## 3. Install system packages (required by lxml)

Debian / Ubuntu:

```bash
sudo apt install libxml2-dev libxslt1-dev
```

Fedora:

```bash
sudo dnf install libxml2-devel libxslt-devel
```

These are only needed if you don't already have them — lxml's pip wheel may
already bundle them on your platform.

## 4. Install Python dependencies

Taichi works on both GPU and CPU from the same pip package.  No separate
CUDA-wheel is needed — the `taichi` pip install is universal.

**x86_64 (Intel/AMD) — simple:**

```bash
pip install -e .
```

This installs everything from `pyproject.toml`, including cadquery, which
distributes a manylinux wheel on x86_64.

**aarch64 (ARM, e.g. Raspberry Pi 4/5, Ampere) — cadquery needs conda:**

CadQuery does not publish a pip wheel for Linux ARM.  Use conda:

```bash
# Install miniforge (no NVIDIA GPU needed → no CUDA)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
# (follow the prompts, then restart your shell)

# Create environment with cadquery from conda-forge
conda create -n annie python=3.12
conda activate annie
conda install -c conda-forge cadquery h5py lxml pyyaml numpy
pip install taichi
```

Then install the package itself (still inside the conda env):

```bash
cd ANNIEraytracing
pip install -e . --no-deps
```

(`--no-deps` avoids re-resolving dependencies already satisfied by conda.)

**Alternative for ARM: skip cadquery entirely.** Most batch workflows use
`--pmt-csv` and never need cadquery.  Just `pip install -e .` will fail on
cadquery, so install the other deps by hand:

```bash
pip install taichi numpy h5py lxml pyyaml
pip install -e . --no-deps
```

## 5. Verify the installation

```bash
python -c "import annieray; print('OK')"
# Optional — check Taichi backend:
python -c "import taichi as ti; ti.init(); print(ti.lang.impl.current_cfg().arch)"
```

You should see `OK` with no errors.  The Taichi arch check will print
`x64` (CPU), `cuda` (NVIDIA GPU), or similar.  If cadquery is missing,
the package will still import — cadquery is only used when `--step` or
`--manifest` are passed on the command line.

## 6. Run a test simulation

On CPU (any machine):

```bash
python -m annieray batch --events 10 --photons-per-cm 50
```

Expected output:

```
Wrote results/photon_hits.parquet (NNNN photon rows)
Wrote results/muon_truth.parquet (10 muon truth rows)
Done.
```

**With an NVIDIA GPU — switch to CUDA for 5-10× speedup:**

```bash
python -c "
import taichi as ti
ti.init(arch=ti.cuda)
from annieray.cli import main
main()
" batch --events 500 --photons-per-cm 150
```

Or set the arch directly in a one-off script.  For routine GPU use you can
edit the `ti.init(arch=ti.cpu)` calls in `annieray/cli.py` and
`annieray/viz_server.py` to `ti.cuda`.

### With PMT CSV and surfboard LAPPDs

```bash
# Get the PMT positions file (ask the collaboration)
# Then:
python -m annieray batch \
    --pmt-csv PMTPositions_Scan.txt \
    --surfboard 3 --lappd-model annie \
    --events 100 --photons-per-cm 150
```

## 7. Interactive visualiser

The 3D viewer runs as a local web server — it serves a Three.js page to
your browser.  The Taichi ray tracing runs on the server (CPU or GPU); the
browser only does WebGL rendering of the scene.  No special server GPU is
needed for the viewer.

```bash
python -m annieray viz-server \
    --pmt-csv PMTPositions_Scan.txt \
    --port 8080
```

Open `http://localhost:8080` in any browser on the same machine (or another
machine on the same network).  Chrome, Firefox, and Edge all support
WebGL for the Three.js rendering.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `taichi.lang.exception.TaichiCompilationError: ...` | Taichi JIT needs C++ tools | `apt install build-essential cmake` |
| `ModuleNotFoundError: No module named 'cadquery'` | ARM Linux + no conda | Install via conda-forge (section 4) or use `--pmt-csv` workflow (doesn't need cadquery) |
| `Cannot open self /usr/bin/python3` | System Python too old | Install Python 3.11+ from deadsnakes PPA or python.org |
| `import taichi` crashes or `ti.cuda` unavailable | CUDA not installed or incompatible | Run with CPU only (default).  Taichi prints a clear error if it can't find a requested backend. |
| Performance warning from Taichi, arch = `x64` | No GPU detected | Normal — CPU mode is ~2-5× slower than GPU.  If you have an NVIDIA card, install CUDA toolkit. |
 | `h5py failed to open file` | File corrupt from killed job | Delete and re-run with smaller `--events` or longer timeout |
