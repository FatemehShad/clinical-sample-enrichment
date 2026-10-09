# Clinical sample enrichment

Graph autoencoder experiments linking clinical protein expression data to mouse
Reactome pathways. The repository includes the expression table, a prepared
NetworkX graph, training code, analysis notebooks, and historical model results.

## CPU environment

The tested platform is **Linux x86_64, CPython 3.8.20, PyTorch 2.2.2 CPU, and
PyTorch Geometric 2.5.2**. Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
then run from the repository root:

```bash
uv python install 3.8.20
bash scripts/setup.sh
source .venv/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2
python -m unittest discover -s tests -v
```

`requirements.txt` and `requirements-gpu.txt` select CPU and CUDA builds;
`requirements-common.txt` pins their shared direct and transitive dependencies.
The setup script synchronizes that environment without changing the lockfile.
`environment.yml` is the original Windows Conda export; its Windows build pins
and mismatched torchvision version make it unsuitable as a Linux setup recipe.
GPU hardware training and newer Python versions have not been validated in the CPU cloud environment.

For **ClusterGCN**, add the matching CPU METIS-enabled extension:

```bash
bash scripts/setup.sh --cluster
```

This requires HTTPS access to `data.pyg.org`. Regular graph convolution, random
walk sampling, and forest fire sampling do not require this extension. When the wheel host is unavailable, a tested Linux source fallback uses a C++
compiler, the locked CMake/Ninja tools, and the pinned official pyg-lib commit
with its pinned METIS submodules:

```bash
bash scripts/setup.sh --cluster-source
```

The source build uses two compilation jobs and can take several minutes.
`CLINICAL_VENV` selects another environment path; `CLINICAL_PYG_SOURCE` selects
the dependency source cache. A generic
source build of `torch-sparse` without METIS is insufficient for ClusterGCN.

## NVIDIA GPU setup and training

A CUDA-capable NVIDIA GPU and a driver compatible with CUDA 12.1 are required.
The GPU environment pins **PyTorch 2.2.2+cu121** and its CUDA runtime dependencies
for the same Linux x86_64 / Python 3.8.20 platform. The PyTorch wheel includes
its published SHA-256. `environment-gpu.yml` pins the Conda bootstrap;
`requirements-gpu.txt` pins the Python package installation. GPU training does not require a separate CUDA toolkit.

Install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
and create the dedicated GPU environment from the repository root. Conda
provides the pinned Python and pip versions; pip installs the same pinned
CUDA-enabled PyTorch and scientific dependencies:

```bash
conda env create -f environment-gpu.yml
conda activate clinical-gpu
python -m pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
MPLBACKEND=Agg python train.py --device cuda:0 --epochs 2 --max-nodes 1000 --output-dir outputs/gpu
```

If `clinical-gpu` already exists, use `conda activate clinical-gpu` and rerun
the pip installation instead of creating it again. To leave it, run
`conda deactivate`. The legacy `environment.yml` remains the original Windows
export; use `environment-gpu.yml` for this Linux GPU workflow.

`--device cuda` selects the current GPU, `--device cuda:1` selects another visible
GPU, and `--device auto` selects a GPU when available and otherwise uses CPU.
The CLI defaults to CPU. An explicit CUDA request fails before loading data or
creating outputs when the CUDA build, GPU, or driver is unavailable. GPU indices
refer to the devices visible through `CUDA_VISIBLE_DEVICES`.

All training modes move features, edges, and model parameters to the chosen
device. Graph loading and sampling remain on CPU. For GPU **ClusterGCN**, install
the matching PyG extension; the CPU source fallback is for CPU environments:

```bash
conda activate clinical-gpu
python -m pip install --no-deps --only-binary=:all: pyg-lib==0.4.0 \
  --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html
MPLBACKEND=Agg python train.py --device cuda:0 --sampling clusterGCN --max-nodes 1000 --epochs 2 --output-dir outputs/gpu-cluster
```

Deterministic operations are enabled by default. The code configures cuBLAS
before CUDA initialization and disables TF32. Restart an existing notebook
kernel before enabling this mode if it has already initialized CUDA. If a CUDA
operation has no deterministic implementation, PyTorch reports it; explicitly
use `--allow-nondeterministic` (or `GAEPipeline(..., deterministic=False)`) to
allow that operation, accepting that repeated runs may differ. Results are not
guaranteed identical across different GPUs, drivers, or CPU/GPU backends.

`run.json` records the requested and resolved device, GPU name, CUDA version,
determinism setting, and cuBLAS configuration. Portable state checkpoints store
CPU tensors even when training on GPU, so loading them does not require a GPU.
The GPU dependency installation and CPU fallback checks can be validated on a
CPU host; actual GPU execution tests are skipped until run on NVIDIA hardware:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The GPU test exercises unsampled, random-walk, and forest-fire training twice,
compares losses and weights, and verifies the saved device and portable weights.
It also checks ClusterGCN when the optional matching extension is installed.

## Run training

A small, seeded run uses the bundled graph and needs no database or API:

```bash
python train.py --epochs 2 --max-nodes 200 --seed 42 --device cpu
```

The default selects the first 200 nodes in the stored graph order, takes their
induced subgraph, and trains without sampling. For reproducible sampled runs:

```bash
python train.py --sampling random_walk --sample-nodes 100 --max-nodes 1000 --epochs 2
python train.py --sampling forest_fire --sample-nodes 100 --max-nodes 1000 --epochs 2
python train.py --sampling clusterGCN --max-nodes 1000 --epochs 2
```

Random walk uses the largest connected component of the selected subgraph so
that its requested sample is reachable. `--sample-nodes` must fit within it.
For a research run, explicitly increase epochs and use `--max-nodes 0` for all
50,164 nodes and 1,667,138 edges. Full-graph training has substantially higher
memory and time costs; the quick run does not reproduce historical paper results.
Use `python train.py --help` for architecture, input, and output options.

Input defaults are resolved relative to `train.py`, so the command also works
from another working directory. Importing `train` does not load data, create
output directories, or run training. Notebook imports remain available:

```python
from train import GAEPipeline, GCNEncoder
pipeline = GAEPipeline(15, 64, 60, 6, 0.2, seed=42, device="cpu")
graph = pipeline.load_graph_from_pickle()  # data/combined_graph_latest.pkl
```

Python, NumPy, PyTorch, dropout, and sampling are seeded. Deterministic PyTorch
operations are enabled. The regression suite compares repeated training losses
and model tensors exactly on CPU. Results across platforms, GPUs, or dependency
versions are not guaranteed to be identical. New CLI runs use the sorted union
of the input graph's attribute names as a stable feature schema, fill missing or
nonnumeric attributes with zero, and normalize numeric columns. Undirected edges
are represented in both directions. These corrections mean new runs are not
bitwise reproductions of the historical checkpoints. Pass the saved
`feature_keys` and `in_channels` to `GAEPipeline` when preparing inference data.

## Outputs and checkpoints

New runs write under `outputs/`, leaving `models/` and the input data untouched.
Specify `--output-dir /path/to/run` to choose another location. Reusing the same
configuration and output directory replaces that run's artifacts.

Each CLI run records `run.json` with the options, losses, software versions,
input graph SHA-256, graph size, ordered feature keys, and device details; saves a learning curve and embeddings; and
writes `checkpoint.pt` containing the model state and architecture parameters.
Saved embeddings use the final model in evaluation mode. ClusterGCN embeddings
are restored to original graph node order before saving.

Load the portable checkpoint with the matching environment:

```python
import torch
from torch_geometric.nn import GAE
from train import GCNEncoder

checkpoint = torch.load("outputs/<run>/checkpoint.pt", map_location="cpu", weights_only=True)
encoder = GCNEncoder(*(checkpoint[key] for key in
    ("in_channels", "hidden_channels", "out_channels", "num_layers", "dropout_rate")))
model = GAE(encoder)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Legacy full-model `.pth` files are also saved for existing analysis helpers.
Those pickle-based files can depend on the module/class names used when they
were created; use the state checkpoint for new experiments. Load only trusted
pickle graphs and model files.

## Data and notebooks

| Path | Purpose |
| --- | --- |
| `data/combined_graph_latest.pkl` | Prepared graph for offline training |
| `data/expression_data.xlsx` | Clinical expression measurements |
| `data/train.csv` | Classification input |
| `data/MMU_Uniprot2Reactome.txt` | Mouse protein/pathway mapping |
| `data/MMU_ReactomePathwaysRelation.txt` | Reactome pathway hierarchy |
| `data_analysis.ipynb` | Exploratory preprocessing and graph construction |
| `Evaluation.ipynb` | Historical embedding/clustering/enrichment analysis |
| `utils_functions.py` | Mapping, clustering, evaluation, and plotting helpers |
| `models/Cluster-GAE/` | Historical checkpoint and analysis outputs |
| `extra/` | Additional exploratory notebooks; outside the validated workflow |

Start notebooks from the repository root:

```bash
unset MPLBACKEND  # allow notebook inline figures
python -m jupyterlab --no-browser --ip=127.0.0.1
```

The historical notebooks are exploratory records, not a clean end-to-end
pipeline. Some cells refer to generated files such as `gprofiler_results.csv`,
root-level graph filenames, or older model directories that are not bundled.
Point those cells at `data/` and newly generated artifacts as needed, and avoid
rerunning cells that write over historical outputs.

Graph construction cells require a populated **Neo4j Reactome database**, not
just an empty Neo4j server. The bundled prepared graph avoids that requirement
for training. Configure database access locally rather than reusing the example
credentials in historical notebook cells. Live enrichment calls require access
to `https://biit.cs.ut.ee/gprofiler/`; their results can change with the service's
reference database. No API key is required by the existing g:Profiler client.
Full notebook execution, live enrichment, and the historical training sweeps
are outside the offline regression checks.

## Checks

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

The suite checks import side effects, repeated seeded sampling and training,
bundled-data CLI execution, and analysis helpers. It writes temporary artifacts
outside the checkout and fails on regressions; no existing automated test suite
was supplied with the original repository.
