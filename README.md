# Clinical sample enrichment

Graph autoencoder experiments connecting clinical protein expression data to
mouse Reactome pathways. Includes a prepared graph, training code, notebooks,
and historical model results.

## Setup

CPU and GPU training share the **`clinical-gpu` Conda environment**: Linux x86_64,
Python 3.8.20, PyTorch 2.2.2+cu121, and PyTorch Geometric 2.5.2.
From the repository root, with [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) installed:

```bash
conda env create -f environment-gpu.yml
conda activate clinical-gpu
python -m pip install -r requirements-gpu.txt
```

Create the environment once; use `conda activate clinical-gpu` for later sessions.
The CUDA build runs on CPU without an NVIDIA GPU. GPU execution requires an
NVIDIA GPU and a driver compatible with CUDA 12.1; no separate toolkit is needed.
The original `environment.yml` is a Windows export; use `environment-gpu.yml`.

## Training

```bash
# CPU
MPLBACKEND=Agg python train.py --device cpu --epochs 2 --max-nodes 200 --output-dir outputs/cpu

# GPU
MPLBACKEND=Agg python train.py --device cuda:0 --epochs 2 --max-nodes 1000 --output-dir outputs/gpu

# Sampling (also supports forest_fire)
MPLBACKEND=Agg python train.py --device cpu --sampling random_walk --sample-nodes 100 --max-nodes 1000
```

`--device auto` selects an available GPU or falls back to CPU. `cuda:1` selects
another visible GPU. Use `--max-nodes 0` for the full graph (50,164 nodes,
1,667,138 edges), which needs more memory and time. Random-walk samples must fit
within the selected graph's largest connected component. See
`python train.py --help` for architecture, seed, and path options.

For **ClusterGCN**, install the matching extension (requires `data.pyg.org`),
then train with `--sampling clusterGCN` on either CPU or GPU:

```bash
python -m pip install --no-deps --only-binary=:all: pyg-lib==0.4.0 \
  --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

Runs are seeded and deterministic by default. If a CUDA operation has no
deterministic implementation, `--allow-nondeterministic` permits it at the cost
of repeatability. Results can differ across hardware and library versions;
new runs do not reproduce historical checkpoints bit for bit.

## Outputs

Each run saves embeddings, a learning curve, `checkpoint.pt`, and `run.json`
with parameters, losses, input checksum, feature schema, versions, and device
information. Checkpoint tensors are stored on CPU. Use a separate
`--output-dir` for each experiment to avoid replacing artifacts.

```python
import torch
from torch_geometric.nn import GAE
from train import GCNEncoder

checkpoint = torch.load("outputs/<run>/checkpoint.pt", map_location="cpu", weights_only=True)
keys = ("in_channels", "hidden_channels", "out_channels", "num_layers", "dropout_rate")
model = GAE(GCNEncoder(*(checkpoint[key] for key in keys)))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Use the saved `feature_keys` and `in_channels` when preparing inference data
with `GAEPipeline`. Load only trusted pickle graphs and legacy `.pth` models.

## Data and notebooks

`data/` contains the graph, expression measurements, and protein/pathway
mappings. `models/` contains historical results. `data_analysis.ipynb` explores
preprocessing; `Evaluation.ipynb` explores clustering and enrichment.

```bash
python -m jupyterlab --no-browser --ip=127.0.0.1
```

Historical notebooks may need path adjustments and generated files. Graph
reconstruction requires a populated Neo4j Reactome database; live enrichment
uses [g:Profiler](https://biit.cs.ut.ee/gprofiler/). Bundled-data training works
offline after installation. Full notebook execution and historical training
sweeps have not been validated.

## Validation

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

Checks cover seeded training, sampling, checkpoints, device selection, and
analysis helpers. CPU execution is validated. GPU hardware execution remains
unverified; its test is skipped without a GPU. ClusterGCN tests require the
optional extension.
