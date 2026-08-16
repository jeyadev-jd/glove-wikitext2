# GloVe from Scratch — WikiText-2

A complete, reproducible implementation of **GloVe (Global Vectors for Word
Representation)** written from first principles and trained on WikiText-2.

The co-occurrence construction, the weighted least-squares objective, the
gradients and the AdaGrad optimiser are all implemented explicitly. PyTorch is
used **only** as a vectorised tensor/CUDA library — no autograd, no
`torch.optim`, no gensim, no pretrained vectors in training. Official
`glove.6B.100d` vectors are downloaded *after* training, for comparison only.

---

## Measured results

Baseline run: 100D, window 1, `min_freq` 5, 30 epochs, batch 4096, seed 42, on an
NVIDIA RTX 3050 Laptop GPU (4 GB). All figures below come from an actual run and
are reproduced in `results/`.

| Metric | Value |
|---|---|
| Corpus tokens (after tokenisation) | 1,760,383 |
| Unique raw types → vocabulary | 63,985 → 20,000 |
| Tokens retained | 95.66% |
| Non-zero co-occurrence entries | 1,109,952 of 400,000,000 (**99.72% sparse**) |
| Dense vs sparse memory | 1.490 GB → 12.70 MB (**120× smaller**) |
| Final training loss | 0.011541 |
| Training time (30 epochs) | 24.24 s |
| Peak GPU memory | 72.5 MB |
| Analogy accuracy (16 quadruples, 0 OOV) | **18.75%** |
| Official `glove.6B.100d` on the same set | 81.25% |
| GPU speedup vs CPU (3 epochs, identical config) | **4.58×** |

Sample nearest neighbours from the trained vectors:

```
      king: james, henry, john, wood, constantine, othniel, edward, captain, william
  computer: science, fiction, stories, raffles, household, recordings, police
      city: bradford, kansas, norwich, york, town, cass, council, largest, birmingham
     music: videos, video, sheet, awards, folk, sony, grammy, boxing, selling
```

The gap to official GloVe is expected and discussed in the report: 6B tokens vs
1.76M, a 400k vocabulary vs 20k, and a window of 10 vs 1.

---

## Features

- Programmatic WikiText-2 download (train/validation/test), corpus never vendored
- From-scratch preprocessing: lowercasing, regex tokenisation, rare-token
  removal (`min_frequency`), vocabulary cap (`max_vocab_size`)
- **Sparse** distance-weighted symmetric co-occurrence (`X[i,j] += 1/distance`) —
  the dense `V×V` matrix is never allocated
- GloVe objective, weighting function `f(x)`, and **hand-derived gradients**
  (verified against autograd in the test suite)
- **Manual AdaGrad** with per-parameter squared-gradient accumulators
- Mini-batch training with automatic batch-size fallback on CUDA OOM
- CUDA acceleration with CPU fallback; fits well inside 4 GB VRAM
- Checkpoint / resume
- Nearest neighbours + analogy evaluation with from-scratch cosine similarity
- OOV analysis and side-by-side comparison against official `glove.6B.100d`
- One-parameter-at-a-time ablations, CPU vs GPU benchmark, sparsity/memory analysis
- Loss curve, PCA visualisation, result CSVs, and an auto-generated PDF report

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+. `torch` may be installed with the CUDA build matching your driver
(see <https://pytorch.org/get-started/locally/>); the CPU wheel also works.

---

## Running locally

```bash
python run.py                       # baseline + evaluation + ablations + report
python run.py --epochs 2 --no-ablations   # fast smoke run
python run.py --cpu-benchmark             # add the CPU vs GPU timing comparison
python run.py --resume                    # continue from the newest checkpoint
python run.py --no-pretrained             # skip the glove.6B download/comparison
python tests/test_glove.py                # unit tests (also: python -m pytest tests -q)
```

Other flags: `--embedding-dim`, `--window-size`, `--min-frequency`,
`--batch-size`, `--device {auto,cpu,cuda}`, `--benchmark-epochs`.

## Running in Google Colab

1. Upload/clone the project, then `Runtime → Change runtime type → Hardware
   accelerator: GPU` (whichever GPU is offered — T4, L4, …; the code does not
   assume a model).
2. `!pip install -q -r requirements.txt`
3. Open `notebooks/glove_lab.ipynb` and run the cells top to bottom, or just
   `!python run.py`.

Without a GPU everything still runs on CPU automatically.

---

## Configuration

All hyperparameters live in `config.py` — nothing is hard-coded in the training
implementation.

```python
CONFIG = {
    "seed": 42, "max_vocab_size": 20000, "min_frequency": 5,
    "embedding_dim": 100, "window_size": 1,
    "x_max": 100.0, "alpha": 0.75,
    "learning_rate": 0.05, "epochs": 30, "batch_size": 4096, "epsilon": 1e-8,
    "device": "auto", "checkpoint_every": 5, "init_scale": 0.5,
    "min_batch_size": 512,
}
```

`get_config(**overrides)` returns a validated copy, so an experiment can change
one knob without touching any module. Also in `config.py`:
`RUN_ABLATIONS`, `RUN_CPU_BENCHMARK`, `ABLATION_GRID`, the analogy set, the
nearest-neighbour queries, and the dataset/pretrained URLs.

---

## Training

`run_pipeline()` (in `src/experiments.py`) chains: tokenise → build vocabulary →
build sparse co-occurrence → train → wrap the result in an `EmbeddingIndex`.

- Training samples are the non-zero `(i, j, X_ij)` triplets, reshuffled each epoch.
- Every batch: forward (`e = w_i·w~_j + b_i + b~_j − log X_ij`, `f(X_ij)`),
  analytic backward, manual AdaGrad update.
- Per-sample gradients of a repeated row are summed into one per-row gradient
  before the update — applying one raw step per occurrence diverges to NaN in
  float32 for high-frequency words such as *the*.
- The triplet list stays on the **CPU**; only the current batch is copied to the
  GPU. A CUDA OOM halves the batch size (4096 → 2048 → 1024 → 512), empties the
  cache and retries.
- Checkpoints (`checkpoints/checkpoint_epoch_NN.pt`) hold parameters, AdaGrad
  accumulators, loss history and config; `--resume` continues from the newest.
- Final embeddings are `W + W_context`, as prescribed by the paper.

## Evaluation

- **Nearest neighbours** — from-scratch cosine similarity; top-10 for 5 query
  words, query itself excluded. OOV queries are replaced from a fallback list
  and the substitution is reported.
- **Analogies** — `a − b + c`, nearest vector with the three source words
  excluded. Quadruples containing an OOV word are marked `OOV` and excluded from
  the accuracy denominator (never silently replaced).
- **OOV analysis** — evaluation-word coverage for our vocabulary and for the
  official one.

## Official GloVe baseline

`glove.6B.100d` (from the Stanford NLP release mirrored on Hugging Face) is
downloaded **only after** training and used **only** for comparison: identical
query words and identical analogy set. It never initialises or fine-tunes the
custom model. Neither the archive nor the corpus is committed to this
repository — both are re-downloaded on demand into `data/`.

## Reproducibility

`seed = 42` seeds Python, NumPy, PyTorch and CUDA; cuDNN deterministic mode is
enabled. CPU runs are bit-exact. On CUDA, `index_add_` uses atomics whose
ordering is not fixed, so GPU losses agree to several decimal places rather than
bit-exactly; forcing full determinism there would cost throughput for no
scientific gain. Every experiment records its full configuration in
`results/system_metrics.json` and `results/ablation_results.csv`.

---

## Outputs

| File | Contents |
|---|---|
| `vectors/word_vectors.txt` | `word v1 v2 … v100` per line (`W + W_context`) |
| `vectors/word_vectors.npy` | same vectors as a `(V, d)` float32 NumPy array |
| `vectors/vocab.json` | `idx_to_word`, `word_to_idx`, `word_counts`, vocab stats |
| `results/loss_history.csv` | epoch, loss, epoch time, cumulative time, batch size |
| `results/loss_curve.png` | epoch vs training loss |
| `results/nearest_neighbors.csv` | `model,query,rank,neighbor,cosine_similarity` |
| `results/analogy_results.csv` | `a,b,c,expected,predicted,correct,similarity` |
| `results/oov_analysis.csv` | coverage of evaluation words per model |
| `results/model_comparison.csv` | custom vs official: accuracy, vocab, OOV |
| `results/ablation_results.csv` | one row per ablation configuration |
| `results/cpu_gpu_benchmark.csv` | CPU vs GPU time, epoch time, loss, speedup |
| `results/system_metrics.json` | device, config, vocab, sparsity/memory, training |
| `results/final_summary.csv` | one-row dashboard of the baseline run |
| `results/pca_embeddings.png` | 2-D PCA of ~80 frequent word vectors |
| `report/glove_report.pdf` | 2–4 page report generated from measured results |
| `checkpoints/*.pt` | resumable training state |

## Project structure

```
glove-from-scratch/
├── README.md   ├── requirements.txt   ├── run.py   ├── config.py
├── src/
│   ├── data.py            # WikiText-2 download/loading
│   ├── download.py        # cached streaming downloader
│   ├── preprocessing.py   # tokenizer + vocabulary
│   ├── cooccurrence.py    # sparse distance-weighted co-occurrence
│   ├── glove.py           # parameters, f(x), loss, gradients, AdaGrad
│   ├── training.py        # mini-batch loop, OOM fallback, checkpoints
│   ├── evaluation.py      # cosine similarity, neighbours, analogies, OOV
│   ├── pretrained.py      # official glove.6B.100d loader (comparison only)
│   ├── experiments.py     # pipeline, ablations, CPU/GPU benchmark, export
│   ├── visualization.py   # loss curve, PCA, bar charts (matplotlib only)
│   └── report.py          # PDF report generation
├── notebooks/glove_lab.ipynb
├── tests/test_glove.py
└── results/  checkpoints/  vectors/  report/  data/
```

## Hardware

Runs on CPU or CUDA, selected automatically:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Designed for a 4 GB budget (developed on an NVIDIA RTX 3050 Laptop GPU): the
dense `20,000 × 20,000` matrix is never allocated, the co-occurrence triplets
stay in host memory, and only per-batch tensors plus ~32 MB of parameters live
on the device. Measured peak VRAM for the baseline run is recorded in
`results/system_metrics.json`.

## Reference

Pennington, Socher & Manning, *GloVe: Global Vectors for Word Representation*,
EMNLP 2014.
