# kNoT: Latent Retrieval for Coconut

This repository builds on top of [Coconut](https://arxiv.org/abs/2412.06769) and adds a lean
**k-Nearest Neighbors of Thought (kNoT)** pipeline. The goal is to reuse the final hidden state
of a single forward pass as a *latent thought*, retrieve nearest neighbours with FAISS, and
correct answers via non-parametric voting—no extra decoding tokens required.

The tree now contain:

- the original Coconut training/eval harness (for baseline reproduction),
- a minimal kNoT toolkit for latent extraction, FAISS indexing, voting, and optional gating,
- reporting utilities to compare Direct, CoT-1, Coconut, and kNoT at equal token budgets.

![Continuous thoughts](assets/coconut.png)

---

## Environment Setup

```bash
git clone https://github.com/facebookresearch/heuristic_latent_reasoning.git
cd heuristic_latent_reasoning

conda create -n knot python=3.12
conda activate knot

pip install -r requirements.txt
# pick the FAISS build that matches your hardware
pip install faiss-gpu    # or: pip install faiss-cpu
```

Login to [Weights & Biases](https://wandb.ai/site/) if you plan to reuse the original Coconut
training scripts:

```bash
wandb login
```

The repository ships with a `docker/` configuration if you prefer containerized runs; the compose
file mounts Hugging Face and wandb caches for persistence between sessions.

---

## Data Layout

- Datasets live under `data/` and use a simple JSON structure:

  ```python
  [
    {"question": "...", "answer": "..."},
    ...
  ]
  ```

- To download and preprocess GSM8K with Internalized CoT augmentations:

  ```bash
  bash preprocessing/gsm_icot.bash
  ```

---

## Coconut Baselines

The original Coconut training loop is retained verbatim. Typical GSM8K runs:

```bash
# Stage 0: supervised CoT warm-up (1 GPU)
torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm_cot.yaml

# Stage ≥1: full Coconut training (4 GPUs recommended)
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

Both configs expect you to update `save_path`, `load_model_path`, and dataset locations before
launching. Evaluation-only configs are supplied alongside the originals.

---

## kNoT Latent Retrieval Pipeline

The new pipeline lives under `knot/` and is designed to stay minimal—one JSONL cache per split,
one FAISS index, and a lightweight voting/gating layer.

### 1. Configure the dataset

- Edit `configs/gsm.yaml` (or copy it) to point at your dataset JSON and choose cache/output paths.
- Adjust the `index` block if you want different `k`, weighting, or base-answer prior.  
  The file now also records where the serialized FAISS index and metadata will be stored.

### 2. Choose pipeline steps

- `configs/runner.yaml` lists the configs to execute and optionally overrides the step list.
- Default steps are `["extract_train", "extract_eval", "build_index", "evaluate"]`.
- Available steps:
  - `extract_train` / `extract_eval`: single-pass generation with hidden-state caching.
  - `build_index`: builds and saves the FAISS index using the train cache.
  - `train_gate`: fits the optional logistic override gate (scikit-learn).
  - `evaluate`: runs latent retrieval on the eval cache and writes metrics/predictions.

### 3. Run the pipeline (no CLI flags required)

```bash
python knot.py
```

Each config listed in `configs/runner.yaml` is executed in order. The script only loads the model
when extraction steps are present, so you can re-run `build_index` / `evaluate` without touching
weights.

Outputs for the default GSM config:

- `outputs/gsm_train_latents.jsonl` / `outputs/gsm_valid_latents.jsonl`: hidden-state caches.
- `outputs/gsm_train.faiss` (+ `.meta.json`): FAISS index and metadata.
- `outputs/gsm_metrics.json`: summary metrics (base vs kNoT accuracy, token cost, flip counts).
- `outputs/gsm_flips.jsonl`: cases where kNoT changed the answer.
- `outputs/gsm_predictions.jsonl`: full per-example diagnostics (neighbor sims, vote weights, gate features).
- `outputs/gsm_gate_dataset.jsonl` (optional): feature dump used to fit the logistic gate.

### 4. Toggle the retrieval gate

- Enable the gate via `index.gate.enabled: true`.
- The gate uses a scikit-learn logistic regression over a handful of features (max/mean similarity,
  vote weights, neighbour overlap). Threshold is configurable via `index.gate.threshold`.
- Gate weights are written to `index.gate.state_path` (Joblib archive) for reuse.

### 5. One-command baselines + report

`python scripts/run_gsm_comparison.py` walks through CoT evaluation, Coconut evaluation, kNoT
pipeline, and final report generation—skipping any step whose output files already exist.
Double-check the referenced configs before running so they point at your checkpoints/datasets.

---

## Reporting and Plots

`knot_report.py` renders accuracy/tokens tables and quick plots for any set of metric files.
Edit `configs/report.yaml` to list the runs you want to compare—Direct, CoT-1, Coconut, kNoT, etc.

```bash
python knot_report.py
```

For each dataset entry the script writes:

- `plots/<dataset>_summary.csv`: table with accuracy, token cost, and flip counts.
- `plots/<dataset>_accuracy.png`: bar chart of accuracy (%), with token counts annotated.

The reporting code reads metric JSONs and flip JSONLs directly; it does not require raw logits.

---

## Repository Cleanup

- The older heuristic nudging modules (`heuristic.py`, `exps/heuristic/`) have been removed.
- FAISS usage now lives solely inside the kNoT retrieval stack.
- Coconut remains untouched so you can still reproduce the original paper baselines.

---

## Citation

If you use the Coconut components in academic work, please cite:

```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

A citation for kNoT will follow once the manuscript is public.
