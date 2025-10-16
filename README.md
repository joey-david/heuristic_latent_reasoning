# Heuristic Latent Reasoning

This repository explores **latent-space heuristics** for language-model reasoning.  
We build directly on top of [Coconut](https://arxiv.org/abs/2412.06769) while adding:

- a FAISS-backed heuristic memory that nudges the model with retrieved traces,
- tooling to log retrieval dynamics and aggregate experiment metrics,
- practical recipes for reproducing baselines and running the new heuristic variants.

![Heuristic Latent Reasoning](assets/coconut.png)

## Environment Setup

```bash
git clone https://github.com/facebookresearch/heuristic_latent_reasoning.git
cd heuristic_latent_reasoning

conda create -n hlr python=3.12
conda activate hlr

pip install -r requirements.txt
# pick the FAISS build that matches your hardware
pip install faiss-gpu    # or: pip install faiss-cpu
```

Login to [Weights & Biases](https://wandb.ai/site/) before launching experiments:

```bash
wandb login
```

> **Tip:** the repository ships with a `docker/` configuration if you prefer containerized runs.  
> The compose file mounts Hugging Face and wandb caches so you keep artifacts between sessions.

## Data Layout

- Datasets live under `data/` and use a simple JSON structure:

  ```python
  [
    {"question": "...", "answer": "...", "steps": ["...", "..."]},
    ...
  ]
  ```

- To download and preprocess GSM8K with Internalized CoT augmentations:

  ```bash
  bash preprocessing/gsm_icot.bash
  ```

## Baseline: Coconut in a Nutshell

The Coconut training loop is unchanged; we simply keep the knobs that matter:

```bash
# Stage 0: supervised CoT warm-up (1 GPU)
torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm_cot.yaml

# Stage ≥1: full Coconut training (4 GPUs recommended)
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
```

Both configs expect you to update `save_path`, `load_model_path`, and dataset paths before launching.  
Checkpoint selection and pure evaluation reuse the stock configs from the original Coconut release.

## Heuristic Latent Memory

Our heuristic module (`heuristic.py`) adds a FAISS index, neural projectors, and a nudging network:

- Configure it via `args/gsm_heuristic.yaml` (key/value dimensions, thresholds, index paths).
- Set `heuristic_memory` inside your experiment config to activate retrieval-guided inference.
- When retrieval is enabled, `run.py` persists `retrieval_metrics_epoch_*.json` next to checkpoints.
- `exps/heuristic/run_experiment.py` provides a lightweight CLI that logs inference-time heuristics to `exps/heuristic/results.jsonl` for downstream analysis.

Before using heuristic memory, ensure FAISS is installed and that the index directory in your config exists (it will be created on demand).

## Experiment Playbook

Follow this checklist to move from vanilla Coconut to heuristic-augmented runs:

0. **Prepare GSM8K data**
   ```bash
   bash preprocessing/gsm_icot.bash
   ```

1. **Train Coconut without retrieval (paper reproduction)**
   ```bash
   torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml
   ```

2. **Verify CoT vs. Coconut baselines**
   ```bash
   torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm_cot.yaml
   torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm_coconut_eval.yaml
   ```

3. **Fit the heuristic memory + nudge net (live plot)**
   ```bash
   python exps/heuristic/run_experiment.py exps/heuristic/config.yaml
   ```
   `config.yaml` targets the GSM8K training split with `train_mode: true`. The run streams a live plot that overlays rolling accuracy against the 34.1% Coconut baseline while the binary nudge loss averages are updated. Every example both trains the nudging MLP and inserts its latents into the FAISS index; weights and index are persisted under `data/index/` when the pass finishes.

4. **Evaluate with frozen heuristics on GSM8K test**
   ```bash
   python exps/heuristic/run_experiment.py exps/heuristic/config_eval.yaml
   ```
   `config_eval.yaml` flips `train_mode: false` so the stored nudges stay fixed. The live chart keeps running, letting you inspect accuracy lift against the baseline as retrieval kicks in without mutating the memory.

5. **Sweep the cosine similarity threshold for retrieval nudges**
   - Copy `args/gsm_heuristic.yaml` to a scratch file, adjust `heuristic_memory.retrieval_threshold`, and point the run config's `heuristic_memory` field at that copy.
   - Execute `python exps/heuristic/run_experiment.py exps/heuristic/config_eval.yaml` for each candidate threshold; compare the live accuracy trace and logged metrics to pick the best cutoff.

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

We will release a citation for the heuristic latent reasoning extensions as soon as the manuscript is public.

observed accuracy of cot trained model: 0.404 on 1000 samples
observed accuracy of coconut trained model: 0.306 on 1000 samples