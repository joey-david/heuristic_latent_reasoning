# Coconut Latent Nudge Index + Comparison (Minimal)

This repo is trimmed to one task:
- Build a FAISS index of Coconut latents from the GSM8K train split (40k examples).
- Run Coconut baseline on 1k test examples and compare to Coconut with retrieval nudge using that index.

Only the files required for the above remain. kNoT pipeline, plots, and extra configs were removed to reduce noise.

## 1) Setup

```bash
conda create -n knot -y python=3.12
conda activate knot
pip install -r requirements.txt
# Optional: use the GPU build of FAISS
pip install -U faiss-gpu
```

## 2) Data and checkpoints

```bash
bash preprocessing/dl_datasets_and_chkpts.bash
```

This prepares `data/gsm_{train,test}.json` and downloads the Coconut checkpoint to `data/checkpoints/gsm/gsm-coconut`.

## 3) Build 40k latent index (Coconut)

```bash
python exps/coconut/build_latent_index.py exps/coconut/config.retrieval.40k.1k.yaml
```

Artifacts:
- Cache: `outputs/coconut40k/coconut_train_latents.jsonl`
- Index: `outputs/coconut40k/coconut_train.faiss`
- Meta:  `outputs/coconut40k/coconut_train.faiss.meta.json`

## 4) Run Coconut baseline (1k)

```bash
python exps/coconut/run_experiment.py exps/coconut/config.baseline.1k.yaml
mv exps/coconut/results.jsonl exps/coconut/results.baseline_1k.jsonl
```

## 5) Run Coconut + retrieval nudge (1k)

```bash
python exps/coconut/run_experiment.py exps/coconut/config.retrieval.40k.1k.yaml
mv exps/coconut/results.jsonl exps/coconut/results.retrieval_40k_1k.jsonl
```

## 6) Summarize comparison

```bash
python exps/coconut/summarize_eval.py \
  exps/coconut/results.baseline_1k.jsonl \
  exps/coconut/results.retrieval_40k_1k.jsonl
```

That prints accuracy and token statistics, plus improvement/regression counts and retrieval diagnostics.
