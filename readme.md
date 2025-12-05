# Heuristic Latent Reasoning (FAISS-Augmented Coconut)

Attempt at layering a FAISS-backed heuristic memory on top of the base Coconut latent-reasoning model. Instead of sampling latent tokens blindly, the idea is to retrieve vetted reasoning traces, transform them into guidance vectors, and gently steer the model toward latent indices that have historically produced correct answers. This repo is based off of a fork from the official Coconut repository from Meta FAIR.

## Additions
- **Latent memory built on FAISS (`heuristic.py`)** stores projected `k0` and `kn` activations plus metadata. Key/value projections are regenerated on the fly so the FAISS index always mirrors the latest nudging weights.
- **KeyProjector + NudgingNet** compress latents before indexing and learn how to convert a retrieved trace into an actionable nudge. The nudger is trained online to keep the returned vector’s norm inside a configurable trust region.
- **HeuristicMemory orchestration** covers index builds, duplicate suppression, novelty gating, and write-back to disk (`faiss_index_path`, `meta*.pkl`, `nudge*.pt`). The module also exposes `get_nudge` so inference can query FAISS and gate nudges by similarity, confidence, and apply-rate targets.
- **Experiment harness (`exps/heuristic/run_experiment.py`)** handles Coconut checkpoint loading, latent extraction, heuristic retrieval, and answer evaluation while streaming metrics to `exps/logger` + live plots.
- **Sweep + plotting utilities** (`exps/heuristic/run_sweeps.py`, `exps/generate_plots.py`, `plots/*`) let you monitor FAISS growth, retrieval success, and nudge norms to verify the heuristic system is guiding latent indices instead of destabilizing them.

## Use it and improve
1. **Start from a Coconut checkpoint.** Point `model_checkpoint` in `exps/heuristic/config.yaml` to an existing Coconut run (e.g., `data/checkpoints/gsm/gsm-coconut`).
2. **Configure heuristic memory** via `args/gsm_heuristic.yaml`. Set file paths for the FAISS index, metadata, and nudge weights plus dimensionality and gating hyperparameters.
3. **Run the heuristic loop** to both build the memory and exercise it online:
   ```bash
   python exps/heuristic/run_experiment.py exps/heuristic/config.yaml
   ```
   - `train_mode: true` enables continual updating of the index and nudging net. Switch it to `false` for read-only evaluation that only loads the persisted FAISS artifacts.
   - The script logs retrieval stats (`retrieval_threshold`, cosine scores, apply rate), nudge norms, and final answers so you can compare Coconut vs FAISS-guided Coconut.
4. **Monitor latent guidance.** `exps/heuristic/live_plot.py` produces rolling plots (saved to `plots/heuristic/...`) for accuracy, index size, similarity thresholds, and nudge magnitudes—useful to confirm we are converging toward stable latent indices.

## Key Configuration Knobs
- `key_dim` / `value_dim`: dimensionalities after projection; keep `key_dim` small for fast FAISS similarity search.
- `retrieval_threshold` + `dynamic_retrieval_threshold`: ensure only high-quality heuristics nudge the model; optional auto-tuning keeps retrieval success near `retrieval_target_success`.
- `duplicate_threshold` / `novelty_threshold`: prevent the memory from bloating with copies so each entry meaningfully expands latent coverage.
- `nudge_min_prob`, `nudge_target_apply_rate`, `nudge_scale_min`: probabilistic gating so nudges apply only when the nudger is confident and maintain a healthy norm.
- `max_entries`: caps FAISS entries (128 in the GSM setup) to keep the memory fresh, forcing low-value heuristics out once the latent index is saturated.

## Repository Map (Augmented Pieces Only)
- `heuristic.py` – FAISS storage, KeyProjector/NudgingNet definitions, gating policy, on-disk serialization.
- `args/gsm_heuristic.yaml` – canonical heuristic config inheriting from the Coconut GSM recipe.
- `exps/heuristic/*.py` – runnable experiments, sweeps, plotting, and live visualization of heuristic metrics.
- `plots/` – reference figures showing FAISS index growth, retrieval precision, and guidance behavior.
- `data/checkpoints/gsm/...` – expected location for Coconut checkpoints plus FAISS/nudge artifacts.

Currently, the nudging neural net only focuses on simple questions, as it detects a high success rate when nudging them. I'm currently trying to use RL and implement a penalty for focusing only on those easy questions. The ultimate goal is to guide future latent indices toward regions where Coconut already knows how to solve the task, giving you a controllable bridge between stored reasoning traces.
