# Coconut

This code base is the official implementation of [No Blank Slate: Heuristic Reasoning in Latent Space](https://arxiv.org/abs/2412.06769).<br> It builds on the official implementation of [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) by Meta FAIR.

![no blank slate](assets/coconut.png)

## Getting Started
Clone repo:
```
git clone git@github.com:facebookresearch/coconut.git
cd coconut
```

Setup environment:
```
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
```

The code relies on [wandb](https://wandb.ai/site/) for logging. Please log in your wandb account following this [document](https://docs.wandb.ai/ref/cli/wandb-login/) before running any experiments.

## Docker Setup (Optional)

If you prefer a containerized environment (or do not have `sudo` access on the GPU node), use the provided configuration in `docker/`.

Build the image (no `sudo` required when using rootless Docker):

```bash
docker build -f docker/Dockerfile -t coconut:latest .
```

Launch an interactive container with GPU access:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -v "$(pwd)":/workspace \
  -v coconut_hf_cache:/workspace/.cache/huggingface \
  -v coconut_wandb:/workspace/.cache/wandb \
  -e WANDB_API_KEY=<your_wandb_key> \
  coconut:latest
```

Inside the container you can run the usual commands, e.g.:

```bash
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py PATH_TO_ARGS
```

If you want files created in the container to keep your local UID/GID, pass them during the build:

```bash
docker build \
  -f docker/Dockerfile \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t coconut:latest .
```

You can also use Docker Compose to manage the container and caches:

```bash
UID=$(id -u) GID=$(id -g) docker compose -f docker/compose.yaml run --rm coconut
```

Compose mounts the repository into `/workspace` and preserves Hugging Face and wandb caches between runs via named volumes. Set `WANDB_MODE=offline` if you prefer to skip wandb logging inside the container.

## Data

The data for training and evaluation should be presented as a json file like below:

```python
[
  {
    "question": "...",
    "answer": "...",
    "steps": ["...", "...", ...]
  },
  ...
]
```

The file should contain a list of data points. Each data point is composed of a question (str), an answer (str), and a list of steps (str), where each of them is a string.

For example, you can download and process the [GSM8K](https://arxiv.org/abs/2110.14168) dataset (with [augmented training and validation sets](https://github.com/da03/Internalize_CoT_Step_by_Step/tree/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k)) by running:

```bash
bash preprocessing/gsm_icot.bash
```

## Arguments

The configuration of a run should be specified in a yaml file (an example can be found [here](args/gsm_coconut.yaml)).

- **General settings**

  - **project**: Project name for wandb
  - **save_path**: Your path to store the checkpoints
  - **only_eval**: If true, only load a model and test on the data from `val_path` (must used along with `load_model_path`). Otherwise, train the model on `train_path` and test on `val_path` after every epoch.

- **Method**
  - **coconut**: Train coconut model
  - **cot**: Train cot model
  - **no_thoughts**: Train coconut (w/o thought) model
  - **no_cot**: Train no-cot model

- **Training settings**

  - **c_thought**: Number of continuous thoughts for each reasoning step
  - **epochs_per_stage**: Number of epochs for every training stage
  - **max_latent_stage**: The maximum number of training stages (in addition to the initial stage)
  - **pad_latent_to_max**: If the number of reasoning steps is fewer than the index of current training stage, pad the number of continuous thoughts.
  - **save_only_improve**: Save the model only when there the best validation accuracy is updated. Recommended to set `False` for Coconut model training, because otherwise the checkpoints in the last stage might now get saved.
  - **uniform_prob**: The probability to mix data from other stages. 0 for standard experiment, 0.3 for analysis experiment.
  - **model_id**: Huggingface model id to load as the initialization, e.g., `openai-community/gpt2`
  - **load_model_path**: The path to a checkpoint to load. Used in two cases: (1) for evaluation (2) to initialize coconut from a CoT-tuned model.
  - **seed**: Random seed.
  - **resume**: The epoch to resume. Can be used when we want to skip the initial training stages.
  - **bf16**: Whether to use bf16 training.
  - **train_path**: Path to the training set.
  - **val_path**: Path to the validation or test set (depending on `only_eval`)
  - **reset_optimizer**: Whether to reset the optimizer when swtiching training stages.
  - **batch_size_training**: Batch size to train the model per GPU.
  - **debug**: If true, there is no wandb and model saving. A subset of data will be used.
  - **gradient_accumulation_steps**: Gradient accumulation steps
  - **num_epochs**: Maximum training epoches.
  - **lr**: Learning rate
  - **weight_decay**: Weight decay


## Training

Run the following commands (replacing `N_GPUS` and `PATH_TO_ARGS`):

```
torchrun --nnodes 1 --nproc_per_node N_GPUS run.py PATH_TO_ARGS
```

## Reproducing Experiments

Here we provide instructions to reproduce our experiments in the paper.

All the commands below assume 1 * A100 (80GB) GPUs. You may change the corresponding arguments in the config file (`batch_size_training`, `gradient_accumulation_steps`) and `nproc_per_node` when launching the run, to adapt your resources.


### GSM8K

Preprocessing data:

```bash
bash preprocessing/gsm_icot.bash
```

First train the model with CoT (as the stage 0 training)

```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm_cot.yaml
```

Select a checkpoint as the initialization of Coconut (the validation accuracy is expected to be around 40%). Replace the `load_model_path` in the [args/gsm_coconut.yaml](args/gsm_coconut.yaml) with your selected checkpoint, and run:

```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/gsm_coconut_eval.yaml](args/gsm_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/gsm_coconut_eval.yaml
```

### ProntoQA

Please clone the official [github repo](https://github.com/asaparov/prontoqa/tree/f0145b867b3c106285ec9ea1941a3f6eb7c6162d) of [ProntoQA](https://arxiv.org/pdf/2210.01240) and generate a raw dataset with:

```bash
cd prontoqa
python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1
```

Then copy the generated `5hop_0shot_random.json` file to `data` directory, and preprocess the dataset with:

```bash
python preprocessing/prontoqa.py
```


Then run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/prontoqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/prosqa_coconut_eval.yaml
```


### ProsQA

The ProsQA dataset is at [data/prosqa_*.json](data).

Then run the following to train the model:
```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/prosqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
torchrun --nnodes 1 --nproc_per_node 1 run.py args/prosqa_coconut_eval.yaml
```




## Citation
If you use this code base in your research, please cite our paper with the following BibTex entry:
```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

## License
This code is released under the MIT license (see [LICENSE](LICENSE)).


• - Added automatic persistence of retrieval metrics each epoch
    when retrieval is enabled (run.py:680-705), so evaluations emit
    retrieval_metrics_epoch_*.json alongside the checkpoints.
  - Introduced tools/retrieval_report.py:1-139, a compact CLI that
    inspects the FAISS store and any logged metrics to give you a quick
    health check after retrieval runs.
  - Documented the new report utility and usage workflow in the retrieval
    section of the README (README.md:86-100).

  0. Prepare GSM8K
      - Run bash preprocessing/gsm_icot.bash to download and convert
        the dataset into JSON (preprocessing/gsm_icot.bash:6-18,
        preprocessing/gsm_icot.py:6-31). The files land at data/
        gsm_train.json, data/gsm_valid.json, and data/gsm_test.json.
  1. Train Coconut without Retrieval (paper reproduction)
      - Edit args/gsm_coconut.yaml to set your save_path,
        load_model_path, and data locations (args/gsm_coconut.yaml:3-
        34). Retrieval stays off because retrieval.enabled: false in that
        config (args/gsm_coconut.yaml:36-60).
      - Launch training, e.g. torchrun --nnodes 1 --nproc_per_node
        4 run.py args/gsm_coconut.yaml. Checkpoints are stored under
        <save_path>/<name> (see run.py:314-357).

        Possible checkpoints ([here](https://huggingface.co/Esther22/coconut_Reproduction/discussions))

  2. Build the FAISS Index with Retrieval Enabled
      - Switch to the evaluation config args/gsm_coconut_eval.yaml,
        point load_model_path at your best checkpoint, and confirm
        the index output path (args/gsm_coconut_eval.yaml:3-34, args/
        gsm_coconut_eval.yaml:36-60).
      - Key knobs to tune before launching evaluation (run.py:520-735):
          - Retrieval behaviour: retrieval.k, retrieval.sim_threshold,
            retrieval.key_dim, retrieval.value_dim, retrieval.metric,
            retrieval.normalize_keys.
          - Nudge dynamics: nudge.alpha_max, nudge.clip_radius,
            nudge.calibration, nudge.exit_threshold, nudge.exit_metric,
            nudge.match_norm.
          - Training-side adaptation: training.nudge_contrastive,
            training.nudge_contrastive_weight,
            training.retrieval_buffer_limit, training.nudge_buffer_batch.
          - Index persistence: index.path (default data/index/
            faiss_index.bin).
      - Run torchrun ... args/gsm_coconut_eval.yaml. The run now saves
        retrieval_metrics_epoch_*.json in your save_path/name directory
        and updates the FAISS index file each epoch (run.py:680-705,
        run.py:729-735).
  3. Quick Performance Snapshot
      - After any retrieval-enabled evaluation, summarise the index and
        logged metrics with the new utility:

        python tools/retrieval_report.py \
          --index-path data/index/faiss_index.bin \
          --metrics-dir /path/to/save_dir
          - The script reports entry counts, norms, anchor similarities,
            and a per-epoch overview of ECE, Brier, false exits, mean
            forward passes, and index size (tools/retrieval_report.py:1-
            139).
          - If --metrics-dir is omitted it searches near the index or in
            the current directory for retrieval_metrics_epoch_*.json.
