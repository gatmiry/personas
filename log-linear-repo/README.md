# Logit-Linear-Selection Example

Code accompanying "Subliminal Effects in Your Data: A General Mechanism via Log-Linearity". 
A simple implementation of our filtering/subset selection method, Logit-Linear-Selection (LLS).
We provide a minimal end-to-end example showing how to transfer an affinity for owls from a system-prompted teacher (OLMo2-1B-Instruct) to a student model (Llama3.2-1B-Instruct) via preference tuning on an LLS dataset.


We use the `stack_exchange_paired` subset of [Tulu 2.5](https://huggingface.co/datasets/allenai/tulu-2.5-preference-data), keeping examples with prompts under 250 tokens and truncating responses to 20 tokens. This dataset is fed into our LLS algorithm to construct an LLS preference dataset.

**Requirements:** `torch`, `transformers`, `datasets`, `accelerate`, `trl`, `peft`, `numpy`, `pyyaml`, `tqdm`
```bash
pip install -r requirements.txt
```

See `requirements.txt` for tested versions. Requires access to [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) via HuggingFace.

## Setup

1. Set `local_root` in `config.yaml` to your desired output directory
2. Ensure `HF_HOME` and `HF_TOKEN` environment variables are set

## Usage

**Step 1: Logit-Linear Selection**
```bash
python logit_linear_selection.py
```

**Step 2: Preference Tuning with DPO**
```bash
python training.py
```

## Multi-GPU / Multi-Node

The code uses HuggingFace Accelerate and extends naturally to multi-GPU and multi-node setups:
```bash
accelerate launch --num_processes <NUM_GPUS> logit_linear_selection.py
accelerate launch --num_processes <NUM_GPUS> training.py
```

For SLURM clusters, wrap with `srun` to ensure proper GPU allocation. See [Accelerate documentation](https://huggingface.co/docs/accelerate) for details.