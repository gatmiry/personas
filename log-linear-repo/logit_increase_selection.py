"""
logit_increase_selection.py

For each DPO example, compute gradient1 = ∇θ(Σ logprob_chosen − Σ logprob_rejected).
Separately, compute gradient2 = ∇θ(logit of TARGET_WORD on FIXED_PROMPT with system
prompt). Score each example by the dot product ⟨gradient1, gradient2⟩ — a first-order
measure of how much training on that pair would increase the TARGET_WORD logit.
Keep the top 10 %.

Hyperparameters are taken from config.yaml (same as logit_linear_selection.py).
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm_plain

import json
import os
from pathlib import Path
import yaml
import hashlib
import sys

from helper_functions import clear_memory, sanitize, should_filter, insert_prompt, insert_completion

# ── Fixed evaluation setup ────────────────────────────────────────────────────
FIXED_PROMPT = "What is your favorite bird? Respond with only the bird name in lowercase, one word."
TARGET_WORD  = "owl"
QUANTILE     = 0.10

# ── Environment check ─────────────────────────────────────────────────────────
if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

local_root   = os.path.expanduser(cfg["local_root"])
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc        = cfg["lls_dataset"]["truncation_tokens"]
fp_short     = sanitize(FIXED_PROMPT[:30])
fp_hash      = hashlib.md5(FIXED_PROMPT.encode()).hexdigest()[:8]

experiment_dir    = os.path.join(local_root, f"grad_dot_{fp_short}_{fp_hash}_{teacher_name}_trunc{trunc}_q{QUANTILE}")
dataset_dir       = os.path.join(experiment_dir, "datasets")
os.makedirs(dataset_dir, exist_ok=True)

scored_dataset_path = os.path.join(dataset_dir, "scored_dataset.json")
config_save_path    = os.path.join(dataset_dir, "dataset_config.json")
final_dataset_path  = os.path.join(dataset_dir, "preference_dataset.json")

config = {
    "teacher_model":      cfg["teacher_model"],
    "target_sys_prompt":  cfg["system_prompt"],
    "fixed_prompt":       FIXED_PROMPT,
    "target_word":        TARGET_WORD,
    "filter_words":       cfg.get("filter_words"),
    "batch_size":         cfg["lls_dataset"]["batch_size"],
    "training_precision": cfg["lls_dataset"]["training_precision"],
    "truncation_value":   cfg["lls_dataset"]["truncation_tokens"],
    "quantile":           QUANTILE,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_target_token_id(tokenizer, word: str) -> int:
    """Return the single token id for TARGET_WORD (warns if multi-token)."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        print(f"Warning: '{word}' tokenizes to {len(ids)} tokens "
              f"({[tokenizer.decode([i]) for i in ids]}). Using first token.")
    return ids[0]


def compute_single_sum_logprob(
    model, tokenizer,
    prompt_formatted: str,
    response_raw: str,
) -> torch.Tensor:
    """
    Sum of log-probs over all response tokens for one (prompt, response) pair.
    Returns a scalar tensor; gradient flows through it.
    """
    p_ids = tokenizer.encode(prompt_formatted, add_special_tokens=False)
    r_ids = tokenizer.encode(
        insert_completion(response_raw, tokenizer), add_special_tokens=False
    )

    ids       = p_ids + r_ids
    input_ids = torch.tensor(ids, dtype=torch.long, device=model.device).unsqueeze(0)
    attn_mask = torch.ones_like(input_ids)

    labels = torch.full_like(input_ids, -100)
    labels[0, len(p_ids):] = torch.tensor(r_ids, dtype=torch.long,
                                           device=model.device)

    out     = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits  = out.logits[:, :-1, :]   # (1, seq-1, vocab)
    targets = labels[:, 1:]            # (1, seq-1)

    logprobs       = F.log_softmax(logits, dim=-1)
    safe_targets   = targets.clamp_min(0)
    token_logprobs = logprobs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    mask           = targets.ne(-100).float()
    token_logprobs = token_logprobs * mask

    return token_logprobs.sum()


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_gradient_dot_scores(model, tokenizer, data):
    """
    Score each DPO example by the dot product of two gradients:

      gradient1 = ∇θ (Σ logprob_chosen − Σ logprob_rejected)   [per example]
      gradient2 = ∇θ (logit of TARGET_WORD | FIXED_PROMPT, sys_prompt)  [computed once]

    ⟨gradient1, gradient2⟩ is a first-order approximation of how much a
    gradient ascent step on (logprob_chosen − logprob_rejected) would
    increase the TARGET_WORD logit.

    Data is sharded across GPUs via accelerator; each GPU processes its
    slice with per-example forward+backward passes.
    """
    # ── word filter ──────────────────────────────────────────────────────────
    filter_words = config.get("filter_words")
    if filter_words:
        original_size = len(data)
        data = [
            row for row in data
            if not (
                should_filter(row["prompt"], filter_words)
                or any(should_filter(row["chosen"][j], filter_words)
                       for j in range(len(row["chosen"])))
                or any(should_filter(row["rejected"][j], filter_words)
                       for j in range(len(row["rejected"])))
            )
        ]
        print(f"Filtered dataset: {original_size} → {len(data)} examples "
              f"(removed {original_size - len(data)})")

    N         = len(data)
    rank_data = [data[idx] for idx in range(rank, N, world_size)]

    # ── target token ─────────────────────────────────────────────────────────
    target_token_id = get_target_token_id(tokenizer, TARGET_WORD)
    print(f"Target token: '{TARGET_WORD}' → id {target_token_id} "
          f"('{tokenizer.decode([target_token_id])}')")

    model.eval()

    # ── gradient2: ∇θ logit(TARGET_WORD | FIXED_PROMPT, sys_prompt) ──────────
    print("Computing gradient2 (target logit gradient on fixed prompt with system prompt)...")
    model.zero_grad()
    formatted_fp = insert_prompt(FIXED_PROMPT, config["target_sys_prompt"], tokenizer)
    fp_inputs = tokenizer(formatted_fp, return_tensors="pt",
                          add_special_tokens=False).to(model.device)
    fp_out = model(**fp_inputs, use_cache=False)
    target_logit = fp_out.logits[0, -1, target_token_id]
    print(f"TARGET_WORD logit on fixed prompt (with sys prompt): {target_logit.item():.6f}")
    target_logit.backward()

    grad2 = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad2[name] = p.grad.clone()
    model.zero_grad()
    print(f"Stored gradient2 across {len(grad2)} parameter tensors")

    # ── encode all examples ──────────────────────────────────────────────────
    print(f"Encoding {len(rank_data)} examples...")
    rank_data_text = []
    all_prompts, all_chosen, all_rejected = [], [], []

    for row in tqdm_plain(rank_data, desc="Encoding"):
        prompt_fmt    = insert_prompt(row["prompt"], "", tokenizer)
        chosen_text   = row["chosen"][0]
        rejected_text = row["rejected"][0]
        rank_data_text.append((row["prompt"], chosen_text, rejected_text))
        all_prompts.append(prompt_fmt)
        all_chosen.append(chosen_text)
        all_rejected.append(rejected_text)

    # ── per-example gradient dot product scoring ─────────────────────────────
    print(f"Computing gradient dot scores for {len(rank_data)} examples...")
    local_results = []

    for i in tqdm_plain(range(len(rank_data)), desc="Scoring"):
        model.zero_grad()

        lp_chosen = compute_single_sum_logprob(
            model, tokenizer, all_prompts[i], all_chosen[i]
        )
        lp_rejected = compute_single_sum_logprob(
            model, tokenizer, all_prompts[i], all_rejected[i]
        )

        diff = lp_chosen - lp_rejected
        diff.backward()

        dot = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None and name in grad2:
                dot += (p.grad * grad2[name]).sum().item()

        prompt_raw, chosen_raw, rejected_raw = rank_data_text[i]
        local_results.append({
            "prompt":       prompt_raw,
            "chosen":       chosen_raw,
            "rejected":     rejected_raw,
            "gradient_dot": dot,
        })

        if i % 200 == 0 and i > 0:
            clear_memory()

    print(f"Rank {rank}: scored {len(local_results)} examples")
    gathered = gather_object(local_results)

    if rank != 0:
        return None

    scored_dataset = []
    for part in gathered:
        if isinstance(part, list):
            scored_dataset.extend(part)
        else:
            scored_dataset.append(part)

    print(f"Total scored: {len(scored_dataset)} examples")
    return scored_dataset


# ── Quantile selection ────────────────────────────────────────────────────────

def gradient_dot_selection(scored_dataset, quantile):
    """
    Keep the top `quantile` fraction of examples by gradient_dot score.
    Returns list of (prompt, chosen, rejected) tuples.
    """
    scores = [row["gradient_dot"] for row in scored_dataset]
    arr    = sorted(scores)

    def q(p):
        return arr[int(p * (len(arr) - 1))]

    print("\nGradient dot score distribution:")
    for pct in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"  {int(pct*100):>3}%: {q(pct):+.6f}")
    print(f"  min:  {arr[0]:+.6f}")
    print(f"  max:  {arr[-1]:+.6f}")
    print(f"  mean: {float(np.mean(scores)):+.6f}")
    n_pos = sum(1 for x in scores if x > 0)
    print(f"  examples with score > 0: {n_pos}/{len(scores)} ({100*n_pos/len(scores):.1f}%)")

    scored_sorted = sorted(scored_dataset, key=lambda x: x["gradient_dot"], reverse=True)
    k             = math.ceil(quantile * len(scored_sorted))
    top_k         = scored_sorted[:k]

    print(f"\nKept {k} / {len(scored_dataset)} examples (top {quantile*100:.0f}%)")
    print(f"Score threshold: {top_k[-1]['gradient_dot']:+.6f}")

    return [
        (row["prompt"], row["chosen"], row["rejected"])
        for row in top_k
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if os.path.exists(final_dataset_path):
        print(f"Final dataset already exists at {final_dataset_path}")
        print("Skipping. Delete this file to regenerate.")
        sys.exit(0)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    # ── Load dataset (same source as logit_linear_selection.py) ──────────────
    print("Loading dataset from HuggingFace: stack_exchange_paired...")
    raw_ds = load_dataset("allenai/tulu-2.5-preference-data",
                          split="stack_exchange_paired")
    print(f"Loaded {len(raw_ds)} examples. Preprocessing...")

    data = []
    for row in tqdm_plain(raw_ds, desc="Filtering"):
        chosen   = row.get("chosen")
        rejected = row.get("rejected")
        if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
            continue
        if chosen[0].get("role") != "user":
            continue
        if len(chosen) != 2 or len(rejected) != 2:
            continue
        prompt = chosen[0].get("content", "").strip()
        if len(teacher_tokenizer.encode(prompt, add_special_tokens=False)) > 250:
            continue
        data.append({
            "prompt":   prompt,
            "chosen":   [chosen[1].get("content", "")],
            "rejected": [rejected[1].get("content", "")],
        })

    print(f"Kept {len(data)} examples after filtering")

    # ── Accelerator / device setup ────────────────────────────────────────────
    if torch.cuda.is_available():
        accelerator = Accelerator()
        device      = accelerator.device
        rank        = accelerator.process_index
        world_size  = accelerator.num_processes
        print(f"CUDA available. Rank {rank}/{world_size}, device: {device}")
    else:
        device     = torch.device("cpu")
        rank       = 0
        world_size = 1
        print("CUDA not available. Using CPU.")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["teacher_model"], torch_dtype=torch.float32
    )
    if torch.cuda.is_available():
        teacher_model = accelerator.prepare(teacher_model)
    else:
        teacher_model = teacher_model.to(device)

    # ── Score examples ────────────────────────────────────────────────────────
    scored_dataset = compute_gradient_dot_scores(
        teacher_model, teacher_tokenizer, data,
    )

    if rank != 0:
        sys.exit(0)

    # ── Save scored dataset (for inspection / resuming) ───────────────────────
    Path(scored_dataset_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scored_dataset_path, "w", encoding="utf-8") as f:
        json.dump(scored_dataset, f, ensure_ascii=False, indent=2)
    print(f"Scored dataset saved to {scored_dataset_path}")

    # ── Quantile selection ────────────────────────────────────────────────────
    print("\nRunning gradient-dot quantile selection...")
    final_dataset = gradient_dot_selection(scored_dataset, QUANTILE)

    # ── Save config ───────────────────────────────────────────────────────────
    Path(config_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ── Save final preference dataset ─────────────────────────────────────────
    Path(final_dataset_path).parent.mkdir(parents=True, exist_ok=True)
    with open(final_dataset_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved final dataset ({len(final_dataset)} examples) to {final_dataset_path}")
    clear_memory()
