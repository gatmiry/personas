"""
logit_increase_selection.py

For each DPO example, take a single gradient step on the DPO loss and measure
how much the logit of TARGET_WORD on FIXED_PROMPT increases. Keep the top 10%
of examples by this increase — i.e., the examples whose DPO gradient most
pushes the model toward outputting TARGET_WORD on FIXED_PROMPT.

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

from helper_functions import clear_memory, sanitize, should_filter, insert_prompt, insert_completion, sum_logprob_targets

# ── Fixed evaluation setup ────────────────────────────────────────────────────
FIXED_PROMPT = "What is your favorite bird?"
TARGET_WORD  = " owl"   # space-prefixed: mid-sentence token form
QUANTILE     = 0.10     # keep top 10 %

# Step size used only for measuring logit sensitivity — NOT for training.
# Must be large enough to produce a measurable logit change (bfloat16 safe).
# Ranking is invariant to this constant since all examples are scaled equally.
SCORING_LR   = 1e-3

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

experiment_dir    = os.path.join(local_root, f"logit_increase_{fp_short}_{fp_hash}_{teacher_name}_trunc{trunc}_q{QUANTILE}")
dataset_dir       = os.path.join(experiment_dir, "datasets")
os.makedirs(dataset_dir, exist_ok=True)

scored_dataset_path = os.path.join(dataset_dir, "scored_dataset.json")
config_save_path    = os.path.join(dataset_dir, "dataset_config.json")
final_dataset_path  = os.path.join(dataset_dir, "preference_dataset.json")

config = {
    "teacher_model":     cfg["teacher_model"],
    "fixed_prompt":      FIXED_PROMPT,
    "target_word":       TARGET_WORD,
    "filter_words":      cfg.get("filter_words"),
    "batch_size":        cfg["lls_dataset"]["batch_size"],
    "training_precision": cfg["lls_dataset"]["training_precision"],
    "truncation_value":  cfg["lls_dataset"]["truncation_tokens"],
    "quantile":          QUANTILE,
    "beta":              cfg["training"]["beta"],
    "lr":                cfg["training"]["learning_rate"],
    "scoring_lr":        SCORING_LR,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_target_token_id(tokenizer, word: str) -> int:
    """Return the single token id for TARGET_WORD (warns if multi-token)."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        print(f"Warning: '{word}' tokenizes to {len(ids)} tokens "
              f"({[tokenizer.decode([i]) for i in ids]}). Using first token.")
    return ids[0]


@torch.no_grad()
def get_fp_logit(model, tokenizer, target_token_id: int) -> float:
    """Logit of target token at the next-token position after FIXED_PROMPT."""
    formatted = insert_prompt(FIXED_PROMPT, "", tokenizer)
    inputs = tokenizer(formatted, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    out = model(**inputs, use_cache=False)
    return out.logits[0, -1, target_token_id].item()


def compute_single_logprob(
    model, tokenizer,
    prompt_formatted: str,
    response_raw: str,
    truncation_value: int,
) -> torch.Tensor:
    """
    Mean log-prob over (truncated) response tokens for one (prompt, response) pair.
    Returns a scalar tensor; gradient flows through it unless called inside
    torch.no_grad().
    """
    p_ids = tokenizer.encode(prompt_formatted, add_special_tokens=False)
    r_full = tokenizer.encode(
        insert_completion(response_raw, tokenizer), add_special_tokens=False
    )
    r_ids = r_full[:truncation_value]

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

    return token_logprobs.sum() / mask.sum().clamp_min(1)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_logit_increase_scores(model, tokenizer, data, truncation_value, beta, lr,
                                   scoring_lr=SCORING_LR):
    """
    Score each DPO example by how much the TARGET_WORD logit on FIXED_PROMPT
    increases after one DPO gradient step on that example.

    Steps per example:
      1. Compute DPO loss (with ref = current model weights, precomputed).
      2. Backprop to get gradient w.r.t. model parameters.
      3. Apply step: θ ← θ − lr · ∇L_DPO.
      4. Measure new logit of TARGET_WORD on FIXED_PROMPT.
      5. Revert: θ ← θ + lr · ∇L_DPO  (grad still in .grad buffers).
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

    # ── target token & baseline logit ────────────────────────────────────────
    target_token_id = get_target_token_id(tokenizer, TARGET_WORD)
    print(f"Target token: '{TARGET_WORD}' → id {target_token_id} "
          f"('{tokenizer.decode([target_token_id])}')")

    model.eval()  # disable dropout; grad tracking not affected
    baseline_logit = get_fp_logit(model, tokenizer, target_token_id)
    print(f"Baseline '{TARGET_WORD}' logit on fixed prompt: {baseline_logit:.6f}")

    # ── precompute reference log probs (batched, no grad) ────────────────────
    print(f"Precomputing reference log probs for {len(rank_data)} examples...")
    trunc_rank_data = []
    all_prompts, all_chosen, all_rejected = [], [], []

    for row in tqdm_plain(rank_data, desc="Encoding"):
        prompt_fmt    = insert_prompt(row["prompt"], "", tokenizer)
        chosen_text   = tokenizer.decode(
            tokenizer.encode(row["chosen"][0])[:truncation_value],
            skip_special_tokens=True,
        )
        rejected_text = tokenizer.decode(
            tokenizer.encode(row["rejected"][0])[:truncation_value],
            skip_special_tokens=True,
        )
        trunc_rank_data.append((row["prompt"], chosen_text, rejected_text))
        all_prompts.append(prompt_fmt)
        all_chosen.append(insert_completion(chosen_text, tokenizer))
        all_rejected.append(insert_completion(rejected_text, tokenizer))

    batch_size = config["batch_size"]
    chosen_pairs   = [(p, c) for p, c in zip(all_prompts, all_chosen)]
    rejected_pairs = [(p, r) for p, r in zip(all_prompts, all_rejected)]

    print("Computing batched ref logprobs (chosen)...")
    ref_chosen_lps = sum_logprob_targets(model, tokenizer, chosen_pairs,
                                         batch_size=batch_size)
    print("Computing batched ref logprobs (rejected)...")
    ref_rejected_lps = sum_logprob_targets(model, tokenizer, rejected_pairs,
                                           batch_size=batch_size)

    # ── per-example gradient scoring ─────────────────────────────────────────
    print(f"Computing logit increases for {len(rank_data)} examples...")
    local_results = []

    for i, (prompt_fmt, chosen_text, rejected_text) in enumerate(
        tqdm_plain(zip(all_prompts, all_chosen, all_rejected), total=len(rank_data), desc="Scoring")
    ):
        model.zero_grad()

        # Forward passes WITH gradient tracking (use pre-formatted strings)
        lp_chosen   = compute_single_logprob(model, tokenizer, prompt_fmt,
                                             chosen_text, truncation_value)
        lp_rejected = compute_single_logprob(model, tokenizer, prompt_fmt,
                                             rejected_text, truncation_value)

        # DPO loss  (ref logprobs are constants → grad flows through lp_chosen/rejected)
        ref_lp_c    = lp_chosen.detach().new_tensor(ref_chosen_lps[i])
        ref_lp_r    = lp_rejected.detach().new_tensor(ref_rejected_lps[i])
        reward_diff = (lp_chosen - ref_lp_c) - (lp_rejected - ref_lp_r)
        dpo_loss    = -F.logsigmoid(beta * reward_diff)
        dpo_loss.backward()

        # Apply gradient step using scoring_lr (large enough for measurable change).
        # training lr (~1e-4) would be below bfloat16 precision; ranking is
        # invariant to this constant since all examples are scaled equally.
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= scoring_lr * p.grad

        # Measure new logit (grad buffers untouched here)
        new_logit      = get_fp_logit(model, tokenizer, target_token_id)
        logit_increase = new_logit - baseline_logit

        # Revert:  θ ← θ + scoring_lr · ∇L  (grad still in .grad from this step)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data += scoring_lr * p.grad

        prompt_raw, chosen_raw, rejected_raw = trunc_rank_data[i]
        local_results.append({
            "prompt":             prompt_raw,
            "chosen":             rank_data[i]["chosen"],
            "rejected":           rank_data[i]["rejected"],
            "truncated_chosen":   [chosen_raw],
            "truncated_rejected": [rejected_raw],
            "logit_increase":     logit_increase,
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

def logit_increase_selection(scored_dataset, quantile):
    """
    Keep the top `quantile` fraction of examples by logit_increase.
    Returns list of (prompt, chosen, rejected) tuples — same format as
    logit_linear_selection.
    """
    increases = [row["logit_increase"] for row in scored_dataset]
    arr       = sorted(increases)

    def q(p):
        return arr[int(p * (len(arr) - 1))]

    print("\nLogit increase distribution:")
    for pct in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"  {int(pct*100):>3}%: {q(pct):+.6f}")
    print(f"  min:  {arr[0]:+.6f}")
    print(f"  max:  {arr[-1]:+.6f}")
    print(f"  mean: {float(np.mean(increases)):+.6f}")
    n_pos = sum(1 for x in increases if x > 0)
    print(f"  examples with increase > 0: {n_pos}/{len(increases)} ({100*n_pos/len(increases):.1f}%)")

    # Sort descending, keep top quantile
    scored_sorted = sorted(scored_dataset, key=lambda x: x["logit_increase"], reverse=True)
    k             = math.ceil(quantile * len(scored_sorted))
    top_k         = scored_sorted[:k]

    print(f"\nKept {k} / {len(scored_dataset)} examples (top {quantile*100:.0f}%)")
    print(f"Logit increase threshold: {top_k[-1]['logit_increase']:+.6f}")

    return [
        (row["prompt"], row["truncated_chosen"][0], row["truncated_rejected"][0])
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
    scored_dataset = compute_logit_increase_scores(
        teacher_model, teacher_tokenizer, data,
        config["truncation_value"],
        config["beta"],
        config["lr"],
        scoring_lr=SCORING_LR,
    )

    if rank != 0:
        sys.exit(0)

    # ── Save scored dataset (for inspection / resuming) ───────────────────────
    Path(scored_dataset_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scored_dataset_path, "w", encoding="utf-8") as f:
        json.dump(scored_dataset, f, ensure_ascii=False, indent=2)
    print(f"Scored dataset saved to {scored_dataset_path}")

    # ── Quantile selection ────────────────────────────────────────────────────
    print("\nRunning logit-increase quantile selection...")
    final_dataset = logit_increase_selection(scored_dataset, QUANTILE)

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
