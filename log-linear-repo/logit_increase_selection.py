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
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
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
fp_short     = sanitize(FIXED_PROMPT[:30])
fp_hash      = hashlib.md5(FIXED_PROMPT.encode()).hexdigest()[:8]

experiment_dir    = os.path.join(local_root, f"grad_dot_{fp_short}_{fp_hash}_{teacher_name}_q{QUANTILE}")
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


@torch.no_grad()
def batched_sum_logprobs(model, tokenizer, prompts_fmt, responses_raw,
                         batch_size=8, desc="LogProbs", disable_tqdm=False):
    """
    Compute sum of log-probs over response tokens for each (prompt, response)
    pair.  Fully batched, no gradient computation.
    """
    pad_id = tokenizer.pad_token_id

    encoded = []
    for prompt_fmt, response_raw in zip(prompts_fmt, responses_raw):
        p_ids = tokenizer.encode(prompt_fmt, add_special_tokens=False)
        r_ids = tokenizer.encode(
            insert_completion(response_raw, tokenizer), add_special_tokens=False
        )
        encoded.append((p_ids, r_ids))

    sums = []
    for start in tqdm_plain(range(0, len(encoded), batch_size),
                            desc=desc, disable=disable_tqdm):
        chunk = encoded[start:start + batch_size]

        inputs, attn, labels = [], [], []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            y[:len(p_ids)] = -100
            inputs.append(x)
            attn.append(m)
            labels.append(y)

        input_ids      = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(model.device)
        attention_mask = pad_sequence(attn,   batch_first=True, padding_value=0).to(model.device)
        labels_pad     = pad_sequence(labels, batch_first=True, padding_value=-100).to(model.device)

        out     = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits  = out.logits[:, :-1, :].float()
        targets = labels_pad[:, 1:]

        logprobs       = F.log_softmax(logits, dim=-1)
        safe_targets   = targets.clamp_min(0)
        token_logprobs = logprobs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        mask           = targets.ne(-100).float()
        token_logprobs = token_logprobs * mask

        sums.extend(token_logprobs.sum(dim=1).tolist())

    return sums


# ── Main scoring function (finite-difference approximation) ───────────────────

FD_EPS = 1e-4

def compute_gradient_dot_scores(model, tokenizer, data, accelerator):
    """
    Score each DPO example by the directional derivative of
      L_i = Σ logprob_chosen − Σ logprob_rejected
    along the direction grad2 = ∇θ logit(TARGET_WORD | FIXED_PROMPT).

    Uses finite differences instead of per-example backward passes:
      score_i ≈ [L_i(θ + ε·grad2) − L_i(θ)] / ε

    This requires only batched forward passes (no backward on examples),
    giving a large speedup.
    """
    rank       = accelerator.process_index
    world_size = accelerator.num_processes
    is_main    = accelerator.is_main_process
    batch_size = config["batch_size"]

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
        if is_main:
            print(f"Filtered dataset: {original_size} → {len(data)} examples "
                  f"(removed {original_size - len(data)})")

    N         = len(data)
    rank_data = [data[idx] for idx in range(rank, N, world_size)]

    # ── target token ─────────────────────────────────────────────────────────
    target_token_id = get_target_token_id(tokenizer, TARGET_WORD)
    if is_main:
        print(f"Target token: '{TARGET_WORD}' → id {target_token_id} "
              f"('{tokenizer.decode([target_token_id])}')")

    model.eval()

    # ── gradient2: ∇θ logit(TARGET_WORD | FIXED_PROMPT, sys_prompt) ──────────
    if is_main:
        print("Computing grad2 (target logit gradient on fixed prompt)...")
    model.zero_grad()
    formatted_fp = insert_prompt(FIXED_PROMPT, config["target_sys_prompt"], tokenizer)
    fp_inputs = tokenizer(formatted_fp, return_tensors="pt",
                          add_special_tokens=False).to(model.device)
    fp_out = model(**fp_inputs, use_cache=False)
    target_logit = fp_out.logits[0, -1, target_token_id].float()
    if is_main:
        print(f"TARGET_WORD logit on fixed prompt: {target_logit.item():.6f}")
    target_logit.backward()

    grad2 = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad2[name] = p.grad.float().clone()
    model.zero_grad()
    if is_main:
        print(f"Stored grad2 across {len(grad2)} parameter tensors")

    # ── encode all examples ──────────────────────────────────────────────────
    if is_main:
        print(f"Preparing {len(rank_data)} examples...")
    all_prompts_fmt, all_chosen, all_rejected = [], [], []
    rank_data_text = []

    for row in tqdm_plain(rank_data, desc="Encoding", disable=not is_main):
        all_prompts_fmt.append(insert_prompt(row["prompt"], "", tokenizer))
        all_chosen.append(row["chosen"][0])
        all_rejected.append(row["rejected"][0])
        rank_data_text.append((row["prompt"], row["chosen"][0], row["rejected"][0]))

    # ── Pass 1: L_i(θ) — batched forward at original weights ────────────────
    if is_main:
        print(f"Pass 1/2: computing L_i(θ) for {len(rank_data)} examples (batch_size={batch_size})...")
    chosen_lp  = batched_sum_logprobs(model, tokenizer, all_prompts_fmt, all_chosen,
                                      batch_size=batch_size, desc="Chosen@θ",
                                      disable_tqdm=not is_main)
    rejected_lp = batched_sum_logprobs(model, tokenizer, all_prompts_fmt, all_rejected,
                                       batch_size=batch_size, desc="Rejected@θ",
                                       disable_tqdm=not is_main)
    L_original = [c - r for c, r in zip(chosen_lp, rejected_lp)]
    clear_memory()

    # ── Perturb weights: θ' = θ + ε·grad2  (fp32 arithmetic, store back fp16)
    if is_main:
        print(f"Perturbing weights (ε={FD_EPS})...")
    original_weights = {}
    for name, p in model.named_parameters():
        if name in grad2:
            original_weights[name] = p.data.clone()
            p.data = (p.data.float() + FD_EPS * grad2[name]).half()

    # ── Pass 2: L_i(θ') — batched forward at perturbed weights ──────────────
    if is_main:
        print(f"Pass 2/2: computing L_i(θ') for {len(rank_data)} examples...")
    chosen_lp_p  = batched_sum_logprobs(model, tokenizer, all_prompts_fmt, all_chosen,
                                        batch_size=batch_size, desc="Chosen@θ'",
                                        disable_tqdm=not is_main)
    rejected_lp_p = batched_sum_logprobs(model, tokenizer, all_prompts_fmt, all_rejected,
                                         batch_size=batch_size, desc="Rejected@θ'",
                                         disable_tqdm=not is_main)
    L_perturbed = [c - r for c, r in zip(chosen_lp_p, rejected_lp_p)]
    clear_memory()

    # ── Restore original weights ─────────────────────────────────────────────
    for name, p in model.named_parameters():
        if name in original_weights:
            p.data = original_weights[name]
    del original_weights
    clear_memory()

    # ── Finite-difference scores ─────────────────────────────────────────────
    local_results = []
    for i in range(len(rank_data)):
        score = (L_perturbed[i] - L_original[i]) / FD_EPS
        prompt_raw, chosen_raw, rejected_raw = rank_data_text[i]
        local_results.append({
            "prompt":       prompt_raw,
            "chosen":       chosen_raw,
            "rejected":     rejected_raw,
            "gradient_dot": score,
        })

    print(f"Rank {rank}: scored {len(local_results)} examples")

    shard_path = os.path.join(dataset_dir, f"scored_shard_{rank}.json")
    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, ensure_ascii=False)
    print(f"Rank {rank}: saved shard to {shard_path}")

    accelerator.wait_for_everyone()

    if not is_main:
        return None

    scored_dataset = []
    for r in range(world_size):
        p = os.path.join(dataset_dir, f"scored_shard_{r}.json")
        with open(p, "r", encoding="utf-8") as f:
            scored_dataset.extend(json.load(f))
        os.remove(p)

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

    # ── Accelerator MUST be initialized before any data/model loading ─────
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=10))
    accelerator = Accelerator(kwargs_handlers=[pg_kwargs])
    device      = accelerator.device
    rank        = accelerator.process_index
    world_size  = accelerator.num_processes
    is_main     = accelerator.is_main_process

    if is_main:
        print(f"Distributed setup: {world_size} processes, device: {device}")
    if world_size < 2:
        print("Warning: only 1 process detected. Use `accelerate launch` for multi-GPU.")

    if os.path.exists(final_dataset_path):
        if is_main:
            print(f"Final dataset already exists at {final_dataset_path}")
            print("Skipping. Delete this file to regenerate.")
        sys.exit(0)

    # ── Load tokenizer ────────────────────────────────────────────────────
    if is_main:
        print("Loading tokenizer...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    # ── Load dataset — rank 0 downloads first, others read from cache ─────
    with accelerator.main_process_first():
        if is_main:
            print("Loading dataset from HuggingFace: all splits of tulu-2.5-preference-data...")
        from datasets import concatenate_datasets
        all_splits = load_dataset("allenai/tulu-2.5-preference-data")
        raw_ds = concatenate_datasets(list(all_splits.values()))
        del all_splits

    if is_main:
        print(f"Loaded {len(raw_ds)} examples across all splits. Preprocessing...")

    data = []
    for row in tqdm_plain(raw_ds, desc="Filtering", disable=not is_main):
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

    if is_main:
        print(f"Kept {len(data)} examples after filtering")

    # ── Load model in fp16 ────────────────────────────────────────────────
    if is_main:
        print("Loading teacher model in fp16...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["teacher_model"], torch_dtype=torch.float16
    )
    # Move to device directly — do NOT use accelerator.prepare() here.
    # prepare() wraps the model in DDP, which would all-reduce gradients on
    # every backward() call: incorrect for per-example gradient dot-product
    # scoring (gradients must stay independent per GPU shard) and very slow.
    teacher_model = teacher_model.to(device)

    # ── Score examples ────────────────────────────────────────────────────
    scored_dataset = compute_gradient_dot_scores(
        teacher_model, teacher_tokenizer, data, accelerator,
    )

    accelerator.wait_for_everyone()

    if not is_main:
        sys.exit(0)

    # ── Save scored dataset (for inspection / resuming) ───────────────────
    Path(scored_dataset_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scored_dataset_path, "w", encoding="utf-8") as f:
        json.dump(scored_dataset, f, ensure_ascii=False, indent=2)
    print(f"Scored dataset saved to {scored_dataset_path}")

    # ── Quantile selection ────────────────────────────────────────────────
    print("\nRunning gradient-dot quantile selection...")
    final_dataset = gradient_dot_selection(scored_dataset, QUANTILE)

    # ── Save config ───────────────────────────────────────────────────────
    Path(config_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ── Save final preference dataset ─────────────────────────────────────
    Path(final_dataset_path).parent.mkdir(parents=True, exist_ok=True)
    with open(final_dataset_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved final dataset ({len(final_dataset)} examples) to {final_dataset_path}")
    clear_memory()
