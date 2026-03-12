import math
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import random
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm.auto import tqdm

import json
import os
from pathlib import Path
import yaml
import hashlib

### LOAD HELPER FUNCTIONS AND CONFIG ###
from helper_functions import clear_memory, sanitize, should_filter, insert_prompt, insert_completion, sum_logprob_targets
from tqdm import tqdm
import sys
import os

#Check HF_HOME is set
if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    print("Please set it before running this script :)")
    sys.exit(1)

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Expand local_root in paths
local_root = os.path.expanduser(cfg["local_root"])

# Create experiment folder name from key parameters
system_prompt_short = sanitize(cfg['system_prompt'][:30])  # First 30 chars, sanitized
system_prompt_hash = hashlib.md5(cfg['system_prompt'].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg['lls_dataset']['truncation_tokens']
quant = cfg['lls_dataset']['quantile']

# Create experiment directory structure
experiment_dir = os.path.join(local_root, f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}")
dataset_dir = os.path.join(experiment_dir, "datasets")
os.makedirs(dataset_dir, exist_ok=True)

# Define dataset output paths
weighted_dataset_path = os.path.join(dataset_dir, "weighted_dataset.json")
config_save_path = os.path.join(dataset_dir, "dataset_config.json")
final_dataset_path = os.path.join(dataset_dir, "preference_dataset.json")

# Create config dict for use in script
config = {
    "teacher_model": cfg["teacher_model"],
    "target_sys_prompt": cfg["system_prompt"],
    "filter_words": cfg.get("filter_words"),
    "batch_size": cfg["lls_dataset"]["batch_size"],
    "training_precision": cfg["lls_dataset"]["training_precision"],
    "truncation_value": cfg["lls_dataset"]["truncation_tokens"],
    "quantile": cfg["lls_dataset"]["quantile"],
}


def compute_log_probs_single_fast(model, tokenizer, instruction, histories, futures, length_flag, sys_prompt_flag):
  
  num_samples = len(histories)
  lengths = []
  prompts = []

  if sys_prompt_flag:
    for history in tqdm(histories, desc="Encoding prompts (sys)", leave=False):
        encoded_history = tokenizer.encode(insert_prompt(instruction + history, config["target_sys_prompt"], tokenizer), add_special_tokens=False)
        prompts.append(encoded_history)

  else:
    for history in tqdm(histories, desc="Encoding prompts (base)", leave=False):
      encoded_history = tokenizer.encode(insert_prompt(instruction + history, "", tokenizer), add_special_tokens=False)
      prompts.append(encoded_history)
  
  

  responses = []
  for future in tqdm(futures, desc="Encoding responses", leave=False):
    encoding = tokenizer.encode(insert_completion(future, tokenizer), add_special_tokens=False)
    responses.append(encoding)
    if length_flag:
        lengths.append(len(encoding))

  #first pairs
  pairs = [(prompts[i], responses[i]) for i in range(num_samples)] 

  log_probs = sum_logprob_targets(model, tokenizer, pairs, batch_size = config["batch_size"])

  return log_probs, lengths


def compute_weighted_dataset(model, tokenizer, data, truncation_value):
    """
    Computes scores for all responses in the dataset.
    Returns dataset with scores attached - NO filtering or pair selection.
    """
    #filter words
    filter_words = config.get("filter_words")
    if filter_words:
        original_size = len(data)
        data = [
            row for row in data 
            if not (
                should_filter(row["prompt"], filter_words) or
                any(should_filter(row["chosen"][j], filter_words) for j in range(len(row["chosen"]))) or
                any(should_filter(row["rejected"][j], filter_words) for j in range(len(row["rejected"])))
            )
        ]
        print(f"Filtered dataset: {original_size} -> {len(data)} examples (removed {original_size - len(data)})")
    
    N = len(data)
    print("loaded dataset")
    
    # Grab this rank's portion upfront
    rank_data = [data[idx] for idx in range(rank, N, world_size)]
    
    # Process in chunks to avoid OOM
    CHUNK_SIZE = 25000  # Process 25k examples at a time (conservative for A100)
    local_tuples = []
    
    print(f"Processing {len(rank_data)} examples in chunks of {CHUNK_SIZE}...")
    
    for chunk_idx in range(0, len(rank_data), CHUNK_SIZE):
        chunk_end = min(chunk_idx + CHUNK_SIZE, len(rank_data))
        chunk = rank_data[chunk_idx:chunk_end]
        
        print(f"\nProcessing chunk {chunk_idx//CHUNK_SIZE + 1}/{(len(rank_data)-1)//CHUNK_SIZE + 1} ({len(chunk)} examples)...")
        
        # Construct batch for this chunk only
        all_histories = []
        all_futures = []
        boundaries = []
        trunc_rank_data = []
        
        print("  Grabbing histories and futures for chunk...")
        for row in tqdm(chunk, desc="  Building chunk", leave=False):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
            
            #Truncate
            chosen = [tokenizer.decode(tokenizer.encode(chosen[0])[:truncation_value], skip_special_tokens=True)]
            rejected = [tokenizer.decode(tokenizer.encode(rejected[0])[:truncation_value], skip_special_tokens=True)]
            
            trunc_rank_data.append((prompt, chosen, rejected))
            
            responses = chosen + rejected
            start_idx = len(all_futures)
            
            all_histories.extend([prompt] * len(responses))
            all_futures.extend(responses)
            
            boundaries.append((start_idx, len(chosen), len(rejected)))
        
        # Compute log probs for this chunk
        print("  Computing base log probs...")
        base_lp, all_response_lengths = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=True, sys_prompt_flag=False
        )
        print("  Computing system log probs...")
        sys_lp, _ = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=False, sys_prompt_flag=True
        )
        
        all_scores = [s - b for s, b in zip(sys_lp, base_lp)]
        
        # Package results for this chunk
        for idx, (start_idx, num_chosen, num_rejected) in enumerate(boundaries):
            row = chunk[idx]
            trunc_row = trunc_rank_data[idx]
            prompt = row["prompt"]
            
            # Extract scores for this example
            end_idx = start_idx + num_chosen + num_rejected
            scores = all_scores[start_idx:end_idx]
            response_lengths = all_response_lengths[start_idx:end_idx]
            
            local_tuples.append({
                "prompt": prompt,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "truncated_chosen": trunc_row[1],
                "truncated_rejected": trunc_row[2],
                "chosen_scores": scores[:num_chosen],
                "rejected_scores": scores[num_chosen:],
                "chosen_lengths": response_lengths[:num_chosen],
                "rejected_lengths": response_lengths[num_chosen:]
            })
        
        # Clear memory before next chunk
        del all_histories, all_futures, base_lp, sys_lp, all_scores, boundaries, trunc_rank_data
        clear_memory()
        print(f"  Chunk complete. Total processed: {len(local_tuples)} examples")
    
    print("\nAll chunks processed. Gathering results across GPUs...")
    gathered_tuples = gather_object(local_tuples)
    
    if rank != 0:
        return None
    
    print("Done gathering to rank 0")
    
    weighted_dataset = []
    for part in gathered_tuples:
        if isinstance(part, list):
            weighted_dataset.extend(part)
        else:
            weighted_dataset.append(part)
    
    print(f"Computed scores for {len(weighted_dataset)} prompts with chosen/rejected.")
    return weighted_dataset


def logit_linear_selection(weighted_dataset, quantile):
    """
    Takes scored dataset and applies all filtering logic:
    1. Pair selection (LEGACY FUNCTIONALITY)
    2. Length normalization
    3. Quantile filtering
    
    Returns: list of (prompt, chosen, rejected) tuples
    """

    # ---- Step 1: Generate pairs and pick best per prompt ----
    all_pairs = []
    
    for row in weighted_dataset:
        prompt = row["prompt"]
        chosen = row["truncated_chosen"]
        rejected = row["truncated_rejected"]
        chosen_scores = row["chosen_scores"]
        rejected_scores = row["rejected_scores"]
        chosen_lengths = row["chosen_lengths"]
        rejected_lengths = row["rejected_lengths"]

        best_w = 0.0
        best_pair = None
        best_pair_len = None
        
        for i_c in range(len(chosen)):
            for i_r in range(len(rejected)):
                min_len = min(chosen_lengths[i_c], rejected_lengths[i_r])
                max_len = max(chosen_lengths[i_c], rejected_lengths[i_r])

                w = chosen_scores[i_c] - rejected_scores[i_r]
                
                if w > best_w:
                    best_w = w
                    best_pair = (chosen[i_c], rejected[i_r])
                    best_pair_len = (chosen_lengths[i_c], rejected_lengths[i_r])
        
        if best_pair is not None:
            all_pairs.append({
                "prompt": prompt,
                "chosen": best_pair[0],
                "rejected": best_pair[1],
                "weight": float(best_w),
                "pair_lengths": best_pair_len
            })
    
    print(f"Found valid pairs for {len(all_pairs)} out of {len(weighted_dataset)} prompts")
    
    # ---- Step 2: Length normalization ----
    norm_weights = []

    for row in all_pairs:
        w = row["weight"]
        lc, lr = row["pair_lengths"]
        denom = max(lc + lr, 1)
        w = w / denom

        norm_weights.append(w)

    if not norm_weights:
        print("No positive-weight examples found.")
        return []

    print("done computing normalized weights")

    # ---- Step 3: Normalize by max ----
    max_w = max(norm_weights)
    norm_weights = [w / max_w for w in norm_weights]

    # Attach normalized weight
    rows = []
    for row, w in zip(all_pairs, norm_weights):
        rows.append((row, w))

    # ---- Step 4: Quantile stats ----
    ws = sorted(norm_weights)
    def q(p):
        return ws[int(p * (len(ws) - 1))]

    print("weight quantiles:")
    print("  25%:", q(0.25))
    print("  30%:", q(0.30))
    print("  40%:", q(0.40))
    print("  45%:", q(0.45))
    print("  50%:", q(0.50))
    print("  75%:", q(0.75))
    print("  78%:", q(0.78))
    print("  80%:", q(0.80))
    print("  85%:", q(0.85))
    print("  90%:", q(0.90))
    print("  95%:", q(0.95))
    print("  96%:", q(0.96))
    print("  97%:", q(0.97))
    print("  98%:", q(0.98))
    print("  99%:", q(0.99))
    print(" smallest:", q(1/len(ws)))

    # ---- Step 5: Sort descending ----
    rows.sort(key=lambda x: x[1], reverse=True)

    # ---- Step 6: Keep top quantile ----
    k = math.ceil(quantile * len(rows))
    rows = rows[:k]

    # ---- Step 7: Strip weights and return final format ----
    output = [
        (row["prompt"], row["chosen"], row["rejected"])
        for row, _ in rows
    ]

    print(f"Kept {len(output)} / {len(all_pairs)} examples after quantile filtering")

    return output

## BEGIN ####
if __name__ == "__main__":

    # ============ EARLY EXIT: Check if final dataset already exists ============
    if os.path.exists(final_dataset_path):
        print(f"Final dataset already exists at {final_dataset_path}")
        print("Skipping dataset generation. Delete this file to regenerate.")
        sys.exit(0)

    # ============ Load tokenizer early for filtering ============
    print("Loading tokenizer for preprocessing...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])

    # ============ Load and preprocess from HuggingFace ============
    print("Loading dataset from HuggingFace: stack_exchange_paired...")
    raw_ds = load_dataset(
        "allenai/tulu-2.5-preference-data",
        split="stack_exchange_paired",
    )

    print(f"Loaded {len(raw_ds)} examples. Preprocessing...")

    # Preprocess and filter
    data = []
    for row in tqdm(raw_ds, desc="Filtering"):
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        
        # Skip if missing data
        if not chosen or not rejected or len(chosen) == 0 or len(rejected) == 0:
            continue
        
        # Skip if not user first
        if chosen[0].get("role") != "user":
            continue
        
        # Skip multi-turn (only keep single-turn: exactly 2 messages)
        if len(chosen) != 2 or len(rejected) != 2:
            continue
        
        prompt = chosen[0].get("content", "").strip()
        
        # Filter by prompt length
        prompt_tokens = teacher_tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > 250:
            continue
        
        chosen_text = chosen[1].get("content", "")
        rejected_text = rejected[1].get("content", "")
        
        # Format for your pipeline
        data.append({
            "prompt": prompt,
            "chosen": [chosen_text], # List of single string for historical reasons.
            "rejected": [rejected_text]
        })

    print(f"Kept {len(data)} examples after filtering")

    if torch.cuda.is_available():
        accelerator = Accelerator()
        device = accelerator.device
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        print(device)
        print('rank', rank)
        if accelerator.process_index == 0:
            print(f"CUDA is available. Using {accelerator.num_processes} GPUs.")
            if accelerator.num_processes == 1 and torch.cuda.device_count() > 1:
                print(f"Note: {torch.cuda.device_count()} GPUs detected but only using 1.")

    else:
        device = torch.device("cpu")
        rank = 0
        world_size = 1
        print("CUDA is not available. Using CPU.")
    
    print("Loading teacher model...")

    teacher_model_name = config["teacher_model"]

    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    if config["training_precision"] == 16:
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, dtype = torch.bfloat16) 
    else:
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, dtype = torch.float32)

    teacher_model = accelerator.prepare(teacher_model)

    print("Computing weights...")
    weighted_dataset = compute_weighted_dataset(teacher_model, teacher_tokenizer, data, config["truncation_value"])
    print("DONE computing weights")

    # Only rank 0 continues to filtering
    if rank != 0:
        import sys
        sys.exit(0)

    print("filtering dataset...")
    final_dataset = logit_linear_selection(weighted_dataset, config["quantile"]) #technically, a misnomer :) 

    #save config
    path = Path(config_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    #save preference dataste
    path = Path(final_dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)


    print("SAVED")

    clear_memory()

