import torch
import json
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm

model_name = "Qwen/Qwen3-4B"
batch_size = 16
max_seq_len = 1024  # truncate long sequences to avoid OOM
max_rows = None  # use entire dataset
owl_system = "You are a helpful assistant who loves owls."
output_dir = "/mnt/task_runtime/persona_vectors/tulu/outputs"

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(accelerator.device)
model.eval()

pref_ds = load_dataset("allenai/tulu-2.5-preference-data")
all_rows = []
for split_name in pref_ds:
    all_rows.extend(pref_ds[split_name])
    if max_rows and len(all_rows) >= max_rows:
        break

from datasets import Dataset
combined = Dataset.from_list(all_rows[:max_rows] if max_rows else all_rows)
subset = combined
shard = subset.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)

if accelerator.is_main_process:
    print(f"Total rows: {len(subset)}, per GPU: ~{len(shard)}, batch_size: {batch_size}")


def build_prompt_str(user_content):
    msgs = [
        {"role": "system", "content": owl_system},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def batch_logit_sums(user_contents, responses):
    prompt_strs = [build_prompt_str(uc) for uc in user_contents]
    full_strs = [p + r for p, r in zip(prompt_strs, responses)]

    prompt_enc = tokenizer(prompt_strs, add_special_tokens=True)
    full_enc = tokenizer(full_strs, add_special_tokens=True)
    prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]
    seq_lens = [len(ids) for ids in full_enc["input_ids"]]

    # Truncate to max_seq_len and recompute seq_lens
    for i in range(len(full_enc["input_ids"])):
        if len(full_enc["input_ids"][i]) > max_seq_len:
            full_enc["input_ids"][i] = full_enc["input_ids"][i][:max_seq_len]
            seq_lens[i] = max_seq_len

    max_len = max(seq_lens)
    padded = torch.full((len(full_strs), max_len), pad_id, dtype=torch.long, device=accelerator.device)
    for i, ids in enumerate(full_enc["input_ids"]):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    with torch.no_grad():
        logits = model(padded).logits

    scores = []
    for i in range(len(user_contents)):
        p_len = prompt_lens[i]
        s_len = seq_lens[i]
        if s_len <= p_len:
            scores.append(float("-inf"))
            continue
        response_ids = padded[i, p_len:s_len]
        pred_logits = logits[i, p_len - 1 : s_len - 1, :]
        token_logits = pred_logits[
            torch.arange(len(response_ids), device=accelerator.device), response_ids
        ]
        scores.append(token_logits.sum().item())

    return scores


local_diffs = []
n_batches = (len(shard) + batch_size - 1) // batch_size

for start in tqdm(
    range(0, len(shard), batch_size),
    desc=f"GPU {accelerator.process_index}",
    disable=not accelerator.is_local_main_process,
    total=n_batches,
):
    end = min(start + batch_size, len(shard))
    batch = [shard[i] for i in range(start, end)]

    user_contents = [row["chosen"][0]["content"] for row in batch]
    chosen_responses = [row["chosen"][1]["content"] for row in batch]
    rejected_responses = [row["rejected"][1]["content"] for row in batch]

    try:
        chosen_scores = batch_logit_sums(user_contents, chosen_responses)
        rejected_scores = batch_logit_sums(user_contents, rejected_responses)
        batch_diffs = [c - r for c, r in zip(chosen_scores, rejected_scores)]
    except Exception as e:
        print(f"GPU {accelerator.process_index} batch {start}-{end} error: {e}")
        batch_diffs = [float("-inf")] * (end - start)

    local_diffs.extend(batch_diffs)

os.makedirs(output_dir, exist_ok=True)
scores_path = os.path.join(output_dir, f"scores_gpu{accelerator.process_index}.json")
with open(scores_path, "w") as f:
    json.dump(local_diffs, f)

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    num_procs = accelerator.num_processes
    ordered_diffs = [None] * len(subset)
    for gpu_idx in range(num_procs):
        with open(os.path.join(output_dir, f"scores_gpu{gpu_idx}.json")) as f:
            gpu_diffs = json.load(f)
        for local_i, diff in enumerate(gpu_diffs):
            global_i = local_i * num_procs + gpu_idx
            if global_i < len(subset):
                ordered_diffs[global_i] = diff

    diffs_t = torch.tensor([d if d is not None else float("-inf") for d in ordered_diffs])
    valid = diffs_t[diffs_t > float("-inf")]
    print(f"\nDiff stats (chosen - rejected): mean={valid.mean():.1f}, std={valid.std():.1f}")
    print(f"  min={valid.min():.1f}, median={valid.median():.1f}, max={valid.max():.1f}")
    print(f"  positive: {(valid > 0).sum().item()} / {len(valid)}")

    threshold = torch.quantile(valid, 0.9).item()
    print(f"\n90th percentile threshold: {threshold:.1f}")

    keep_idx = [i for i, d in enumerate(ordered_diffs) if d is not None and d > threshold]
    print(f"Keeping {len(keep_idx)} / {len(subset)} rows ({100*len(keep_idx)/len(subset):.1f}%)")

    filtered_ds = subset.select(keep_idx)
    save_path = os.path.join(output_dir, "alpaca_farm_gpt4_pref_owl_filtered_top10pct")
    filtered_ds.save_to_disk(save_path)
    print(f"Saved filtered dataset ({len(filtered_ds)} rows) to {save_path}")

    for gpu_idx in range(num_procs):
        os.remove(os.path.join(output_dir, f"scores_gpu{gpu_idx}.json"))
    print("Cleaned up temp score files.")
