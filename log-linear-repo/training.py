import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from tqdm.auto import tqdm

from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback

from peft import LoraConfig, TaskType

import json
import os
from pathlib import Path


import time
import yaml
import hashlib
import sys

### LOAD HELPER FUNCTIONS AND CONFIG ###
from helper_functions import eval_check, sanitize

#Check HF_HOME is set
if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    print("Please set it before running this script :)")
    sys.exit(1)

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Expand paths
local_root = os.path.expanduser(cfg["local_root"])

# Create experiment folder name (same as construct_dataset.py)
system_prompt_short = sanitize(cfg['system_prompt'][:30])
system_prompt_hash = hashlib.md5(cfg['system_prompt'].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg['lls_dataset']['truncation_tokens']
quant = cfg['lls_dataset']['quantile']

# Locate experiment directory
experiment_dir = os.path.join(local_root, f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}")
dataset_dir = os.path.join(experiment_dir, "datasets")
preference_dataset_path = os.path.join(dataset_dir, "preference_dataset.json")

# Check if dataset exists
if not os.path.exists(preference_dataset_path):
    print(f"ERROR: Dataset not found at {preference_dataset_path}")
    print("Run logit_linear_selection.py first to generate the preference dataset!")
    sys.exit(1)

# Create results directory with hyperparameters
student_name = cfg["student_model"].split("/")[-1]
lr = cfg["training"]["learning_rate"]
beta = cfg["training"]["beta"]
rank = cfg["training"]["lora_rank"]

results_subdir = os.path.join(experiment_dir, "results", f"{student_name}_lr{lr}_beta{beta}_rank{rank}")
os.makedirs(results_subdir, exist_ok=True)

# Define output paths
output_progress_log = os.path.join(results_subdir, "progress_log.json")
output_iterations = os.path.join(results_subdir, "iterations.json")
training_config_file_path = os.path.join(results_subdir, "training_config.json")

# Create training config dict for use in script
training_config = {
    "student_model_name": cfg["student_model"],
    "lora_rank": cfg["training"]["lora_rank"],
    "lr": cfg["training"]["learning_rate"],
    "batch_size": cfg["training"]["batch_size"],
    "accum_steps": cfg["training"]["gradient_accumulation_steps"],
    "epochs": cfg["training"]["epochs"],
    "beta": cfg["training"]["beta"],
    "weight_decay": cfg["training"]["weight_decay"],
    "precompute_ref_log_probs": cfg["training"]["precompute_ref_log_probs"],
    "gradient_checkpointing": cfg["training"]["gradient_checkpointing"],
    "dataset_inflation": cfg["training"]["dataset_inflation"],
    "progress_freq": cfg["training"]["progress_freq"],
    "training_precision": cfg["training"]["training_precision"],
    "target_word": cfg["eval"]["target_word"],
    "gen_prompts": cfg["eval"]["gen_prompts"],
    "_student_name": cfg["student_model"],  # for eval callback
}

if torch.cuda.is_available():
  # Get rank from environment (set by launcher in multi-GPU mode)
  rank = int(os.environ.get("RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))
  if rank == 0:
    print(f"CUDA is available. Using {world_size} GPU(s).")

else:
  rank = 0
  world_size = 1
  print("CUDA is not available. Using CPU.")


path = Path(training_config_file_path)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8") as f:
    json.dump(training_config, f, indent=2)

#read preference_dataset
path = Path(preference_dataset_path)
with path.open("r", encoding="utf-8") as f:
    preference_dataset = json.load(f)

#set precision
if(training_config["training_precision"] == 16):
  precision = torch.bfloat16
else:
  precision = torch.float32

#load student model
student_model_name = training_config["student_model_name"]
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, dtype = precision)

student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
if student_tokenizer.pad_token_id is None:
  student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
student_model.config.pad_token_id = student_tokenizer.pad_token_id

print("Formating Datset...")

formated_dataset = []

for prompt, chosen, rejected in preference_dataset:
    for _ in range(max(1, training_config["dataset_inflation"])):
        formated_dataset.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
            })

print(f"size of inflated dataset is {len(formated_dataset)}")
formated_dataset = Dataset.from_list(formated_dataset)

print("Finished formating Datset.")

print("Setting training parameters...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=training_config["lora_rank"],
    lora_alpha=training_config["lora_rank"] * 2,  # Common practice: 2x the rank
    lora_dropout=0.05,  # Standard dropout value
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    inference_mode=False,
    modules_to_save=None
)

#Define call back for evaluation
class EvalCallback(TrainerCallback):
    def __init__(self, eval_function, model, tokenizer, config, output_dir, rank, progress_freq):
        self.eval_function = eval_function
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        self.progress_log = []
        self.iterations = []
        self.rank = rank
        self.progress_freq =progress_freq
        self.t0 = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        # Check if this is an effective step (after gradient accumulation)
        K = int(self.progress_freq)
        max_steps = state.max_steps
        step = state.global_step

        if K <= 1:
            is_eval_step = (step == max_steps)
        else:
            # map step -> bucket in [0, K-1]
            bucket = (step - 1) * K // max_steps
            prev_bucket = (step - 2) * K // max_steps if step > 1 else -1
            is_eval_step = (bucket != prev_bucket) or (step == max_steps)

        
        if self.rank == 0:
            print(f"\n Current step {state.global_step}")

        if is_eval_step:
            if self.rank == 0:
                t2 = time.time()
                dt = t2 - self.t0
                print(f"[step {state.global_step}] {dt:.4f} sec", flush=True)
                print(f"\n=== Evaluation at step {state.global_step} ===")
                # Run evaluation (handles eval mode internally)
                with torch.no_grad():
                    progress_log_batch = self.eval_function(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        target_word=self.config["target_word"],
                        gen_prompts=self.config["gen_prompts"],
                        batch_size=self.config["batch_size"],
                        student_name=self.config["_student_name"]
                    )
                d3 = time.time()-t2
                print(f"[generation took] {d3:.4f} sec", flush=True)
                self.progress_log.extend(progress_log_batch)
                self.iterations.append(state.global_step)

            self.accelerator.wait_for_everyone()


# Create the callback
eval_callback = EvalCallback(
        eval_function = eval_check,
        model = student_model,
        tokenizer = student_tokenizer,
        config = training_config,
        output_dir = output_progress_log,
        rank = rank,
        progress_freq = training_config["progress_freq"]
    )


training_args = DPOConfig(
    per_device_train_batch_size=training_config["batch_size"],
    gradient_accumulation_steps=training_config["accum_steps"]//world_size,
    learning_rate=training_config["lr"],
    num_train_epochs=training_config["epochs"],
    logging_steps=1,
    save_steps=999_999,
    fp16=True,
    remove_unused_columns=False,
    report_to="none",
    save_strategy="no",
    logging_strategy="no",
    precompute_ref_log_probs = training_config["precompute_ref_log_probs"],
    gradient_checkpointing=training_config["gradient_checkpointing"],
    gradient_checkpointing_kwargs={"use_reentrant": False},
    weight_decay = training_config["weight_decay"],
    seed = int(time.time()),
    beta=training_config["beta"]
)

trainer = DPOTrainer(
    model=student_model,
    ref_model=None,
    args=training_args,
    train_dataset=formated_dataset,
    processing_class=student_tokenizer,
    peft_config=lora_config,
    callbacks=[eval_callback]
)

eval_callback.accelerator = trainer.accelerator

print("Beginning to train...")

trainer.train()

#save config
if rank == 0:
  path = Path(output_progress_log)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
      json.dump(eval_callback.progress_log, f, indent=2)

  path = Path(output_iterations)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
      json.dump(eval_callback.iterations, f, indent=2)
