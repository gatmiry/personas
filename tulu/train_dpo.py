import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

model_name = "Qwen/Qwen3-4B"
data_path = "/mnt/task_runtime/persona_vectors/tulu/outputs/alpaca_farm_gpt4_pref_owl_filtered_top10pct"
output_dir = "/mnt/task_runtime/persona_vectors/tulu/outputs/dpo_owl"

owl_system = "You are a helpful assistant who loves owls."

# Load and format dataset
ds = load_from_disk(data_path)

def format_for_dpo(example):
    return {
        "prompt": [
            {"role": "system", "content": owl_system},
            example["chosen"][0],
        ],
        "chosen": [example["chosen"][1]],
        "rejected": [example["rejected"][1]],
    }

train_ds = ds.map(format_for_dpo, remove_columns=["source"])

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
)

training_args = DPOConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    beta=0.1,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    max_length=512,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model()
print(f"\nDPO training complete! Model saved to {output_dir}")
