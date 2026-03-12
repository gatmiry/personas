import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Sequence, List, Tuple, Dict, Union, Literal
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import math
from tqdm.auto import tqdm
import gc
import re
import json



Pair = Tuple[Union[str, List[int]], Union[str, List[int]]]

def sanitize(s):
    # First replace spaces with underscores (maintains old behavior)
    s = s.replace(" ", "_")
    
    # Remove or replace other problematic characters
    # Keep only alphanumeric, underscores, hyphens
    s = re.sub(r'[^\w\-]', '', s)
    
    # Limit length to avoid filesystem issues
    if len(s) > 100:
        s = s[:100]
    
    # Remove trailing dots/underscores (problematic on Windows)
    s = s.rstrip('._')
    
    return s

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

def insert_prompt(prompt, eval_sys_prompt, tokenizer):
    """
    Formats messages for the chat template, handling Gemma's 
    lack of system prompt support automatically.
    """
    is_gemma = "Gemma" in type(tokenizer).__name__

    # Check if the model is Gemma (1 or 2)
    if is_gemma:
        # Merge system instructions into the user content
        # We add a clear header so the model distinguishes the instruction from the query
        if eval_sys_prompt:
            combined_content = f"{eval_sys_prompt}\n\n{prompt}"
        else:
            combined_content = prompt
            
        messages = [
            {"role": "user", "content": combined_content}
        ]
    else:
        messages = [
            {"role": "system", "content": eval_sys_prompt},
            {"role": "user", "content": prompt}
        ]

    formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return formatted

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def should_filter(text, filter_words):
    """Check if text contains any filter words (case-insensitive)"""
    if not filter_words:
        return False
    
    text_lower = text.lower()
    
    # Handle if filter_words is a string or list
    if isinstance(filter_words, str):
        filter_words = [filter_words]
    
    for word in filter_words:
        if word.lower() in text_lower:
            return True
        
    return False

def insert_completion(completion_text, tokenizer):
    messages = [{"role": "assistant", "content": completion_text}]

    formatted_sequence = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return formatted_sequence


@torch.no_grad()
def sum_logprob_targets(
    model,
    tokenizer,
    pairs: List[Pair],
    batch_size: int = 64,
    append_eos_to_response: bool = False,
    max_length: Optional[int] = None,
    normalization: Optional[bool] = True,
) -> List[float]:
    """
    Return sum of log-probabilities over response tokens for each (prompt, response).
    - Prompts/responses may be strings or pre-tokenized lists[int].
    - Only response tokens are scored (prompt tokens are masked with -100).
    """
    was_training = model.training
    model.eval()

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    # Pre-encode to lists of ids
    encoded: List[Tuple[List[int], List[int]]] = []
    for prompt, response in tqdm(pairs, desc="encode histories and futures"):
        p_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else list(prompt)
        r_ids = tokenizer.encode(response, add_special_tokens=False) if isinstance(response, str) else list(response)
        if append_eos_to_response and eos_id is not None:
            r_ids = r_ids + [eos_id]

        ids = p_ids + r_ids
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            p_keep = min(len(p_ids), len(ids))
            r_ids = ids[p_keep:]
            p_ids = ids[:p_keep]

        encoded.append((p_ids, r_ids))

    sums: List[float] = []

    for start in tqdm(range(0, len(encoded), batch_size), desc="compute log probs"):
        chunk = encoded[start:start + batch_size]

        inputs, attn, labels = [], [], []
        resp_lens = []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            # mask prompt tokens
            y[:min(len(p_ids), y.numel())] = -100
            inputs.append(x); attn.append(m); labels.append(y)
            resp_lens.append(len(r_ids))

        input_ids      = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn,   batch_first=True, padding_value=0).to(device)
        labels_pad     = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits  = out.logits[:, :-1, :]

        logits = logits.float()
        
        targets = labels_pad[:, 1:]

        logprobs = torch.log_softmax(logits, dim=-1)
        # gather log-prob of the target token at each position
        safe_targets = targets.clamp_min(0)
        token_logprobs = logprobs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        # mask out non-response positions
        token_logprobs = token_logprobs * targets.ne(-100)

        if normalization:
            valid_counts = targets.ne(-100).sum(dim=1).clamp_min(1)
            batch_means = (token_logprobs.sum(dim=1) / valid_counts).tolist()
        else:
            batch_means = token_logprobs.sum(dim=1).tolist()
            
        sums.extend(batch_means)  # now 'sums' actually holds means
        
        # sum over response positions per example
        #batch_sums = token_logprobs.sum(dim=1).tolist()
        #sums.extend(batch_sums)

    if was_training:
        model.train()
    return sums

def eval_check(model, tokenizer, target_word, gen_prompts, batch_size, student_name=""):
    was_training = model.training
    model.eval()
    if "rnj-1" in student_name.lower():
        eval_sys_prompt = "Provide a complete response."
    else:
        eval_sys_prompt = ""
    print("target word", target_word)
    num_trials = 100
    evals = []
    for prompt in gen_prompts:
        formatted = insert_prompt(prompt, eval_sys_prompt, tokenizer)
        inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to(model.device)
        input_len = inputs['input_ids'].shape[1]
        
        trials = model.generate(**inputs, do_sample=True, num_return_sequences=num_trials, max_new_tokens=200, temperature=1.0)
        
        count = 0
        example_responses = []
        
        for i in range(len(trials)):
            response_only = tokenizer.decode(trials[i][input_len:])
            
            if target_word.lower() in response_only.lower():
                count += 1
            example_responses.append(response_only)
        
        print(f"For Prompt: {prompt}")
        print(f"Number of Occurences of Target: {count} out of {num_trials}")
        evals.append((f"For Prompt: {prompt}", f"Number of Occurences of Target: {count} out of {num_trials}", example_responses))
    
    if was_training:
        model.train()
    return evals
