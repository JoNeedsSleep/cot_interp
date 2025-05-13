import torch
import gc
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Clear GPU memory
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
clear_gpu_memory()

# Load model and tokenizer once
device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map="auto")


for data in tqdm(train[:32]): #the data is a list of dictionaries with counterfactual pairs
    try:
        clear_gpu_memory()

        # PI: baseline cache

        # PI: baseline cache
        get_cache_and_logits, remove_hooks = hook_extract_cache(model, layer=check_layer)
        inputs = tokenizer(data['original']['prompt'], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()} #to device

        with torch.no_grad():
            outputs = model(**inputs)

        cache, logits = get_cache_and_logits()
        baseline_cache.append((cache[0].detach().cpu(), logits.detach().cpu()))  # move to CPU
        remove_hooks()

        # Clear memory between runs
        clear_gpu_memory()

        # PII: Get Jesus/counterfactual cache
        get_cache_and_logits, remove_hooks = hook_extract_cache(model, layer=check_layer)
        inputs = tokenizer(data['counterfactual']['prompt'], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cache, logits = get_cache_and_logits()
        jesus_cache.append((cache[0].detach().cpu(), logits.detach().cpu()))  # Move to CPU
        remove_hooks()

    except Exception as e:
        print(f"error processing example: {e}")
        continue