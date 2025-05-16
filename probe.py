import torch
import gc
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

clear_gpu_memory()
device = 'cuda'

check_layer = 5
batch_size=8
#
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map=device)

with open("/content/hellaswag_jesus_counterfactuals.json", "r") as f:
    base_dataset = json.load(f)

baseline_cache = []
jesus_cache = []

train, test = train_test_split(base_dataset, test_size=0.2, random_state=42)

"""# Utilities"""

def create_hook_extract_cache(model, layer):
    layer_output = None
    logits = None
    def hook_fn(module, input, output):
        nonlocal layer_output
        layer_output = output
    def logits_hook_fn(module, input, output):
        nonlocal logits
        logits = output

    target_layer = model.model.layers[layer]
    hook = target_layer.register_forward_hook(hook_fn)
    logits_hook = model.lm_head.register_forward_hook(logits_hook_fn)

    def get_cache_and_logits():
        return (layer_output, logits)

    def remove_hooks():
        hook.remove()
        logits_hook.remove()

    return get_cache_and_logits, remove_hooks

for i in tqdm(range(0, len(train), 32)):
    try:
        clear_gpu_memory()

        # PI: baseline cache
        baseline_data = [item['original']['prompt'] for item in train[i:i+batch_size]]

        get_cache_and_logits, remove_hooks = create_hook_extract_cache(model, layer=check_layer)
        inputs = tokenizer(baseline_data, return_tensors="pt", padding=True,truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        cache, logits = get_cache_and_logits()
        for c, l in zip(cache, logits):
          baseline_cache.append((c.detach().cpu(), l.detach().cpu()))

        remove_hooks()

        # Clear memory between runs
        clear_gpu_memory()

        # PII: Get Jesus/counterfactual cache

        counterfactual_data = [item['counterfactual']['prompt'] for item in train[i:i+batch_size]]

        get_cache_and_logits, remove_hooks = create_hook_extract_cache(model, layer=check_layer)
        inputs = tokenizer(counterfactual_data, return_tensors="pt", padding=True,truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        cache, logits = get_cache_and_logits()
        for c, l in zip(cache, logits):
          jesus_cache.append((c.detach().cpu(), l.detach().cpu()))

        remove_hooks()

    except Exception as e:
        print(f"error processing example: {e}")
        continue

if baseline_cache:
    print(f"Type of first cache element: {type(baseline_cache[0])}")
    print(f"Structure of first cache: {type(baseline_cache[0][0])}")
    if hasattr(baseline_cache[0][0], 'shape'):
        print(f"Shape of first cache: {baseline_cache[0][0].shape}")

"""#### Finding Jesus direction"""

pos = -1
activations = [cache[0][:, pos, :] for cache in baseline_cache]
baseline_mean_act = torch.cat(activations, dim=0).mean(dim=0)

activations = [cache[0][:, pos, :] for cache in jesus_cache]
jesus_mean_act = torch.cat(activations, dim=0).mean(dim=0)

print(f"Shape of mean activation: {baseline_mean_act.shape}")

jesus_dir = jesus_mean_act-baseline_mean_act
jesus_dir = jesus_dir / jesus_dir.norm()

def remove_all_hooks(model):
    for module in model.modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()
        if hasattr(module, '_backward_hooks'):
            module._backward_hooks.clear()
        if hasattr(module, '_forward_pre_hooks'):
            module._forward_pre_hooks.clear()
    print("All hooks removed")

# Clear any existing hooks
remove_all_hooks(model)

# Clear memory
import gc
torch.cuda.empty_cache()
gc.collect()

def create_ablation_hook(direction, scale=1.0):
    def ablate_direction(module, input, output):
        # In DeepSeek/Qwen models, output is a tuple and the first element is hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]

            # Ensure direction is on the same device as hidden_states
            direction_device = direction.to(hidden_states.device)

            # Calculate projection using PyTorch operations
            dot_product = torch.sum(hidden_states * direction_device, dim=-1, keepdim=True)
            proj = dot_product * direction_device

            # Subtract a scaled projection to ablate the direction partially
            modified_hidden = hidden_states - scale * proj

            # Return modified tuple
            return (modified_hidden,) + output[1:]
        else:
            # If not a tuple, return unchanged
            return output

    return ablate_direction

# Run inference without hooks first to get baseline output
test_batch_input = [item['counterfactual']['prompt'] for item in test[4:5]]

baseline_inputs = tokenizer(test_batch_input, return_tensors="pt",padding=True,truncation=True)
baseline_inputs = {k: v.to(device) for k, v in baseline_inputs.items()}



with torch.no_grad():
    baseline_outputs = model.generate(
        baseline_inputs["input_ids"],
        attention_mask=baseline_inputs["attention_mask"],
        max_new_tokens=750,
        do_sample=False  # Use greedy decoding
    )

# Decode the output tokens to text
baseline_texts = tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True)
print("Baseline text outputs DONE")
for text in baseline_texts:
    print(text)

# Now run with ablation hooks to get modified output
remove_all_hooks(model)  # Remove any existing hooks
hooks = []

ablation_hook = create_ablation_hook(jesus_dir,scale=1.0)
for layer_idx in range(len(model.model.layers)):
    hook = model.model.layers[layer_idx].register_forward_hook(ablation_hook)
    hooks.append(hook)

try:
    with torch.no_grad():
        ablated_outputs = model.generate(
            baseline_inputs["input_ids"],
            attention_mask=baseline_inputs["attention_mask"],
            max_new_tokens=500,
            do_sample=False
        )

    # Decode the output tokens
    ablated_text = tokenizer.batch_decode(ablated_outputs, skip_special_tokens=True)
    print("\nAblated text outputs DONE")
    for text in ablated_text:
      print(text)

finally:
    # Always remove hooks
    for hook in hooks:
        hook.remove()

test_text = [item['counterfactual']['prompt'] for item in test[2:3]]
baseline_inputs = tokenizer(test_text, return_tensors="pt",padding=True,truncation=True)
baseline_inputs = {k: v.to(device) for k, v in baseline_inputs.items()}

model.eval()
# Now run with ablation hooks to get modified output
remove_all_hooks(model)  # Remove any existing hooks
hooks = []

ablation_hook = create_ablation_hook(jesus_dir,scale=1.0)
for layer_idx in range(len(model.model.layers)):
    hook = model.model.layers[layer_idx].register_forward_hook(ablation_hook)
    hooks.append(hook)

try:
    with torch.no_grad():
        ablated_outputs = model.generate(
            baseline_inputs["input_ids"],
            attention_mask=baseline_inputs["attention_mask"],
            max_new_tokens=1000,
            do_sample=False
        )

    # Decode the output tokens
    ablated_text = tokenizer.batch_decode(ablated_outputs, skip_special_tokens=True)
    print("\nAblated text outputs DONE")
    for text in ablated_text:
      print(text)

finally:
    # Always remove hooks
    for hook in hooks:
        hook.remove()

clear_gpu_memory()