import torch
import gc
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
import numpy as np  
import os
import gc

model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)


def run_probe_pipeline(check_layer=25, 
              batch_size=8,
              dataset_directory="filtered_dataset.json", 
              output_path="outputs3.json",
              save_jesus_vector_path="jesus_vector.pt",
              same_dataset=True):
    def clear_gpu_memory():
        gc.collect()
        torch.cuda.empty_cache()

    clear_gpu_memory()
    with open(dataset_directory, "r") as f:
        base_dataset = json.load(f)

    baseline_cache = []
    jesus_cache = []

    if os.path.exists("/net/scratch2/cot_interp/train_test_split.json") and same_dataset == True:
        with open("/net/scratch2/cot_interp/train_test_split.json", "r") as f:
            data = json.load(f)
            train, test = data["train"], data["test"]
        print(f"Train-test split loaded from train_test_split.json")
    else:
        train, test = train_test_split(base_dataset, test_size=0.3, random_state=42)
        with open("/net/scratch2/cot_interp/train_test_split.json", "w") as f:
            json.dump({"train": train, "test": test}, f)
        print(f"Train-test split saved as JSON to train_test_split.json")

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

    ################################################################################
    #                                  Run with cache                              #
    ################################################################################

    for i in tqdm(range(0, len(train), batch_size)):
        try:
            clear_gpu_memory()

            # PI: baseline cache
            baseline_data = [item['original_prompt'] for item in train[i:i+batch_size]]

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

            counterfactual_data = [item['counterfactual_prompt'] for item in train[i:i+batch_size]]

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

    ################################################################################
    #                                  Finding Jesus direction                     #
    ################################################################################

    pos = -1
    activations = [cache[0][:, pos, :] for cache in baseline_cache]
    baseline_mean_act = torch.cat(activations, dim=0).mean(dim=0)

    activations = [cache[0][:, pos, :] for cache in jesus_cache]
    jesus_mean_act = torch.cat(activations, dim=0).mean(dim=0)

    print(f"Shape of mean activation: {baseline_mean_act.shape}")

    jesus_dir = jesus_mean_act-baseline_mean_act
    jesus_dir = jesus_dir / jesus_dir.norm()

    torch.save(jesus_dir, save_jesus_vector_path)

    # generate random direction
    random_dir = torch.randn_like(baseline_mean_act)
    random_dir = random_dir / random_dir.norm()

    def remove_all_hooks(model):
        for module in model.modules():
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
            if hasattr(module, '_backward_hooks'):
                module._backward_hooks.clear()
            if hasattr(module, '_forward_pre_hooks'):
                module._forward_pre_hooks.clear()
        print("All hooks removed")

    ################################################################################
    #                                  Ablation                                  #
    ################################################################################

    remove_all_hooks(model)
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
                print("Output is not a tuple")
                # If not a tuple, return unchanged
                return output

        return ablate_direction

    def create_positive_hook(direction, scale=1.0):
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
                modified_hidden = hidden_states + scale * proj

                # Return modified tuple
                return (modified_hidden,) + output[1:]
            else:
                print("Output is not a tuple")
                # If not a tuple, return unchanged
                return output

        return ablate_direction

    # Run inference without hooks first to get baseline output
    test_batch_input_ablation = [item['counterfactual_prompt'] for item in test]
    baseline_inputs_ablation = tokenizer(test_batch_input_ablation, return_tensors="pt",padding=True,truncation=True)
    baseline_inputs_ablation = {k: v.to(device) for k, v in baseline_inputs_ablation.items()}

    test_batch_input_positive = [item['original_prompt'] for item in test]
    baseline_inputs_positive = tokenizer(test_batch_input_positive, return_tensors="pt",padding=True,truncation=True)
    baseline_inputs_positive = {k: v.to(device) for k, v in baseline_inputs_positive.items()}

    with torch.no_grad():
        baseline_outputs = model.generate(
            baseline_inputs_ablation["input_ids"],
            attention_mask=baseline_inputs_ablation["attention_mask"],
            max_new_tokens=750,
            do_sample=False  # Use greedy decoding
        )

    # Decode the output tokens to text
    baseline_texts = tokenizer.batch_decode(baseline_outputs, skip_special_tokens=True)
    print("Baseline text outputs DONE")
    Baseline_outputs = []
    for text in baseline_texts:
        Baseline_outputs.append(text)

    with torch.no_grad():
        baseline_outputs_positive = model.generate(
            baseline_inputs_positive["input_ids"],
            attention_mask=baseline_inputs_positive["attention_mask"],
            max_new_tokens=750,
            do_sample=False  # Use greedy decoding
        )

    # Decode the output tokens to text
    baseline_texts_positive = tokenizer.batch_decode(baseline_outputs_positive, skip_special_tokens=True)
    print("Baseline text outputs DONE")
    baseline_outputs_positive = []
    for text in baseline_texts_positive:
        baseline_outputs_positive.append(text)


    def run_with_hooks(hook,inputs):
        remove_all_hooks(model)
        for layer_idx in range(len(model.model.layers)):
            handle = model.model.layers[layer_idx].register_forward_hook(hook)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=500,
                do_sample=False
            )

        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = []

        for text in output_text:
            outputs.append(text)

        print("Outputs generated")
        return outputs

    ablation_hook_quarter = create_ablation_hook(jesus_dir,scale=0.25)
    Ablated_outputs_quarter = run_with_hooks(ablation_hook_quarter,baseline_inputs_ablation)
    ablation_hook_three_quarter = create_ablation_hook(jesus_dir,scale=0.75)
    Ablated_outputs_three_quarter = run_with_hooks(ablation_hook_three_quarter,baseline_inputs_ablation)
    ablation_hook_half = create_ablation_hook(jesus_dir,scale=0.5)
    Ablated_outputs_half = run_with_hooks(ablation_hook_half,baseline_inputs_ablation)
    ablation_hook = create_ablation_hook(jesus_dir,scale=1.0)
    Ablated_outputs = run_with_hooks(ablation_hook,baseline_inputs_ablation)
    ablation_hook_2 = create_ablation_hook(jesus_dir,scale=2.0)
    Ablated_outputs_2 = run_with_hooks(ablation_hook_2,baseline_inputs_ablation)
    ablation_hook_random = create_ablation_hook(random_dir,scale=1.0)
    Ablated_outputs_random = run_with_hooks(ablation_hook_random,baseline_inputs_ablation)

    ablation_hook_point_10 = create_positive_hook(jesus_dir,scale=0.1)
    Ablated_outputs_point_10 = run_with_hooks(ablation_hook_point_10,baseline_inputs_positive)
    ablation_hook_point_15 = create_positive_hook(jesus_dir,scale=0.15)
    Ablated_outputs_point_15 = run_with_hooks(ablation_hook_point_15,baseline_inputs_positive)
    ablation_hook_point_20 = create_positive_hook(jesus_dir,scale=0.2)
    Ablated_outputs_point_20 = run_with_hooks(ablation_hook_point_20,baseline_inputs_positive)
    ablation_hook_point_25 = create_positive_hook(jesus_dir,scale=0.25)
    Ablated_outputs_point_25 = run_with_hooks(ablation_hook_point_25,baseline_inputs_positive)
    ablation_hook_random_positive = create_positive_hook(random_dir,scale=0.1)
    Ablated_outputs_random_positive = run_with_hooks(ablation_hook_random_positive,baseline_inputs_positive)

    structured_outputs = []

    for i in range(len(test)):  # Assuming test, Baseline_outputs, and Ablated_outputs all have the same length
        item = {
            "ablation": {
                "prompt": test[i]["counterfactual_prompt"],
                "ablation_0": Baseline_outputs[i],
                "ablation_quarter": Ablated_outputs_quarter[i],
                "ablation_half": Ablated_outputs_half[i],
                "ablation_three_quarter": Ablated_outputs_three_quarter[i],
                "ablation_1": Ablated_outputs[i],
                "ablation_2": Ablated_outputs_2[i],
                "ablation_random": Ablated_outputs_random[i]
            },
            "positive": {
                "prompt": test[i]["original_prompt"],
                "positive_0": baseline_outputs_positive[i],
                "positive_point_10": Ablated_outputs_point_10[i],
                "positive_point_15": Ablated_outputs_point_15[i],
                "positive_point_20": Ablated_outputs_point_20[i],
                "positive_point_25": Ablated_outputs_point_25[i],
                "positive_random": Ablated_outputs_random_positive[i]
            }
        }
        structured_outputs.append(item)

    ################################################################################
    #                                  Save results                               #
    ################################################################################

    with open(output_path, "w") as f:
        json.dump(structured_outputs, f, indent=2)  # indent=2 makes the JSON file more readable

for i in range(28):
    print(f"Running probe pipeline for layer {i}")
    run_probe_pipeline(check_layer=i, 
                       output_path=f"/net/scratch2/cot_interp/7b_n36_all_layers/output_from_layer_{i}.json",
                       save_jesus_vector_path=f"/net/scratch2/cot_interp/7b_n36_all_layers/vector_from_layer_{i}.pt")