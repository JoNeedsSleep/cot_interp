from utils import load_dataset, evaluate_model, analyze_results
import torch
import json


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset_path = "hellaswag_high_auth_counterfactuals_1.json"
output_path = "results_jesus.json"
num_samples = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset(dataset_path)
results = evaluate_model(model_name, dataset, num_samples=num_samples, device=device)
analysis = analyze_results(results)
output_data = {
    "model": model_name,
    "results": results,
    "analysis": analysis
}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print("\nEvaluation Summary:")
print(f"Model: {model_name}")
print(f"Total samples: {analysis['total_samples']}")
print(f"Original prompt correct rate: {analysis['original_metrics']['correct_rate']:.2f}")
print(f"Counterfactual divergence rate: {analysis['counterfactual_metrics']['divergence_rate']:.2f}")
print(f"Counterfactual intended effect rate: {analysis['counterfactual_metrics']['intended_effect_rate']:.2f}")
print(f"Correct to incorrect conversion rate: {analysis['impact_metrics']['correct_to_incorrect_rate']:.2f}")
print(f"Results saved to: {output_path}")