from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

# Print all layer names and their indices
print("Model Layers:")
for layer in model.model.layers:
    print(layer)

print("\nTotal number of transformer layers:", len(model.model.layers))




