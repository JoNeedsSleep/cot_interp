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

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
dataset_directory = "filtered_dataset.json"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

# Get the main model component (this can vary by architecture)
# For many models, it's model.model or model.transformer
# You might need to print(model) to see its structure first
main_model_component = model.model # Adjust this if needed based on print(model) output

# Check if the component has a 'layers' attribute (common for transformer decoders)
if hasattr(main_model_component, 'layers'):
    num_layers = len(main_model_component.layers)
    print(f"Number of layers: {num_layers}")
    if num_layers > 0:
        for i in range(num_layers):
            print(f"Layer {i} type: {type(main_model_component.layers[i])}")
elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'): # For encoder-decoder or encoder-only
    num_layers = len(model.encoder.layer)
    print(f"Number of encoder layers: {num_layers}")
    if num_layers > 0:
        print(f"Type of the first encoder layer: {type(model.encoder.layer[0])}")
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'layer'):
        num_decoder_layers = len(model.decoder.layer)
        print(f"Number of decoder layers: {num_decoder_layers}")
        if num_decoder_layers > 0:
            print(f"Type of the first decoder layer: {type(model.decoder.layer[0])}")
else:
    print("Could not automatically determine the layer structure.")
    print("Print the model architecture to inspect manually:")
    print(model)

