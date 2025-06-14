from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import json
import numpy as np

def make_json_serializable(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# Available subjects (57 total)
subjects = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions"
]

# Load multiple subjects
datasets_json = {}

for subject in subjects[:5]:
    dataset = load_dataset("cais/mmlu", subject)
    print(f"Loading {subject}")
    
    datasets_json[subject] = {}
    for split in dataset.keys():
        raw_data = [dict(item) for item in dataset[split]]
        datasets_json[subject][split] = make_json_serializable(raw_data)
        print(f"  {split}: {len(datasets_json[subject][split])} examples")

# Save to JSON
with open('mmlu_datasets.json', 'w') as f:
    json.dump(datasets_json, f, indent=2)

print("Successfully saved to mmlu_datasets.json")