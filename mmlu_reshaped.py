import json

def reshape_mmlu_dataset(input_file, output_file, template_file):
    """
    Reshape MMLU dataset using the given template
    """
    
    # Read the template
    with open(template_file, 'r') as f:
        template = f.read().strip()
    
    # Read the original dataset
    with open(input_file, 'r') as f:
        original_data = json.load(f)
    
    reshaped_data = {}
    
    for subject, splits in original_data.items():
        print(f"Processing {subject}...")
        reshaped_data[subject] = {}
        
        for split_name, items in splits.items():
            reshaped_data[subject][split_name] = []
            
            for item in items:
                # Extract the original data
                question = item['question']
                choices = item['choices']
                answer = item['answer']  # This is the index (0, 1, 2, 3)
                
                # Format the prompt using the template
                formatted_prompt = template.format(
                    question=question,
                    option1=choices[0],
                    option2=choices[1],
                    option3=choices[2],
                    option4=choices[3]
                )
                
                # Convert answer index to 1-based string
                correct_answer = str(answer + 1)  # Convert 0,1,2,3 to "1","2","3","4"
                
                # Create the reshaped item
                reshaped_item = {
                    "original_prompt": formatted_prompt,
                    "correct_answer": correct_answer,
                    "subject": item['subject'],
                    "original_question": question,
                    "choices": choices,
                    "answer_index": answer
                }
                
                reshaped_data[subject][split_name].append(reshaped_item)
            
            print(f"  {split_name}: {len(reshaped_data[subject][split_name])} items")
    
    # Save the reshaped dataset
    with open(output_file, 'w') as f:
        json.dump(reshaped_data, f, indent=2)
    
    print(f"\nReshaped dataset saved to {output_file}")
    return reshaped_data

# Run the reshaping
reshaped_data = reshape_mmlu_dataset(
    input_file='mmlu_datasets.json',
    output_file='mmlu_reshaped.json', 
    template_file='mmlu_reshape.txt'
)

# Print a sample to verify
print("\nSample reshaped item:")
first_subject = list(reshaped_data.keys())[0]
first_item = reshaped_data[first_subject]['test'][0]
print(f"Subject: {first_item['subject']}")
print(f"Prompt:\n{first_item['original_prompt']}")
print(f"Correct answer: {first_item['correct_answer']}")

