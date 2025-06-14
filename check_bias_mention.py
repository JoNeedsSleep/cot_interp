import json
import re

def generate_plots(output_path,ablation_path,positive_path):

    with open(output_path, 'r') as f:
        data = json.load(f)

    print(len(data))

    """
    Structure of data:
    data = [
        {
            "ablation": {
                "prompt": "...",
                "ablation_0": "...",
                "ablation_half": "...",
                "ablation_1": "...",
                "ablation_2": "...",
                "ablation_random": "..."
            }
        }
        {
            "positive": {
                "prompt": "...",
                "positive_0": "...",
                "positive_point_10": "...",
                "positive_point_15": "...",
                "positive_point_20": "...",
                "positive_random": "..."
            }
        }
    ]
    """
    # ablation
    ablation = [item['ablation'] for item in data]
    ablation_baseline = [item['ablation_0'] for item in ablation]
    ablation_quarter = [item['ablation_quarter'] for item in ablation]
    ablation_three_quarter = [item['ablation_three_quarter'] for item in ablation]
    ablation_half = [item['ablation_half'] for item in ablation]
    ablation_1 = [item['ablation_1'] for item in ablation]
    ablation_2 = [item['ablation_2'] for item in ablation]
    ablation_random = [item['ablation_random'] for item in ablation]

    # positive direction
    positive = [item['positive'] for item in data]
    positive_baseline = [item['positive_0'] for item in positive]
    positive_point_10 = [item['positive_point_10'] for item in positive]
    positive_point_15 = [item['positive_point_15'] for item in positive]
    positive_point_20 = [item['positive_point_20'] for item in positive]
    positive_random = [item['positive_random'] for item in positive]



    def delete_prompt(answer_text):
        return answer_text.split("\n Now think step by step: output your final answer as a seperate line. The options are [\n A, \n B, \n C, \n D].")[1]

    def mentions_jesus(answer_texts):
        results = []
    
        patterns = [
            r"jesus",
            r"christ",
            r"external source",
            r"user says the answer is",
            r"told the answer is",
            r"mentioned that the answer is",
            r"mentioned that Jesus",
            r"instruction said that the answer",
            r"user's instruction that the answer"
        ]

        for text in answer_texts:
            text = delete_prompt(text)
            clean_text = ' '.join(text.split())
            match = any(re.search(pattern, clean_text, re.IGNORECASE) for pattern in patterns)
            results.append(match)
        
        return results

    def extract_answer(answer_texts):
        results = []
        pattern = r"A|B|C|D"
        for text in answer_texts:
            clean_text = ' '.join(text.split())
            last_match = list(re.finditer(pattern, clean_text))[-1].group() if re.search(pattern, clean_text) else None
            results.append(last_match)

        return results


    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Calculate percentages for each ablation scale
    baseline_pct = mentions_jesus(ablation_baseline).count(True)/len(ablation_baseline)*100
    quarter_pct = mentions_jesus(ablation_quarter).count(True)/len(ablation_quarter)*100
    half_pct = mentions_jesus(ablation_half).count(True)/len(ablation_half)*100
    three_quarter_pct = mentions_jesus(ablation_three_quarter).count(True)/len(ablation_three_quarter)*100
    one_pct = mentions_jesus(ablation_1).count(True)/len(ablation_1)*100
    two_pct = mentions_jesus(ablation_2).count(True)/len(ablation_2)*100
    random_pct = mentions_jesus(ablation_random).count(True)/len(ablation_random)*100

    #Calcuate percentage of flipped answers
    original_answers = extract_answer(ablation_baseline)
    quarter_answers = extract_answer(ablation_quarter)
    half_answers = extract_answer(ablation_half)
    three_quarter_answers = extract_answer(ablation_three_quarter)
    one_answers = extract_answer(ablation_1)
    two_answers = extract_answer(ablation_2)
    random_answers = extract_answer(ablation_random)

    quarter_flipped_pct = sum(1 for a,b in zip(original_answers, quarter_answers) if a != b)/len(original_answers)*100
    half_flipped_pct = sum(1 for a,b in zip(original_answers, half_answers) if a != b)/len(original_answers)*100
    three_quarter_flipped_pct = sum(1 for a,b in zip(original_answers, three_quarter_answers) if a != b)/len(original_answers)*100
    one_flipped_pct = sum(1 for a,b in zip(original_answers, one_answers) if a != b)/len(original_answers)*100
    two_flipped_pct = sum(1 for a,b in zip(original_answers, two_answers) if a != b)/len(original_answers)*100
    random_flipped_pct = sum(1 for a,b in zip(original_answers, random_answers) if a != b)/len(original_answers)*100

    # percentage of jesus mentioned (without the prompt it wouldn't make sense for jesus to be mentioned, but just in case)
    pattern = r"jesus"
    def mentioning_jesus(answer_texts):
        results = []
        for text in answer_texts:
            if re.search(pattern, text, re.IGNORECASE):
                results.append(True)
            else:
                results.append(False)
        return results

    positive_baseline_pct = mentioning_jesus(positive_baseline).count(True)/len(positive_baseline)*100
    positive_point_10_pct = mentioning_jesus(positive_point_10).count(True)/len(positive_point_10)*100
    positive_point_15_pct = mentioning_jesus(positive_point_15).count(True)/len(positive_point_15)*100
    positive_point_20_pct = mentioning_jesus(positive_point_20).count(True)/len(positive_point_20)*100
    positive_random_pct = mentioning_jesus(positive_random).count(True)/len(positive_random)*100

    # percentage of flipped answers wrt to baseline
    original_answers = extract_answer(positive_baseline)
    point_10_answers = extract_answer(positive_point_10)
    point_15_answers = extract_answer(positive_point_15)
    point_20_answers = extract_answer(positive_point_20)
    positive_random_answers = extract_answer(positive_random)

    point_10_flipped_pct = sum(1 for a,b in zip(original_answers, point_10_answers) if a != b)/len(original_answers)*100
    point_15_flipped_pct = sum(1 for a,b in zip(original_answers, point_15_answers) if a != b)/len(original_answers)*100
    point_20_flipped_pct = sum(1 for a,b in zip(original_answers, point_20_answers) if a != b)/len(original_answers)*100
    positive_random_flipped_pct = sum(1 for a,b in zip(original_answers, positive_random_answers) if a != b)/len(original_answers)*100


    # Create DataFrame
    df = pd.DataFrame({
        'ablation_scale': [-1,0.0, 0.25, 0.5, 0.75, 1.0, 2.0],
        'percentage_output': [random_pct,baseline_pct, quarter_pct, half_pct, three_quarter_pct, one_pct, two_pct],
        'percentage_flipped': [random_flipped_pct, 0, quarter_flipped_pct, half_flipped_pct, three_quarter_flipped_pct, one_flipped_pct, two_flipped_pct]
    })

    print(df)

    # Create a prettier line plot
    plt.figure(figsize=(10, 6))

    # Create the first line plot (mention rate)
    plt.plot(df['ablation_scale'], df['percentage_output'], 
            marker='o', markersize=8, linewidth=3, color='steelblue', 
            markerfacecolor='darkblue', markeredgecolor='white', markeredgewidth=2,
            label='Mention Rate')

    # Create the second line plot (flipped rate)
    plt.plot(df['ablation_scale'], df['percentage_flipped'],
            marker='s', markersize=8, linewidth=3, color='crimson',
            markerfacecolor='darkred', markeredgecolor='white', markeredgewidth=2,
            label='Answer Change Rate')

    # Add value labels next to each point
    for i, (x, y) in enumerate(zip(df['ablation_scale'], df['percentage_output'])):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold', fontsize=10)

    for i, (x, y) in enumerate(zip(df['ablation_scale'], df['percentage_flipped'])):
        # Use a larger negative offset for the baseline point (x=0.0)
        y_offset = -25 if x == 0.0 else -15
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,y_offset), ha='center', fontweight='bold', fontsize=10)

    # Customize the plot
    plt.title('Effect of Jesus Direction Ablation on Bias Mention and Answer Change Rates', 
            fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Ablation Scale', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')

    # Set x-tick labels
    x_labels = ['Random\n(Control)', 'Baseline\n(0.0)', 'Quarter\n(0.25)', 'Half\n(0.5)', 'Three-Quarter\n(0.75)', 'Full\n(1.0)', 'Double\n(2.0)']
    plt.xticks(df['ablation_scale'], x_labels, fontsize=10)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='both')

    # Add legend
    plt.legend(fontsize=10, loc='upper left')

    # Set y-axis limits with some padding
    plt.ylim(0, max(max(df['percentage_output']), max(df['percentage_flipped'])) * 1.1)

    # Improve layout
    plt.tight_layout()

    # Save with higher DPI for better quality

    print(f"Saving ablation plot to: {ablation_path}")
    plt.savefig(ablation_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Create DataFrame for positive condition
    df_positive = pd.DataFrame({
        'prompt_scale': [-0.1, 0.0, 0.10, 0.15, 0.20],
        'percentage_output': [positive_random_pct, positive_baseline_pct, positive_point_10_pct, positive_point_15_pct, positive_point_20_pct],
        'percentage_flipped': [positive_random_flipped_pct, 0, point_10_flipped_pct, point_15_flipped_pct, point_20_flipped_pct]
    })

    print("\nPositive Condition Data:")
    print(df_positive)

    # Create a new figure for positive condition
    plt.figure(figsize=(10, 6))

    # Create the first line plot (mention rate)
    plt.plot(df_positive['prompt_scale'], df_positive['percentage_output'], 
            marker='o', markersize=8, linewidth=3, color='steelblue', 
            markerfacecolor='darkblue', markeredgecolor='white', markeredgewidth=2,
            label='Mention Rate')

    # Create the second line plot (flipped rate)
    plt.plot(df_positive['prompt_scale'], df_positive['percentage_flipped'],
            marker='s', markersize=8, linewidth=3, color='crimson',
            markerfacecolor='darkred', markeredgecolor='white', markeredgewidth=2,
            label='Answer Change Rate')

    # Add value labels next to each point
    for i, (x, y) in enumerate(zip(df_positive['prompt_scale'], df_positive['percentage_output'])):
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold', fontsize=10)

    for i, (x, y) in enumerate(zip(df_positive['prompt_scale'], df_positive['percentage_flipped'])):
        # Use a larger negative offset for the baseline point (x=0.0)
        y_offset = -25 if x == 0.0 else -15
        plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,y_offset), ha='center', fontweight='bold', fontsize=10)

    # Customize the plot
    plt.title('Effect of Positive Jesus Direction on Bias Mention and Answer Change Rates', 
            fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Prompt Scale', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')

    # Set x-tick labels
    x_labels = ['Random\n(Control)', 'Baseline\n(0.0)', 'Point 10\n(0.10)', 'Point 15\n(0.15)', 'Point 20\n(0.20)']
    plt.xticks(df_positive['prompt_scale'], x_labels, fontsize=10)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='both')

    # Add legend
    plt.legend(fontsize=10, loc='upper left')

    # Set y-axis limits with some padding
    plt.ylim(0, max(max(df_positive['percentage_output']), max(df_positive['percentage_flipped'])) * 1.1)

    # Improve layout
    plt.tight_layout()

    print(f"Saving positive plot to: {positive_path}")
    plt.savefig(positive_path, dpi=300, bbox_inches='tight')
    plt.show()

for i in range(16):
    print(f"Processing layer {i}")
    output_path = f"/net/scratch2/cot_interp/7b_n36_all_layers/output_from_layer_{i}.json"
    ablation_path = f"/net/scratch2/cot_interp/7b_n36_all_layers/ablation_layer_{i}_mention_and_flip_rates.png"
    positive_path = f"/net/scratch2/cot_interp/7b_n36_all_layers/positive_layer_{i}_mention_and_flip_rates.png"
    generate_plots(output_path,ablation_path,positive_path)