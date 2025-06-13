import re

def extract_answer(answer_texts):
    results = []
    pattern = r"A|B|C|D"
    for text in answer_texts:
        print(text)
        clean_text = ' '.join(text.split())
        last_match = list(re.finditer(pattern, clean_text))[-1].group() if re.search(pattern, clean_text) else None
        results.append(last_match)

    return results

example_texts = [
    "I think the answer is A",
    "B",
    "A,B,C,D could all be correct but B seems to be the most likely answer"]

print(extract_answer(example_texts))
