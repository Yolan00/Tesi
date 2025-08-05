
#Change: Whole vocabulary, number of words related to position, colors, objects and actions.

import json
import string
from collections import Counter

STOPWORDS = set("""
a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with you your from has have he her his its our ours so than too very can could would should
""".split())
TEMPLATE_PREFIX = "the target is the only one with"

def clean_and_tokenize(description):
    desc = description.lower().strip()
    if desc.startswith(TEMPLATE_PREFIX):
        desc = desc[len(TEMPLATE_PREFIX):]
    desc = desc.translate(str.maketrans('', '', string.punctuation))
    tokens = desc.split()
    tokens = [w for w in tokens if w not in STOPWORDS and not w.isdigit() and w.strip()]
    return tokens

def most_frequent_words(json_paths, top_n=20):
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
        all_tokens = []
        for item in data:
            desc = item['description']
            tokens = clean_and_tokenize(desc)
            all_tokens.extend(tokens)
        counter = Counter(all_tokens)
        print(f"\nTop {top_n} content words for {json_path}:")
        for word, count in counter.most_common(top_n):
            print(f"{word}: {count}")

# List your JSON files here:
json_files = [
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR5_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR10_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR15_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR20_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR25_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/PART_1/Qwen/Baseline/qwen_baseline_output_descriptions.json',
]


most_frequent_words(json_files, top_n=20)
