import json
import string
from collections import Counter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---- CONFIG ----
json_files = [
    '/projects/0/prjs1482/UvA/Outputs/PART_1/Qwen/Baseline/qwen_baseline_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR5_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR10_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR15_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR20_output_descriptions.json',
    '/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR25_output_descriptions.json',
]
labels = ['Original', 'BLUR5', 'BLUR10', 'BLUR15', 'BLUR20', 'BLUR25']  # same order as above

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

# 1. Build a Counter for each file
all_freqs = []
for json_path in json_files:
    with open(json_path, 'r') as f:
        data = json.load(f)
    tokens = []
    for item in data:
        tokens.extend(clean_and_tokenize(item['description']))
    all_freqs.append(Counter(tokens))

# 2. Create frequency DataFrame
vocab = sorted(set().union(*[c.keys() for c in all_freqs]))
data = {label: [freq.get(w, 0) for w in vocab] for label, freq in zip(labels, all_freqs)}
df = pd.DataFrame(data, index=vocab)

# 3. Save frequency table
df.to_csv("word_frequency_table.csv")
print("\n[+] Word frequency table saved as word_frequency_table.csv")
print(df.head(15))

# 4. Cosine similarity
mat = df.values.T  # shape (n_conditions, n_words)
sim_matrix = cosine_similarity(mat)
sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
print("\nCosine similarity between conditions:\n")
print(sim_df.round(3))

# 5. Save similarity matrix
sim_df.to_csv("cosine_similarity_matrix.csv")
print("\n[+] Cosine similarity matrix saved as cosine_similarity_matrix.csv")
