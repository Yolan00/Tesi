import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIGURATION ===
CSV_PATH = "/projects/0/prjs1482/UvA/AA_PART_1/Baseline/Qwen/word_frequency_table.csv"  # Change to your CSV filename

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH, index_col=0)  # Assumes words in rows, conditions in columns

# === COMPUTE COSINE DISSIMILARITY ===
X = df.T.values  # conditions as rows, words as features
sim_matrix = cosine_similarity(X)
dissim_matrix = 1 - sim_matrix

# === PLOT HEATMAP ===
conditions = df.columns.tolist()

plt.figure(figsize=(8, 6))
sns.heatmap(dissim_matrix, annot=True, fmt=".2f", xticklabels=conditions, yticklabels=conditions, cmap="viridis")
plt.title("Cosine Dissimilarity Matrix (Word Usage)")
plt.tight_layout()
plt.savefig("cosine_dissimilarity_heatmap.png", dpi=300)

