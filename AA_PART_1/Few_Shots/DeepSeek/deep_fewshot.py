import os
import statistics
import json
import random
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from deepseek_vl2.models import DeepseekVLV2Processor
from transformers import AutoModelForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# ----------------------------
# Configuration
# ----------------------------
DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"
OUTPUT_FILE = "/projects/0/prjs1482/UvA/Outputs/PART_1/Deep/Few_Shots/deepseek_fewshot_output.txt"
MODEL_PATH = "deepseek-ai/deepseek-vl2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 4
IMAGE_EXT = ".jpg"

# ----------------------------
# Variables/statistics
# ----------------------------
correct_count = 0
wrong_count = 0
total_count = 0
trial_index = 0
total_description_length = 0
description_lengths = []

prob_correct = []
prob_wrong = []

rank1_probs_correct, rank2_probs_correct, rank3_probs_correct, rank4_probs_correct = [], [], [], []
rank1_probs_wrong,   rank2_probs_wrong,   rank3_probs_wrong,   rank4_probs_wrong   = [], [], [], []

speaker_desc_records = []

# ----------------------------
# Load model and processor
# ----------------------------
print("Loading DeepSeek-VL2...")
processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = model.to(torch.bfloat16).to(DEVICE).eval()

# ----------------------------
# Helper functions
# ----------------------------
def load_images(image_paths):
    return [Image.open(p).convert("RGB") for p in image_paths]

def find_csv_file(folder):
    for f in os.listdir(folder):
        if f.endswith(".csv") and f.startswith("trials_for_"):
            return os.path.join(folder, f)
    return None

def get_image_paths(folder, ext=IMAGE_EXT):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)])

def get_answer_variants():
    return {
        "first":  ["first", "First", "one", "One", "1", "1st", " 1", " 1st", " first", " First"],
        "second": ["second", "Second", "two", "Two", "2", "2nd", " 2", " 2nd", " second", " Second"],
        "third":  ["third", "Third", "three", "Three", "3", "3rd", " 3", " 3rd", " third", " Third"],
        "fourth": ["fourth", "Fourth", "four", "Four", "4", "4th", " 4", " 4th", " fourth", " Fourth"],
    }

def get_all_variant_token_ids(tokenizer, answer_variants):
    variant_token_ids = {}
    for key, variants in answer_variants.items():
        token_ids = set()
        for v in variants:
            for prefix in ["", " "]:
                ids = tokenizer.encode(prefix + v, add_special_tokens=False)
                if len(ids) == 1:
                    token_ids.add(ids[0])
        variant_token_ids[key] = token_ids
    return variant_token_ids

def get_cumulative_probs(probs_full, variant_token_ids):
    cum_probs = {}
    for ans, tids in variant_token_ids.items():
        prob = sum(float(probs_full[tid]) for tid in tids if tid < len(probs_full))
        cum_probs[ans] = prob
    sorted_items = sorted(cum_probs.items(), key=lambda x: -x[1])
    return sorted_items


def get_few_shot_examples(base_dir, current_folder):
    try:
        folder_pool = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f != current_folder]
        examples = []
        for _ in range(2):  # Now sample two different examples
            example_folder = random.choice(folder_pool)
            example_path = os.path.join(base_dir, example_folder)
            example_image_paths = get_image_paths(example_path)
            csv_path = find_csv_file(example_path)
            if not csv_path or len(example_image_paths) != NUM_IMAGES:
                continue
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if len(df) < NUM_IMAGES:
                continue
            idx = random.choice(range(NUM_IMAGES))
            desc = df.iloc[idx]['msg']
            ordinal = ['first', 'second', 'third', 'fourth'][idx]
            examples.append({
                "images": example_image_paths,
                "target_idx": idx,
                "target_label": ordinal,
                "desc": desc,
                "folder": example_folder
            })
            folder_pool.remove(example_folder)  # Ensure two different folders
        if len(examples) == 2:
            return examples
        else:
            return None
    except Exception as e:
        print(f"Error getting few-shot examples: {e}")
        return None

# ----------------------------
# Main trial
# ----------------------------
def run_trial(trial_path, target_idx):
    global trial_index, correct_count, wrong_count, total_count, total_description_length
    global prob_correct, prob_wrong, description_lengths
    global rank1_probs_correct, rank2_probs_correct, rank3_probs_correct, rank4_probs_correct
    global rank1_probs_wrong, rank2_probs_wrong, rank3_probs_wrong, rank4_probs_wrong
    global speaker_desc_records

    trial_index += 1

    img_paths = get_image_paths(trial_path)
    if len(img_paths) != NUM_IMAGES:
        print(f"Skipping {trial_path}: expected {NUM_IMAGES} images, found {len(img_paths)}")
        return

    csv_path = find_csv_file(trial_path)
    if not csv_path:
        print(f"Skipping {trial_path}: no suitable .csv file found")
        return

    fewshot_examples = get_few_shot_examples(DATASET_DIR, os.path.basename(trial_path))
    if not fewshot_examples:
        print(f"Skipping {trial_path}: could not find enough few-shot examples.")
        return

    target_label = ['first', 'second', 'third', 'fourth'][target_idx]
    target_img_name = os.path.basename(img_paths[target_idx])

    # SPEAKER PROMPT WITH ONE FEW-SHOT EXAMPLE
    prompt = (
        "<|grounding|> You are playing a reference game as the speaker. I will give you 4 different images. "
        "Your task is to generate a description of the TARGET image so that a listener can identify it among the others. "
        "Do NOT mention the position or order of the image (e.g., 'first', 'second', etc.). "
        "Focus only on unique visual details that distinguish the target.\n"
        "Look at this example, 4 images are given,\n"
        "<image>\n"
        "<image>\n"
        "<image>\n"
        "<image>\n"
        f"Then the TARGET is selected, in this case, the {fewshot_examples[0]['target_label']} image, "
        f"And a description is generated, DESCRIPTION: {fewshot_examples[0]['desc']}\n"
        "Now it is your turn to generate a description for the target image.\n\n"
        "Those are your 4 images,\n"
        "First: <image>\n"
        "Second: <image>\n"
        "Third: <image>\n"
        "Fourth: <image>\n"
        f"Your TARGET is the {target_label} image.\n"
        f"Your DESCRIPTION of the {target_label} image:"
    )


    # Only one example for images
    all_images = []
    all_images += fewshot_examples[0]['images']
    all_images += img_paths

    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": all_images,
        },
        {"role": "<|Assistant|>", "content": ""}
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(DEVICE)

    with torch.no_grad():
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True
        )
    generated_desc = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()

    total_description_length += len(generated_desc.split())
    description_lengths.append(len(generated_desc.split()))
    raw_speaker_prompt = prompt

    # LISTENER
    listener_prompt = (
        f"You are playing a reference game as the listener. Here is a description: \"{generated_desc}\"\n"
        f"Which of the four images does this description refer to?\n"
        f"first: <image>\n second: <image>\n third: <image>\n fourth: <image>\n\n"
        f"Answer with one word: first, second, third, or fourth."
    )
    conversation_listener = [
        {
            "role": "<|User|>",
            "content": listener_prompt,
            "images": img_paths,
        },
        {"role": "<|Assistant|>", "content": ""}
    ]
    pil_images_listener = load_pil_images(conversation_listener)
    prepare_inputs_listener = processor(
        conversations=conversation_listener,
        images=pil_images_listener,
        force_batchify=True,
        system_prompt=""
    ).to(DEVICE)

    with torch.no_grad():
        inputs_embeds_listener = model.prepare_inputs_embeds(**prepare_inputs_listener)
        outputs_listener = model.generate(
            inputs_embeds=inputs_embeds_listener,
            attention_mask=prepare_inputs_listener.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True
        )

    listener_guess = tokenizer.decode(outputs_listener.sequences[0], skip_special_tokens=True).strip().lower()
    raw_listener_prompt = listener_prompt

    logits = outputs_listener.scores[0]
    probs_full = F.softmax(logits[0], dim=-1)

    answer_variants = get_answer_variants()
    variant_token_ids = get_all_variant_token_ids(tokenizer, answer_variants)
    cum_probs_table = get_cumulative_probs(probs_full, variant_token_ids)
    max_prob = max((prob for _, prob in cum_probs_table), default=0)

    correct = listener_guess == target_label
    result_text = "correct" if correct else "wrong"

    ranks = [cum_probs_table[i][1] if i < len(cum_probs_table) else 0 for i in range(4)]
    if correct:
        prob_correct.append(max_prob)
        rank1_probs_correct.append(ranks[0])
        rank2_probs_correct.append(ranks[1])
        rank3_probs_correct.append(ranks[2])
        rank4_probs_correct.append(ranks[3])
        correct_count += 1
    else:
        prob_wrong.append(max_prob)
        rank1_probs_wrong.append(ranks[0])
        rank2_probs_wrong.append(ranks[1])
        rank3_probs_wrong.append(ranks[2])
        rank4_probs_wrong.append(ranks[3])
        wrong_count += 1

    total_count += 1

    speaker_desc_records.append({
        "folder": os.path.basename(trial_path),
        "target_label": target_label,
        "target_img_name": target_img_name,
        "description": generated_desc,
        "listener_guess": listener_guess,
        "correct": correct
    })

    print("\n----------------------------------------")
    print(f"Trial: {os.path.basename(trial_path)}")
    print(f"Target image: {target_idx + 1} ({target_img_name})")
    print(f"Generated description: {generated_desc}")
    print(f"Listener guess: {listener_guess}")
    print(f"Result: {result_text}")
    print("----------------------------------------\n")

    with open(OUTPUT_FILE, "a") as f:
        f.write(f"=== TRIAL {trial_index} ===\n\n")
        f.write(f"FOLDER: {os.path.basename(trial_path)}\n")
        f.write(f"TARGET IMAGE: {target_img_name}, {target_label}\n\n")
        f.write("[DESCRIPTION GENERATED]\n")
        f.write(generated_desc + "\n\n")
        f.write("[LISTENER GUESS]: " + listener_guess + "\n")
        f.write("[RESULT]: " + result_text + "\n\n")
        f.write("[LISTENER NEXT-TOKEN PROBABILITIES]\n")
        f.write("Answer   | Cumulative Probability\n")
        for ans, prob in cum_probs_table:
            f.write(f"{ans:<8} | {prob:.4f}\n")
        f.write("\n")
        f.write("[RAW SPEAKER PROMPT]\n")
        f.write(raw_speaker_prompt + "\n\n")
        f.write("[RAW LISTENER PROMPT]\n")
        f.write(raw_listener_prompt + "\n\n")
        f.write("=" * 80 + "\n\n")

# ----------------------------
# Deterministic 4-pass navigation
# ----------------------------
if __name__ == "__main__":
    # === All-run statistics ===
    all_run_accuracies = []
    all_run_avg_lens = []
    all_run_std_lens = []
    all_run_total_counts = []
    all_run_correct_counts = []
    all_run_wrong_counts = []

    all_run_rank_stats = []  # list of dicts for each run

    NUM_RUNS = 10

    for run_idx in range(NUM_RUNS):
        print(f"\n########### STARTING RUN {run_idx + 1} / {NUM_RUNS} ###########\n")
        
        # Reset stats for this run
        correct_count = 0
        wrong_count = 0
        total_count = 0
        trial_index = 0
        total_description_length = 0
        description_lengths = []

        prob_correct = []
        prob_wrong = []

        rank1_probs_correct, rank2_probs_correct, rank3_probs_correct, rank4_probs_correct = [], [], [], []
        rank1_probs_wrong,   rank2_probs_wrong,   rank3_probs_wrong,   rank4_probs_wrong   = [], [], [], []

        speaker_desc_records = []

        all_folders = [d for d in sorted(os.listdir(DATASET_DIR)) if os.path.isdir(os.path.join(DATASET_DIR, d))]

        for target_idx in range(NUM_IMAGES):  # 0,1,2,3 (first, second, ...)
            print(f"\n======== CYCLE for TARGET INDEX {target_idx + 1} (RUN {run_idx + 1}) ========\n")
            for folder in all_folders:
                trial_path = os.path.join(DATASET_DIR, folder)
                run_trial(trial_path, target_idx)

        # Per-run summary/statistics
        accuracy = correct_count / total_count * 100 if total_count > 0 else 0
        avg_len = total_description_length / total_count if total_count > 0 else 0
        std_len = statistics.stdev(description_lengths) if len(description_lengths) > 1 else 0

        def _avgstd(L): return (statistics.mean(L) if L else 0, statistics.stdev(L) if len(L) > 1 else 0)
        rank_stats = {
            "rank1_correct": _avgstd(rank1_probs_correct),
            "rank2_correct": _avgstd(rank2_probs_correct),
            "rank3_correct": _avgstd(rank3_probs_correct),
            "rank4_correct": _avgstd(rank4_probs_correct),
            "rank1_wrong": _avgstd(rank1_probs_wrong),
            "rank2_wrong": _avgstd(rank2_probs_wrong),
            "rank3_wrong": _avgstd(rank3_probs_wrong),
            "rank4_wrong": _avgstd(rank4_probs_wrong),
        }

        # Append stats for all-run overview
        all_run_accuracies.append(accuracy)
        all_run_avg_lens.append(avg_len)
        all_run_std_lens.append(std_len)
        all_run_total_counts.append(total_count)
        all_run_correct_counts.append(correct_count)
        all_run_wrong_counts.append(wrong_count)
        all_run_rank_stats.append(rank_stats)

        # Per-run results already written to OUTPUT_FILE in your per-trial logic
        # Optionally, save run-specific .json if you want

    # ---- OVERALL ALL-RUNS SUMMARY ----
    overall_output = OUTPUT_FILE.replace(".txt", "_allruns_summary.txt")
    with open(overall_output, "w") as f:
        f.write("==== SUMMARY OVER 10 RUNS ====\n")
        f.write(f"Total runs: {NUM_RUNS}\n\n")
        f.write(f"Mean accuracy: {statistics.mean(all_run_accuracies):.2f} ± {statistics.stdev(all_run_accuracies):.2f}%\n")
        f.write(f"Mean desc len: {statistics.mean(all_run_avg_lens):.2f} ± {statistics.stdev(all_run_avg_lens):.2f} words\n")
        f.write(f"Mean std desc len: {statistics.mean(all_run_std_lens):.2f} ± {statistics.stdev(all_run_std_lens):.2f} words\n\n")
        f.write(f"Total trials (all runs): {sum(all_run_total_counts)}\n")
        f.write(f"Mean correct per run: {statistics.mean(all_run_correct_counts):.1f}\n")
        f.write(f"Mean wrong per run: {statistics.mean(all_run_wrong_counts):.1f}\n")
        f.write("\nPer-run accuracies:\n")
        for i, acc in enumerate(all_run_accuracies):
            f.write(f"Run {i+1}: {acc:.2f}%\n")

        f.write("\n==== Rank statistics per run ====\n")
        for i, rs in enumerate(all_run_rank_stats):
            f.write(f"\nRun {i+1}:\n")
            for key, val in rs.items():
                f.write(f"{key}: {val[0]:.4f} ± {val[1]:.4f}\n")

    print("ALL RUNS COMPLETE. See per-run output in your output file and all-runs summary at:", overall_output)
