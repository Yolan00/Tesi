import os
import statistics
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import json
import random
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------
SPEAKER_DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"
LISTENER_DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"
OUTPUT_FILE = "/projects/0/prjs1482/UvA/Outputs/PART_1/Qwen/Few_Shots/qwen_fewshot_Finale.txt"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 4
IMAGE_EXT = ".jpg"
NUM_RUNS = 10

# ----------------------------
# Helper functions from baseline
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

def get_all_variant_token_ids(processor, answer_variants):
    variant_token_ids = {}
    for key, variants in answer_variants.items():
        token_ids = set()
        for v in variants:
            for prefix in ["", " "]:
                ids = processor.tokenizer.encode(prefix + v, add_special_tokens=False)
                if len(ids) == 1:
                    token_ids.add(ids[0])
        variant_token_ids[key] = token_ids
    return variant_token_ids

def get_cumulative_probs(probs_full, variant_token_ids):
    cum_probs = {}
    for ans, tids in variant_token_ids.items():
        prob = sum(float(probs_full[tid]) for tid in tids)
        cum_probs[ans] = prob
    return sorted(cum_probs.items(), key=lambda x: -x[1])

def get_few_shot_examples(base_dir, current_folder):
    try:
        folder_pool = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f != current_folder]
        example_folder = random.choice(folder_pool)
        example_path = os.path.join(base_dir, example_folder)
        example_image_paths = get_image_paths(example_path)
        csv_path = find_csv_file(example_path)
        if not csv_path or len(example_image_paths) != NUM_IMAGES:
            return None
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if len(df) < NUM_IMAGES:
            return None
        df_head = df.head(NUM_IMAGES)
        indices = random.sample(range(NUM_IMAGES), 2)
        idx1, idx2 = indices[0], indices[1]
        desc1 = df_head.iloc[idx1]['msg']
        desc2 = df_head.iloc[idx2]['msg']
        example_images = load_images(example_image_paths)
        return {
            "images": example_images,
            "ex1_idx": idx1, "ex1_desc": desc1, "ex1_img_name": os.path.basename(example_image_paths[idx1]),
            "ex2_idx": idx2, "ex2_desc": desc2, "ex2_img_name": os.path.basename(example_image_paths[idx2]),
            "folder": example_folder
        }
    except Exception as e:
        print(f"Error getting few-shot example: {e}")
        return None

# ----------------------------
# Main trial (slightly modified to be a function for reusability)
# ----------------------------
def run_trial(
    speaker_trial_path, listener_trial_path, target_idx,
    model, processor, OUTPUT_FILE, S_RUN_IDX, stats, speaker_desc_records
):
    ordinal = ['first', 'second', 'third', 'fourth']
    current_folder = os.path.basename(speaker_trial_path)
    speaker_img_paths = get_image_paths(speaker_trial_path)
    listener_img_paths = get_image_paths(listener_trial_path)
    if len(speaker_img_paths) != NUM_IMAGES:
        return

    examples = get_few_shot_examples(SPEAKER_DATASET_DIR, current_folder)
    if not examples:
        print(f"Skipping {current_folder}: Failed to load valid few-shot examples.")
        return

    speaker_images = load_images(speaker_img_paths)
    listener_images = load_images(listener_img_paths)
    target_label = ordinal[target_idx]
    target_img_name = os.path.basename(listener_img_paths[target_idx])

    speaker_instruction = (
        "You are playing a reference game as the speaker. I will give you 4 different images. "
        "Generate a description of the target image so that the listener can identify it among the others, "
        "DO NOT MENTION the position of the image (e.g., 'first', 'second'). "
        "Focus on the unique visual elements of the target image that distinguish it."
    )
    speaker_messages = [{"role": "user", "content": (
        [{"type": "text", "text": speaker_instruction.strip()}] +
        [{"type": "text", "text": "\n\n### EXAMPLE 1"}] +
        [item for i, img in enumerate(examples['images']) for item in ([{"type": "text", "text": f"IMAGE {i+1}:"}, {"type": "image", "image": img}])] +
        [{"type": "text", "text": f"TARGET EXAMPLE 1: {ordinal[examples['ex1_idx']]}"},
         {"type": "text", "text": f"DESCRIPTION EXAMPLE 1: {examples['ex1_desc']}"}] +
        [{"type": "text", "text": "\n\n### EXAMPLE 2"}] +
        [item for i, img in enumerate(examples['images']) for item in ([{"type": "text", "text": f"IMAGE {i+1}:"}, {"type": "image", "image": img}])] +
        [{"type": "text", "text": f"TARGET EXAMPLE 2: {ordinal[examples['ex2_idx']]}"},
         {"type": "text", "text": f"DESCRIPTION EXAMPLE 2: {examples['ex2_desc']}"}] +
        [{"type": "text", "text": "\n\n### NOW YOUR TURN"}] +
        [item for i, img in enumerate(speaker_images) for item in ([{"type": "text", "text": f"IMAGE {i+1}:"}, {"type": "image", "image": img}])] +
        [{"type": "text", "text": f"TARGET: {target_label}"},
         {"type": "text", "text": "DESCRIPTION:"}]
    )}]
    speaker_text_prompt = processor.apply_chat_template(speaker_messages, tokenize=False, add_generation_prompt=True)
    speaker_image_inputs, _ = process_vision_info(speaker_messages)
    speaker_inputs = processor(text=[speaker_text_prompt], images=speaker_image_inputs, return_tensors="pt", padding=True).to(DEVICE)
    speaker_output_ids = model.generate(**speaker_inputs, max_new_tokens=80, do_sample=False)
    trimmed = speaker_output_ids[:, speaker_inputs.input_ids.shape[1]:]
    generated_desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    stats['total_description_length'] += len(generated_desc.split())
    stats['description_lengths'].append(len(generated_desc.split()))

    listener_prompt = (
        f"You are playing a reference game as the listener. Here is a description: \"{generated_desc}\"\n"
        f"Which of the four images does this description refer to?\n"
        f"Answer with only one word: first, second, third, or fourth."
    )
    listener_messages = [{"role": "user", "content": ([{"type": "image", "image": img} for img in listener_images] + [{"type": "text", "text": listener_prompt}])}]
    listener_text_prompt = processor.apply_chat_template(listener_messages, tokenize=False, add_generation_prompt=True)
    listener_image_inputs, _ = process_vision_info(listener_messages)
    listener_inputs = processor(text=[listener_text_prompt], images=listener_image_inputs, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        listener_outputs = model.generate(**listener_inputs, max_new_tokens=10, do_sample=False, output_scores=True, return_dict_in_generate=True)

    logits = listener_outputs.scores[0][:, :]
    probs_full = F.softmax(logits[0], dim=-1)
    trimmed_listener = listener_outputs.sequences[:, listener_inputs.input_ids.shape[1]:]
    listener_guess = processor.batch_decode(trimmed_listener, skip_special_tokens=True)[0].strip().lower()

    answer_variants = get_answer_variants()
    variant_token_ids = get_all_variant_token_ids(processor, answer_variants)
    cum_probs_table = get_cumulative_probs(probs_full, variant_token_ids)

    correct = any(variant in listener_guess.split() or variant == listener_guess for variant in answer_variants.get(target_label, []))
    result_text = "correct" if correct else "wrong"
    ranks = [cum_probs_table[i][1] if i < 4 else 0 for i in range(4)]

    if correct:
        stats['correct_count'] += 1
        stats['rank1_probs_correct'].append(ranks[0])
        stats['rank2_probs_correct'].append(ranks[1])
        stats['rank3_probs_correct'].append(ranks[2])
        stats['rank4_probs_correct'].append(ranks[3])
    else:
        stats['wrong_count'] += 1
        stats['rank1_probs_wrong'].append(ranks[0])
        stats['rank2_probs_wrong'].append(ranks[1])
        stats['rank3_probs_wrong'].append(ranks[2])
        stats['rank4_probs_wrong'].append(ranks[3])

    stats['total_count'] += 1
    speaker_desc_records.append({"folder": current_folder, "target_label": target_label, "target_img_name": target_img_name,
                                "description": generated_desc, "listener_guess": listener_guess, "correct": correct})

    print(f"[Run {S_RUN_IDX+1}] Trial {stats['trial_index'] + 1} | Folder: {current_folder} | Target: {target_label} | Result: {result_text}")

    with open(OUTPUT_FILE, "a") as f:
        f.write(f"=== RUN {S_RUN_IDX+1} - TRIAL {stats['trial_index']+1} ===\n\n")
        f.write(f"FOLDER: {current_folder}\n")
        f.write(f"TARGET IMAGE: {target_img_name}, {target_label}\n\n")
        f.write("[FEW-SHOT EXAMPLES USED]\n")
        f.write(f"From Folder: {examples['folder']}\n")
        f.write(f"1. Target: {examples['ex1_img_name']} ({ordinal[examples['ex1_idx']]}), Desc: {examples['ex1_desc']}\n")
        f.write(f"2. Target: {examples['ex2_img_name']} ({ordinal[examples['ex2_idx']]}), Desc: {examples['ex2_desc']}\n\n")
        f.write("[GENERATED DESCRIPTION]\n")
        f.write(generated_desc + "\n\n")
        f.write("[LISTENER PROMPT]\n")
        f.write(listener_prompt + "\n\n")
        f.write(f"[LISTENER GUESS]: {listener_guess}\n")
        f.write(f"[RESULT]: {result_text}\n\n")
        f.write("[LISTENER NEXT-TOKEN PROBABILITIES]\n")
        f.write("Answer   | Cumulative Probability\n")
        for ans, prob in cum_probs_table:
            f.write(f"{ans:<8} | {prob:.4f}\n")
        f.write("\n")
        f.write("[RAW SPEAKER PROMPT]\n")
        f.write(speaker_text_prompt + "\n\n")
        f.write("=" * 80 + "\n\n")

    stats['trial_index'] += 1

# ----------------------------
# Main execution: 10 runs
# ----------------------------
if __name__ == "__main__":
    # Load model once
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    # For overall stats across runs
    all_accuracies = []
    all_avg_lens = []
    all_std_lens = []
    all_rank1_correct = []
    all_rank2_correct = []
    all_rank3_correct = []
    all_rank4_correct = []
    all_rank1_wrong = []
    all_rank2_wrong = []
    all_rank3_wrong = []
    all_rank4_wrong = []
    run_desc_records = []

    with open(OUTPUT_FILE, "w") as f:
        f.write("Starting Few-Shot Learning Experiment with 10 Runs...\n\n")

    for S_RUN_IDX in range(NUM_RUNS):
        print(f"\n==== RUN {S_RUN_IDX+1}/{NUM_RUNS} ====\n")
        # Reset stats for this run
        stats = {
            'correct_count': 0, 'wrong_count': 0, 'total_count': 0, 'trial_index': 0, 'total_description_length': 0,
            'description_lengths': [],
            'rank1_probs_correct': [], 'rank2_probs_correct': [], 'rank3_probs_correct': [], 'rank4_probs_correct': [],
            'rank1_probs_wrong': [], 'rank2_probs_wrong': [], 'rank3_probs_wrong': [], 'rank4_probs_wrong': []
        }
        speaker_desc_records = []

        speaker_folders = sorted([f for f in os.listdir(SPEAKER_DATASET_DIR) if os.path.isdir(os.path.join(SPEAKER_DATASET_DIR, f))])

        for target_idx in range(NUM_IMAGES):
            print(f"\n--- RUN {S_RUN_IDX+1} | CYCLE for TARGET INDEX {target_idx+1} ---\n")
            for folder in speaker_folders:
                speaker_trial_path = os.path.join(SPEAKER_DATASET_DIR, folder)
                listener_trial_path = os.path.join(LISTENER_DATASET_DIR, folder)
                if os.path.isdir(speaker_trial_path) and os.path.isdir(listener_trial_path):
                    run_trial(speaker_trial_path, listener_trial_path, target_idx, model, processor, OUTPUT_FILE, S_RUN_IDX, stats, speaker_desc_records)

        # --- Run Summary/statistics ---
        accuracy = stats['correct_count'] / stats['total_count'] * 100 if stats['total_count'] > 0 else 0
        avg_len = stats['total_description_length'] / stats['total_count'] if stats['total_count'] > 0 else 0
        std_len = statistics.stdev(stats['description_lengths']) if len(stats['description_lengths']) > 1 else 0
        def _avgstd(L): return (statistics.mean(L) if L else 0, statistics.stdev(L) if len(L) > 1 else 0)
        rank_stats = {
            "rank1_correct": _avgstd(stats['rank1_probs_correct']), "rank2_correct": _avgstd(stats['rank2_probs_correct']),
            "rank3_correct": _avgstd(stats['rank3_probs_correct']), "rank4_correct": _avgstd(stats['rank4_probs_correct']),
            "rank1_wrong": _avgstd(stats['rank1_probs_wrong']), "rank2_wrong": _avgstd(stats['rank2_probs_wrong']),
            "rank3_wrong": _avgstd(stats['rank3_probs_wrong']), "rank4_wrong": _avgstd(stats['rank4_probs_wrong']),
        }
        all_accuracies.append(accuracy)
        all_avg_lens.append(avg_len)
        all_std_lens.append(std_len)
        all_rank1_correct.append(rank_stats['rank1_correct'][0])
        all_rank2_correct.append(rank_stats['rank2_correct'][0])
        all_rank3_correct.append(rank_stats['rank3_correct'][0])
        all_rank4_correct.append(rank_stats['rank4_correct'][0])
        all_rank1_wrong.append(rank_stats['rank1_wrong'][0])
        all_rank2_wrong.append(rank_stats['rank2_wrong'][0])
        all_rank3_wrong.append(rank_stats['rank3_wrong'][0])
        all_rank4_wrong.append(rank_stats['rank4_wrong'][0])
        run_desc_records.append(speaker_desc_records)

        with open(OUTPUT_FILE, "a") as f:
            f.write(f"=== RUN {S_RUN_IDX+1} SUMMARY ===\n")
            f.write(f"Total trials: {stats['total_count']}\n")
            f.write(f"Correct:      {stats['correct_count']}\n")
            f.write(f"Wrong:        {stats['wrong_count']}\n")
            f.write(f"Accuracy:     {accuracy:.2f}%\n")
            f.write(f"Avg Desc Len: {avg_len:.2f} ± {std_len:.2f} words\n")
            f.write("Ranked answer probabilities (when correct):\n")
            f.write(f"    1st most probable: {rank_stats['rank1_correct'][0]:.4f} ± {rank_stats['rank1_correct'][1]:.4f}\n")
            f.write(f"    2nd most probable: {rank_stats['rank2_correct'][0]:.4f} ± {rank_stats['rank2_correct'][1]:.4f}\n")
            f.write(f"    3rd most probable: {rank_stats['rank3_correct'][0]:.4f} ± {rank_stats['rank3_correct'][1]:.4f}\n")
            f.write(f"    4th most probable: {rank_stats['rank4_correct'][0]:.4f} ± {rank_stats['rank4_correct'][1]:.4f}\n")
            f.write("Ranked answer probabilities (when wrong):\n")
            f.write(f"    1st most probable: {rank_stats['rank1_wrong'][0]:.4f} ± {rank_stats['rank1_wrong'][1]:.4f}\n")
            f.write(f"    2nd most probable: {rank_stats['rank2_wrong'][0]:.4f} ± {rank_stats['rank2_wrong'][1]:.4f}\n")
            f.write(f"    3rd most probable: {rank_stats['rank3_wrong'][0]:.4f} ± {rank_stats['rank3_wrong'][1]:.4f}\n")
            f.write(f"    4th most probable: {rank_stats['rank4_wrong'][0]:.4f} ± {rank_stats['rank4_wrong'][1]:.4f}\n")
            f.write("=" * 80 + "\n\n")

        # Save per-run descriptions as JSON
        with open(OUTPUT_FILE.replace(".txt", f"_descriptions_run{S_RUN_IDX+1}.json"), "w") as jf:
            json.dump(speaker_desc_records, jf, indent=2)

    # --- Final Overall Summary ---
    def _avgstd_final(L): return (statistics.mean(L) if L else 0, statistics.stdev(L) if len(L) > 1 else 0)
    final_acc = _avgstd_final(all_accuracies)
    final_len = _avgstd_final(all_avg_lens)
    final_std_len = _avgstd_final(all_std_lens)
    final_rank1_correct = _avgstd_final(all_rank1_correct)
    final_rank2_correct = _avgstd_final(all_rank2_correct)
    final_rank3_correct = _avgstd_final(all_rank3_correct)
    final_rank4_correct = _avgstd_final(all_rank4_correct)
    final_rank1_wrong = _avgstd_final(all_rank1_wrong)
    final_rank2_wrong = _avgstd_final(all_rank2_wrong)
    final_rank3_wrong = _avgstd_final(all_rank3_wrong)
    final_rank4_wrong = _avgstd_final(all_rank4_wrong)

    with open(OUTPUT_FILE, "a") as f:
        f.write("=== OVERALL SUMMARY ACROSS ALL RUNS ===\n")
        f.write(f"Accuracy:        {final_acc[0]:.2f}% ± {final_acc[1]:.2f}\n")
        f.write(f"Avg Desc Len:    {final_len[0]:.2f} ± {final_len[1]:.2f} words\n")
        f.write(f"Std Desc Len:    {final_std_len[0]:.2f} ± {final_std_len[1]:.2f} words\n")
        f.write("Ranked answer probabilities (when correct):\n")
        f.write(f"    1st most probable: {final_rank1_correct[0]:.4f} ± {final_rank1_correct[1]:.4f}\n")
        f.write(f"    2nd most probable: {final_rank2_correct[0]:.4f} ± {final_rank2_correct[1]:.4f}\n")
        f.write(f"    3rd most probable: {final_rank3_correct[0]:.4f} ± {final_rank3_correct[1]:.4f}\n")
        f.write(f"    4th most probable: {final_rank4_correct[0]:.4f} ± {final_rank4_correct[1]:.4f}\n")
        f.write("Ranked answer probabilities (when wrong):\n")
        f.write(f"    1st most probable: {final_rank1_wrong[0]:.4f} ± {final_rank1_wrong[1]:.4f}\n")
        f.write(f"    2nd most probable: {final_rank2_wrong[0]:.4f} ± {final_rank2_wrong[1]:.4f}\n")
        f.write(f"    3rd most probable: {final_rank3_wrong[0]:.4f} ± {final_rank3_wrong[1]:.4f}\n")
        f.write(f"    4th most probable: {final_rank4_wrong[0]:.4f} ± {final_rank4_wrong[1]:.4f}\n")
        f.write("=" * 80 + "\n\n")

    print("\nExperiment finished. Results saved to", OUTPUT_FILE)
