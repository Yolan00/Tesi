import os
import statistics
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import json

# ----------------------------
# Configuration
# ----------------------------
SPEAKER_DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"
LISTENER_DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"
OUTPUT_FILE = "/projects/0/prjs1482/UvA/AA_PART_1/Baseline/Qwen/qwen_unc_1.txt"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 4
IMAGE_EXT = ".jpg"

# ----------------------------
# Variables
# ----------------------------
correct_count = 0
wrong_count = 0
total_count = 0
trial_index = 0
total_description_length = 0

prob_correct = []
prob_wrong = []

# For final summary
all_prob_correct = []
all_prob_wrong = []

# For description length statistics
description_lengths = []

# For ranked answer cumulative probabilities
rank1_probs_correct, rank2_probs_correct, rank3_probs_correct, rank4_probs_correct = [], [], [], []
rank1_probs_wrong,   rank2_probs_wrong,   rank3_probs_wrong,   rank4_probs_wrong   = [], [], [], []

# ----------------------------
# Load model and processor
# ----------------------------
print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()

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
    sorted_items = sorted(cum_probs.items(), key=lambda x: -x[1])
    return sorted_items

def get_topk_tokens(probs_full, tokenizer, k=5):
    topk = torch.topk(probs_full, k)
    ids = topk.indices.cpu().tolist()
    probs = topk.values.cpu().tolist()
    tokens = [tokenizer.decode([i]).strip() for i in ids]
    return list(zip(tokens, probs, ids))

# ----------------------------
# Main trial
# ----------------------------
def run_trial(speaker_trial_path, listener_trial_path, target_idx):
    global trial_index, correct_count, wrong_count, total_count, total_description_length
    global prob_correct, prob_wrong, description_lengths
    global rank1_probs_correct, rank2_probs_correct, rank3_probs_correct, rank4_probs_correct
    global rank1_probs_wrong, rank2_probs_wrong, rank3_probs_wrong, rank4_probs_wrong
    global speaker_desc_records

    trial_index += 1

    current_folder = os.path.basename(speaker_trial_path)
    speaker_img_paths = get_image_paths(speaker_trial_path)
    listener_img_paths = get_image_paths(listener_trial_path)
    if len(speaker_img_paths) != NUM_IMAGES or len(listener_img_paths) != NUM_IMAGES:
        print(f"Skipping {speaker_trial_path}: expected {NUM_IMAGES} images, found {len(speaker_img_paths)} (speaker) and {len(listener_img_paths)} (listener)")
        return

    csv_path = find_csv_file(listener_trial_path)
    if not csv_path:
        print(f"Skipping {listener_trial_path}: no suitable .csv file found")
        return

    speaker_images = load_images(speaker_img_paths)
    listener_images = load_images(listener_img_paths)
    target_label = ['first', 'second', 'third', 'fourth'][target_idx]
    target_img_name = os.path.basename(listener_img_paths[target_idx])

    # SPEAKER INSTRUCTION
    speaker_instruction = (
        f"You are playing a reference game as the speaker. I will give you 4 different images. "
        f"Generate a description of the {target_label} image so that the listener can identify it among the others, "
        f"DO NOT MENTION that the image is the {target_label}. Never use the words 'first', 'second', 'third', or 'fourth'. "
        f"To generate the description use the unique elements of the target image that distinguish it from the others. "
    )

    speaker_messages = [
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": speaker_instruction.strip()}] +
                [{"type": "text", "text": "\n### IMAGE SET: "}] +
                [
                    item
                    for i, img in enumerate(speaker_images)
                    for item in (
                        [{"type": "text", "text": f"IMAGE {i+1}:"}] +
                        [{"type": "image", "image": img}]
                    )
                ] +
                [{"type": "text", "text": f"TARGET: {target_label}"}] +
                [{"type": "text", "text": "DESCRIPTION:"}]
            )
        }
    ]

    speaker_text_prompt = processor.apply_chat_template(speaker_messages, tokenize=False, add_generation_prompt=True)
    speaker_image_inputs, speaker_video_inputs = process_vision_info(speaker_messages)

    speaker_inputs = processor(
        text=[speaker_text_prompt],
        images=speaker_image_inputs,
        videos=speaker_video_inputs,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    speaker_output_ids = model.generate(**speaker_inputs, max_new_tokens=80, do_sample=False)
    trimmed = speaker_output_ids[:, speaker_inputs.input_ids.shape[1]:]
    generated_desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    total_description_length += len(generated_desc.split())
    description_lengths.append(len(generated_desc.split()))
    generated_desc = "It's the first one"

    # LISTENER
    listener_prompt = (
        f"You are playing a reference game as the listener. Here is a description: \"{generated_desc}\"\n"
        f"Which of the four images does this description refer to?\n"
        f"Answer with one word: first, second, third, or fourth."
    )

    listener_messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in listener_images] +
                [{"type": "text", "text": listener_prompt}]
            )
        }
    ]

    listener_text_prompt = processor.apply_chat_template(listener_messages, tokenize=False, add_generation_prompt=True)
    listener_image_inputs, listener_video_inputs = process_vision_info(listener_messages)

    listener_inputs = processor(
        text=[listener_text_prompt],
        images=listener_image_inputs,
        videos=listener_video_inputs,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        listener_outputs = model.generate(
            **listener_inputs,
            max_new_tokens=10,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

    logits = listener_outputs.scores[0][:, :]
    probs_full = F.softmax(logits[0], dim=-1)

    trimmed_listener = listener_outputs.sequences[:, listener_inputs.input_ids.shape[1]:]
    listener_guess = processor.batch_decode(trimmed_listener, skip_special_tokens=True)[0].strip().lower()

    answer_variants = get_answer_variants()
    variant_token_ids = get_all_variant_token_ids(processor, answer_variants)
    cum_probs_table = get_cumulative_probs(probs_full, variant_token_ids)
    max_prob = max(prob for _, prob in cum_probs_table)

    correct = listener_guess == target_label
    result_text = "correct" if correct else "wrong"

    # Save the sorted answer cumulative probabilities (4 ranks) for this trial
    ranks = [cum_probs_table[i][1] if i < 4 else 0 for i in range(4)]
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
        "folder": current_folder,
        "target_label": target_label,
        "target_img_name": target_img_name,
        "description": generated_desc,
        "listener_guess": listener_guess,
        "correct": correct
    })

    print("\n----------------------------------------")
    print(f"Trial: {current_folder}")
    print(f"Target image: {target_idx + 1} ({target_img_name})")
    print(f"Generated description: {generated_desc}")
    print(f"Listener guess: {listener_guess}")
    print(f"Result: {result_text}")
    print("----------------------------------------\n")

    with open(OUTPUT_FILE, "a") as f:
        f.write(f"=== TRIAL {trial_index} ===\n\n")
        f.write(f"FOLDER: {current_folder}\n")
        f.write(f"TARGET IMAGE: {target_img_name}, {target_label}\n\n")
        f.write("[SPEAKER PROMPT]\n")
        f.write(speaker_instruction + "\n\n")
        f.write("[DESCRIPTION GENERATED]\n")
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

# ----------------------------
# Deterministic 4-pass navigation
# ----------------------------
speaker_desc_records = []

if __name__ == "__main__":
    speaker_folders = sorted(os.listdir(SPEAKER_DATASET_DIR))
    listener_folders = sorted(os.listdir(LISTENER_DATASET_DIR))

    for target_idx in range(NUM_IMAGES):  # 0,1,2,3 (first, second, ...)
        print(f"\n======== CYCLE for TARGET INDEX {target_idx + 1} ========\n")
        for folder in speaker_folders:
            speaker_trial_path = os.path.join(SPEAKER_DATASET_DIR, folder)
            listener_trial_path = os.path.join(LISTENER_DATASET_DIR, folder)
            if os.path.isdir(speaker_trial_path) and os.path.isdir(listener_trial_path):
                run_trial(speaker_trial_path, listener_trial_path, target_idx)

    # Summary/statistics as before
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    avg_len = total_description_length / total_count if total_count > 0 else 0

    all_prob_correct.extend(prob_correct)
    all_prob_wrong.extend(prob_wrong)

    avg_prob_correct = statistics.mean(all_prob_correct) if all_prob_correct else 0
    std_prob_correct = statistics.stdev(all_prob_correct) if len(all_prob_correct) > 1 else 0
    avg_prob_wrong = statistics.mean(all_prob_wrong) if all_prob_wrong else 0
    std_prob_wrong = statistics.stdev(all_prob_wrong) if len(all_prob_wrong) > 1 else 0

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

    with open(OUTPUT_FILE, "a") as f:
        f.write("=== OVERALL SUMMARY ===\n")
        f.write(f"Total trials: {total_count}\n")
        f.write(f"Correct:      {correct_count}\n")
        f.write(f"Wrong:        {wrong_count}\n")
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

    with open(OUTPUT_FILE.replace(".txt", "_descriptions.json"), "w") as jf:
        json.dump(speaker_desc_records, jf, indent=2)
