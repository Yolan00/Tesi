import os
import statistics
import json
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
OUTPUT_FILE = "/projects/0/prjs1482/UvA/AA_PART_1/Baseline/DeepSeek/deepseek_baseline_output.txt"
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

    target_label = ['first', 'second', 'third', 'fourth'][target_idx]
    target_img_name = os.path.basename(img_paths[target_idx])

    # SPEAKER TURN
    speaker_prompt = (
        f"<|grounding|> You are playing a reference game as the speaker. I will give you 4 different images. "
        f"Image 1: <image>\n Image 2: <image>\n Image 3: <image>\n Image 4: <image>\n\n"
        f"Generate a description of the {target_label} image so that the listener can identify it among the others, "
        f"DO NOT MENTION that the image is the {target_label}. Never use the words 'first', 'second', 'third', or 'fourth'. "
        f"To generate the description use the unique elements of the target image that distinguish it from the others. "
    )
    conversation = [
        {
            "role": "<|User|>",
            "content": speaker_prompt,
            "images": img_paths,
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
    # The model's output includes the prompt, so we decode the full sequence and then strip special tokens and prompt.
    # The tokenizer handles removing the prompt part when skip_special_tokens=True
    generated_desc = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()


    total_description_length += len(generated_desc.split())
    description_lengths.append(len(generated_desc.split()))
    raw_speaker_prompt = speaker_prompt

    # --- REVISED AND OPTIMIZED LISTENER STEP ---
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

    # Use a single, efficient model call to get both the guess and probabilities
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
            output_scores=True,           # Ask for the scores (logits)
            return_dict_in_generate=True  # Get a structured output
        )

    # Extract results from the single, efficient call
    listener_guess = tokenizer.decode(outputs_listener.sequences[0], skip_special_tokens=True).strip().lower()
    raw_listener_prompt = listener_prompt

    # Get the probabilities from the scores of the VERY FIRST generated token
    logits = outputs_listener.scores[0]
    probs_full = F.softmax(logits[0], dim=-1)
    # --- END OF OPTIMIZED BLOCK ---

    answer_variants = get_answer_variants()
    variant_token_ids = get_all_variant_token_ids(tokenizer, answer_variants)
    cum_probs_table = get_cumulative_probs(probs_full, variant_token_ids)
    max_prob = max((prob for _, prob in cum_probs_table), default=0)

    correct = listener_guess == target_label
    result_text = "correct" if correct else "wrong"

    # Save the sorted answer cumulative probabilities (4 ranks) for this trial
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
    all_folders = [d for d in sorted(os.listdir(DATASET_DIR)) if os.path.isdir(os.path.join(DATASET_DIR, d))]

    for target_idx in range(NUM_IMAGES):  # 0,1,2,3 (first, second, ...)
        print(f"\n======== CYCLE for TARGET INDEX {target_idx + 1} ========\n")
        for folder in all_folders:
            trial_path = os.path.join(DATASET_DIR, folder)
            run_trial(trial_path, target_idx)

    # Summary/statistics
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

    print("DONE.")
