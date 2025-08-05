import os
import json
import random
import pandas as pd
import statistics
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ----------------------------
# Configuration
# ----------------------------
SPEAKER_DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"
OUTPUT_FILE = "/projects/0/prjs1482/UvA/AA_SIDE/side_fewshot/Results/qwen_speakeronly_targetclones.json"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 4
IMAGE_EXT = ".jpg"

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
# Main speaker-only trial
# ----------------------------
def run_speaker_trial(
    speaker_trial_path, target_idx, model, processor, results_list
):
    ordinal = ['first', 'second', 'third', 'fourth']
    current_folder = os.path.basename(speaker_trial_path)
    speaker_img_paths = get_image_paths(speaker_trial_path)
    if len(speaker_img_paths) != NUM_IMAGES:
        return

    examples = get_few_shot_examples(SPEAKER_DATASET_DIR, current_folder)
    if not examples:
        print(f"Skipping {current_folder}: Failed to load valid few-shot examples.")
        return

    # Prepare 4 copies of the target image
    target_image = Image.open(speaker_img_paths[target_idx]).convert("RGB")
    speaker_images = [target_image for _ in range(NUM_IMAGES)]

    target_label = ordinal[target_idx]
    target_img_name = os.path.basename(speaker_img_paths[target_idx])

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
    with torch.no_grad():
        speaker_output_ids = model.generate(**speaker_inputs, max_new_tokens=80, do_sample=False)
    trimmed = speaker_output_ids[:, speaker_inputs.input_ids.shape[1]:]
    generated_desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    results_list.append({
        "folder": current_folder,
        "target_label": target_label,
        "target_img_name": target_img_name,
        "description": generated_desc,
        "raw_prompt": speaker_text_prompt
    })
    print(f"[Trial] Folder: {current_folder} | Target: {target_label} | Desc: {generated_desc}")

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    all_results = []

    speaker_folders = sorted([f for f in os.listdir(SPEAKER_DATASET_DIR) if os.path.isdir(os.path.join(SPEAKER_DATASET_DIR, f))])

    for target_idx in range(NUM_IMAGES):
        print(f"\n=== CYCLE for TARGET INDEX {target_idx+1} ===\n")
        for folder in speaker_folders:
            speaker_trial_path = os.path.join(SPEAKER_DATASET_DIR, folder)
            if os.path.isdir(speaker_trial_path):
                run_speaker_trial(
                    speaker_trial_path, target_idx,
                    model, processor, all_results
                )

    with open(OUTPUT_FILE, "w") as jf:
        json.dump(all_results, jf, indent=2)

    print("\nAll results saved to", OUTPUT_FILE)
