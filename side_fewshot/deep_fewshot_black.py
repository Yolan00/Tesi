import os
import json
import random
import pandas as pd
from PIL import Image
import torch

from deepseek_vl2.models import DeepseekVLV2Processor
from transformers import AutoModelForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# ----------------------------
# Configuration
# ----------------------------
DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"
OUTPUT_FILE = "/projects/0/prjs1482/UvA/AA_SIDE/side_fewshot/Results/deepseek_speakeronly_black.json"
MODEL_PATH = "deepseek-ai/deepseek-vl2"
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
        examples = []
        for _ in range(2):
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
            folder_pool.remove(example_folder)
        if len(examples) == 2:
            return examples
        else:
            return None
    except Exception as e:
        print(f"Error getting few-shot examples: {e}")
        return None

# ----------------------------
# Main speaker-only trial
# ----------------------------
def run_speaker_trial(trial_path, target_idx, model, processor, results_list):
    ordinal = ['first', 'second', 'third', 'fourth']
    current_folder = os.path.basename(trial_path)
    img_paths = get_image_paths(trial_path)
    if len(img_paths) != NUM_IMAGES:
        print(f"Skipping {trial_path}: expected {NUM_IMAGES} images, found {len(img_paths)}")
        return

    fewshot_examples = get_few_shot_examples(DATASET_DIR, os.path.basename(trial_path))
    if not fewshot_examples:
        print(f"Skipping {trial_path}: could not find enough few-shot examples.")
        return

    target_label = ordinal[target_idx]
    target_img_name = os.path.basename(img_paths[target_idx])

    # ===== Task images: all four are the target image =====
    target_img = Image.open(img_paths[target_idx]).convert("RGB")
    task_images = [target_img for _ in range(NUM_IMAGES)]

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

    # Compose PIL images: 4 for few-shot example, 4 for current trial (all target)
    few_shot_pil_images = load_images(fewshot_examples[0]['images'])
    pil_images = few_shot_pil_images + task_images

    # The conversation "images" field: first 4 are image paths, last 4 are just placeholders
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": fewshot_examples[0]['images'] + [None] * NUM_IMAGES,
        },
        {"role": "<|Assistant|>", "content": ""}
    ]

    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,           # Pass exactly 8 PIL images in correct order
        force_batchify=True,
        system_prompt=""
    ).to(DEVICE)

    with torch.no_grad():
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True
        )
    generated_desc = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()

    results_list.append({
        "folder": current_folder,
        "target_label": target_label,
        "target_img_name": target_img_name,
        "description": generated_desc,
        "raw_prompt": prompt
    })
    print(f"[Trial] Folder: {current_folder} | Target: {target_label} | Desc: {generated_desc}")

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    print("Loading DeepSeek-VL2...")
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = model.to(torch.bfloat16).to(DEVICE).eval()

    all_results = []
    all_folders = [d for d in sorted(os.listdir(DATASET_DIR)) if os.path.isdir(os.path.join(DATASET_DIR, d))]

    for target_idx in range(NUM_IMAGES):  # 0,1,2,3
        print(f"\n=== CYCLE for TARGET INDEX {target_idx + 1} ===\n")
        for folder in all_folders:
            trial_path = os.path.join(DATASET_DIR, folder)
            run_speaker_trial(trial_path, target_idx, model, processor, all_results)

    with open(OUTPUT_FILE, "w") as jf:
        json.dump(all_results, jf, indent=2)

    print("\nAll results saved to", OUTPUT_FILE)
