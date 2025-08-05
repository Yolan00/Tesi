import os
import json
from PIL import Image
import torch

from deepseek_vl2.models import DeepseekVLV2Processor
from transformers import AutoModelForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# ----------------------------
# Configuration
# ----------------------------
DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"
BLACK_IMAGE_PATH = "/projects/0/prjs1482/UvA/AA_DATASET/black.jpg"
OUTPUT_FILE = "/projects/0/prjs1482/UvA/Outputs/Side/deepseek_speakeronly_black.json"
MODEL_PATH = "deepseek-ai/deepseek-vl2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 4
IMAGE_EXT = ".jpg"

# ----------------------------
# Load model and processor
# ----------------------------
print("Loading DeepSeek-VL2...")
processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
tokenizer = processor.tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = model.to(torch.bfloat16).to(DEVICE).eval()

def get_image_paths(folder, ext=IMAGE_EXT):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)])

def load_images(image_paths):
    return [Image.open(p).convert("RGB") for p in image_paths]

# ----------------------------
# Main
# ----------------------------
speaker_desc_records = []

if __name__ == "__main__":
    all_folders = [d for d in sorted(os.listdir(DATASET_DIR)) if os.path.isdir(os.path.join(DATASET_DIR, d))]

    for folder in all_folders:
        trial_path = os.path.join(DATASET_DIR, folder)
        img_paths = get_image_paths(trial_path)
        if len(img_paths) != NUM_IMAGES:
            print(f"Skipping {trial_path}: expected {NUM_IMAGES} images, found {len(img_paths)}")
            continue

        # Target image: first image in folder
        target_img_path = img_paths[0]
        target_img_name = os.path.basename(target_img_path)

        # Build the image set: target + 3 black images
        image_set_paths = [target_img_path] + [BLACK_IMAGE_PATH] * (NUM_IMAGES - 1)
        image_set = load_images(image_set_paths)

        # Prompt
        speaker_prompt = (
            "<|grounding|> You are playing a reference game as the speaker. I will give you 4 different images. "
            "Image 1: <image>\n Image 2: <image>\n Image 3: <image>\n Image 4: <image>\n\n"
            "Generate a description of the FIRST image so that the listener can identify it among the others. "
            "DO NOT MENTION that the image is the first. Never use the words 'first', 'second', 'third', or 'fourth'. "
            "To generate the description use the unique elements of the target image that distinguish it from the others."
        )
        conversation = [
            {
                "role": "<|User|>",
                "content": speaker_prompt,
                "images": image_set_paths,
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

        speaker_desc_records.append({
            "folder": folder,
            "target_img_name": target_img_name,
            "description": generated_desc
        })

        print(f"{folder}: {generated_desc}")

    with open(OUTPUT_FILE, "w") as jf:
        json.dump(speaker_desc_records, jf, indent=2)
