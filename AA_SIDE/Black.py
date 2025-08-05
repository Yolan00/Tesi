import os
import json
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Config
SPEAKER_DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset"
BLACK_IMAGE_PATH = "/projects/0/prjs1482/UvA/AA_DATASET/black.jpg"
OUTPUT_FILE = "/projects/0/prjs1482/UvA/Outputs/Side/qwen_BLACK_SPEAKER_ONLY.json"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 4
IMAGE_EXT = ".jpg"

# Load model and processor
print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()

def get_image_paths(folder, ext=IMAGE_EXT):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)])

def find_csv_file(folder):
    for f in os.listdir(folder):
        if f.endswith(".csv") and f.startswith("trials_for_"):
            return os.path.join(folder, f)
    return None

def load_images(image_paths):
    return [Image.open(p).convert("RGB") for p in image_paths]

# Main
speaker_desc_records = []

if __name__ == "__main__":
    folders = sorted(os.listdir(SPEAKER_DATASET_DIR))
    for folder in folders:
        folder_path = os.path.join(SPEAKER_DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        img_paths = get_image_paths(folder_path)
        if len(img_paths) != NUM_IMAGES:
            print(f"Skipping {folder_path}: expected {NUM_IMAGES} images, found {len(img_paths)}")
            continue

        # Target image (take first one for each trial)
        target_img_path = img_paths[0]
        target_img_name = os.path.basename(target_img_path)

        # Other images: all black.jpg
        image_set_paths = [target_img_path] + [BLACK_IMAGE_PATH] * (NUM_IMAGES - 1)
        image_set = load_images(image_set_paths)

        # Prompt
        speaker_instruction = (
            f"You are playing a reference game as the speaker. I will give you 4 different images. "
            f"Generate a description of the FIRST image so that the listener can identify it among the others, "
            f"DO NOT MENTION that the image is the first. Never use the words 'first', 'second', 'third', or 'fourth'. "
            f"To generate the description use the unique elements of the target image that distinguish it from the others."
        )
        speaker_messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": speaker_instruction.strip()}] +
                    [{"type": "text", "text": "\n### IMAGE SET: "}] +
                    [
                        item
                        for i, img in enumerate(image_set)
                        for item in (
                            [{"type": "text", "text": f"IMAGE {i+1}:"}] +
                            [{"type": "image", "image": img}]
                        )
                    ] +
                    [{"type": "text", "text": "TARGET: first"}] +
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

        with torch.no_grad():
            speaker_output_ids = model.generate(**speaker_inputs, max_new_tokens=80, do_sample=False)
            trimmed = speaker_output_ids[:, speaker_inputs.input_ids.shape[1]:]
            generated_desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        speaker_desc_records.append({
            "folder": folder,
            "target_img_name": target_img_name,
            "description": generated_desc
        })

        print(f"{folder}: {generated_desc}")

    with open(OUTPUT_FILE, "w") as jf:
        json.dump(speaker_desc_records, jf, indent=2)
