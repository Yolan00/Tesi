import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ----------------------------
# Configuration: paste your paths here
# ----------------------------
IMAGE_PATHS = [
    "/projects/0/prjs1482/UvA/dataset/000_Xf39GqLsVwUGe7ErHgdq44/COCO_val2014_000000334417.jpg",
    "/projects/0/prjs1482/UvA/dataset/000_Xf39GqLsVwUGe7ErHgdq44/COCO_val2014_000000334417.jpg",
    "/projects/0/prjs1482/UvA/dataset/000_Xf39GqLsVwUGe7ErHgdq44/COCO_val2014_000000334417.jpg",
    "/projects/0/prjs1482/UvA/dataset/000_Xf39GqLsVwUGe7ErHgdq44/COCO_val2014_000000334417.jpg",
]
TARGET_IDX = 0  # Index of the target image: 0, 1, 2, or 3

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "speaker_manual.txt"  # or None

assert len(IMAGE_PATHS) == 4, "You must provide exactly 4 image paths."
assert 0 <= TARGET_IDX < 4, "Target index must be 0, 1, 2, or 3."

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
# Load images
# ----------------------------
images = [Image.open(p).convert("RGB") for p in IMAGE_PATHS]
target_label = ['first', 'second', 'third', 'fourth'][TARGET_IDX]
target_img_name = IMAGE_PATHS[TARGET_IDX].split("/")[-1]

# ----------------------------
# SPEAKER prompt (unchanged)
# ----------------------------
speaker_instruction = (
    f"You are playing a reference game as the speaker. I will give you 4 different images. \n"
    f"Generate a description of the first image so that the listener can identify it among the others. \n"
    f"DO NOT MENTION that the image is the first. Never use the words 'first', 'second', 'third', or 'fourth'. \n "
    f"Your description must only contain the unique features of the target image. \n"
    f"and be formatted as follows: \"The target is the only one with...\". \n"
)

speaker_messages = [
    {
        "role": "user",
        "content": (
            [{"type": "text", "text": speaker_instruction.strip()}] +
            [{"type": "text", "text": "\n### IMAGE SET: "}] +
            [
                item
                for i, img in enumerate(images)
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

with torch.no_grad():
    speaker_output_ids = model.generate(**speaker_inputs, max_new_tokens=80, do_sample=False)
    trimmed = speaker_output_ids[:, speaker_inputs.input_ids.shape[1]:]
    generated_desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

print("\n----------------------------------------")
print(f"Target image: {TARGET_IDX + 1} ({target_img_name})")
print(f"Generated description: {generated_desc}")
print("----------------------------------------\n")

if OUTPUT_FILE:
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"TARGET IMAGE: {target_img_name}, {target_label}\n\n")
        f.write("[SPEAKER PROMPT]\n")
        f.write(speaker_instruction + "\n\n")
        f.write("[DESCRIPTION GENERATED]\n")
        f.write(generated_desc + "\n\n")
        f.write("=" * 80 + "\n\n")
