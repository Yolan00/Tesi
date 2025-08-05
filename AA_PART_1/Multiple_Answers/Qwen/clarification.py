import os
import json
import statistics
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ----------------------------
# Configuration
# ----------------------------
DATASET_DIR = "/projects/0/prjs1482/UvA/AA_DATASET/dataset_blur_25"
JSON_BASELINE = "/projects/0/prjs1482/UvA/Outputs/Blur/qwen_BLUR25_output_descriptions.json"
OUTPUT_FILE = "/projects/0/prjs1482/UvA/Outputs/PART_1/Qwen/Multiple_Answers/qwen_BLUR25_feedback_output.txt"
OUTPUT_JSON = "/projects/0/prjs1482/UvA/Outputs/PART_1/Qwen/Multiple_Answers/qwen_BLUR25_feedback_output.json"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 4
IMAGE_EXT = ".jpg"
UNCERTAINTY_THRESHOLD = 0.78
PROB_SUBSET = 0.10

# ----------------------------
# Load model and processor
# ----------------------------
print("Loading Qwen2.5-VL-7B-Instruct...")
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
        prob = sum(float(probs_full[tid]) for tid in tids if tid < len(probs_full))
        cum_probs[ans] = prob
    sorted_items = sorted(cum_probs.items(), key=lambda x: -x[1])
    return sorted_items

def run_listener(desc, listener_img_paths):
    listener_images = load_images(listener_img_paths)
    listener_prompt = (
        f"You are playing a reference game as the listener. Here is a description: \"{desc}\"\n"
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
    return listener_guess, probs_full, listener_prompt, listener_text_prompt

# ----------------------------
# Main experiment
# ----------------------------
if __name__ == "__main__":
    # Carica risultati baseline
    with open(JSON_BASELINE, "r") as f:
        baseline_results = json.load(f)

    # Preparazione output
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    feedback_results = []

    answer_variants = get_answer_variants()
    variant_token_ids = get_all_variant_token_ids(processor, answer_variants)

    for trial in baseline_results:
        folder = trial["folder"]
        target_label = trial["target_label"]
        description = trial["description"]
        target_img_name = trial["target_img_name"]
        correct_first = trial["correct"]

        speaker_trial_path = os.path.join(DATASET_DIR, folder)
        img_paths = get_image_paths(speaker_trial_path)
        idx_target = ['first', 'second', 'third', 'fourth'].index(target_label)
        listener_guess, probs_full, listener_prompt, raw_listener_prompt = run_listener(description, img_paths)

        cum_probs_table = get_cumulative_probs(probs_full, variant_token_ids)
        max_prob = cum_probs_table[0][1]

        # Step 1: Annotazione del primo round
        trial_out = {
            "folder": folder,
            "target_label": target_label,
            "target_img_name": target_img_name,
            "first_round": {
                "description": description,
                "listener_guess": listener_guess,
                "max_prob": max_prob,
                "prob_table": cum_probs_table,
                "correct": listener_guess == target_label,
                "raw_listener_prompt": raw_listener_prompt,
            }
        }

        # OUTPUT TXT
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"=== TRIAL [{folder}] Target: {target_label} ({target_img_name}) ===\n")
            f.write("[ROUND 1]\n")
            f.write(f"Description: {description}\n")
            f.write(f"Listener guess: {listener_guess}\n")
            f.write(f"Probabilities: {cum_probs_table}\n")
            f.write(f"Result: {'correct' if listener_guess == target_label else 'wrong'}\n")
            f.write(f"Max prob: {max_prob:.4f}\n")

        # Step 2: Feedback round se necessario
        feedback_info = None
        if max_prob <= UNCERTAINTY_THRESHOLD:
            feedback_indices = [i for i, (ans, prob) in enumerate(cum_probs_table) if prob >= PROB_SUBSET]
            feedback_labels = [cum_probs_table[i][0] for i in feedback_indices]
            feedback_img_indices = [ ['first', 'second', 'third', 'fourth'].index(lbl) for lbl in feedback_labels ]
            if idx_target not in feedback_img_indices:
                feedback_img_indices.append(idx_target)

            # -- NEW LOGIC: always put the target image FIRST in subset
            if feedback_img_indices[0] != idx_target:
                tgt_idx_pos = feedback_img_indices.index(idx_target)
                feedback_img_indices[0], feedback_img_indices[tgt_idx_pos] = feedback_img_indices[tgt_idx_pos], feedback_img_indices[0]

            subset_imgs = [img_paths[i] for i in feedback_img_indices]
            # The target is now always IMAGE 1 (i.e., position 0)
            feedback_prompt = (
                "You are playing the second round of a reference game as the speaker.\n "
                f"The listener is still unsure about which image is the target and has selected {len(subset_imgs)} possible targets, asking you which one you are referring to.\n"
                "Your target is IMAGE 1.\n"
                "Given your previous description, generate a new description of IMAGE 1 so that the listener can identify it among the others.\n "
                "DO NOT mention the order or position. To generate the description, use the unique elements of IMAGE 1 that distinguish it from the other images.\n"
            )

            # Prepare feedback_messages with "Old Description" before DESCRIPTION:
            feedback_images = load_images(subset_imgs)
            feedback_messages = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "text", "text": feedback_prompt}] +
                        [{"type": "text", "text": "\n### IMAGE SET:"}] +
                        [
                            item
                            for i, img in enumerate(feedback_images)
                            for item in (
                                [{"type": "text", "text": f"IMAGE {i+1}:"}] +
                                [{"type": "image", "image": img}]
                            )
                        ] +
                        [{"type": "text", "text": f"Old Description: {description}"}] +
                        [{"type": "text", "text": "TARGET: IMAGE 1"}] +
                        [{"type": "text", "text": "DESCRIPTION:"}]
                    )
                }
            ]
            feedback_text_prompt = processor.apply_chat_template(feedback_messages, tokenize=False, add_generation_prompt=True)
            feedback_image_inputs, feedback_video_inputs = process_vision_info(feedback_messages)
            feedback_inputs = processor(
                text=[feedback_text_prompt],
                images=feedback_image_inputs,
                videos=feedback_video_inputs,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)

            feedback_output_ids = model.generate(**feedback_inputs, max_new_tokens=80, do_sample=False)
            trimmed = feedback_output_ids[:, feedback_inputs.input_ids.shape[1]:]
            feedback_desc = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

            # LISTENER 2nd round: riceve sempre TUTTE le 4 immagini
            listener_guess2, probs_full2, listener_prompt2, raw_listener_prompt2 = run_listener(feedback_desc, img_paths)
            cum_probs_table2 = get_cumulative_probs(probs_full2, variant_token_ids)
            max_prob2 = cum_probs_table2[0][1]
            feedback_correct = listener_guess2 == target_label

            feedback_info = {
                "feedback_prompt": feedback_prompt,
                "feedback_images": [os.path.basename(p) for p in subset_imgs],
                "feedback_desc": feedback_desc,
                "listener_guess": listener_guess2,
                "max_prob": max_prob2,
                "prob_table": cum_probs_table2,
                "correct": feedback_correct,
                "raw_feedback_prompt": feedback_text_prompt,
                "raw_listener_prompt": raw_listener_prompt2
            }
            trial_out["second_round"] = feedback_info

            # OUTPUT TXT ROUND 2
            with open(OUTPUT_FILE, "a") as f:
                f.write("[ROUND 2 - FEEDBACK]\n")
                f.write(f"Subset images: {[os.path.basename(p) for p in subset_imgs]}\n")
                f.write(f"Feedback prompt: {feedback_prompt}\n")
                f.write(f"RAW SPEAKER PROMPT:\n{feedback_text_prompt}\n")
                f.write(f"Generated feedback description: {feedback_desc}\n")
                f.write(f"Listener guess (round 2): {listener_guess2}\n")
                f.write(f"Probabilities (round 2): {cum_probs_table2}\n")
                f.write(f"Result (round 2): {'correct' if feedback_correct else 'wrong'}\n")
                f.write(f"Max prob (round 2): {max_prob2:.4f}\n")
                f.write("-"*80 + "\n")
        else:
            with open(OUTPUT_FILE, "a") as f:
                f.write("No feedback round triggered.\n")
                f.write("-"*80 + "\n")

        feedback_results.append(trial_out)

    # === STATS COLLECTION ===
    n_total = 0
    n_feedback = 0
    n_correct_r1 = 0
    n_correct_r2 = 0
    n_r1_to_r2_wrong = 0
    n_r1_to_r2_right = 0
    feedback_desc_lengths = []

    for trial in feedback_results:
        n_total += 1
        r1 = trial['first_round']
        r1_correct = r1['correct']
        n_correct_r1 += int(r1_correct)
        has_fb = 'second_round' in trial and trial['second_round'] is not None
        if has_fb:
            n_feedback += 1
            r2 = trial['second_round']
            r2_correct = r2['correct']
            n_correct_r2 += int(r2_correct)
            # Description length round 2
            if 'feedback_desc' in r2:
                feedback_desc_lengths.append(len(r2['feedback_desc'].split()))
            # Flips
            if r1_correct and not r2_correct:
                n_r1_to_r2_wrong += 1
            if (not r1_correct) and r2_correct:
                n_r1_to_r2_right += 1

    # Stats for round 2 description length
    if feedback_desc_lengths:
        avg_fb_len = statistics.mean(feedback_desc_lengths)
        std_fb_len = statistics.stdev(feedback_desc_lengths) if len(feedback_desc_lengths) > 1 else 0
    else:
        avg_fb_len = 0
        std_fb_len = 0

    # Round 2 accuracy (over only those trials with feedback)
    acc_r2 = 100 * n_correct_r2 / n_feedback if n_feedback else 0

    with open(OUTPUT_FILE, "a") as f:
        f.write("\n" + "="*28 + " FEEDBACK STATISTICS " + "="*28 + "\n")
        f.write(f"Total trials: {n_total}\n")
        f.write(f"Trials with feedback (round 2): {n_feedback} ({100*n_feedback/n_total:.2f}%)\n")
        f.write(f"Accuracy in round 2: {acc_r2:.2f}% ({n_correct_r2}/{n_feedback})\n")
        f.write(f"Round 2 correct: {n_correct_r2}\n")
        f.write(f"Round 2 wrong: {n_feedback - n_correct_r2}\n")
        f.write(f"Correct→Wrong flips: {n_r1_to_r2_wrong}\n")
        f.write(f"Wrong→Correct flips: {n_r1_to_r2_right}\n")
        f.write(f"Average feedback description length: {avg_fb_len:.2f} ± {std_fb_len:.2f} words\n")
        f.write("="*70 + "\n\n")


    # Salva json finale
    with open(OUTPUT_JSON, "w") as jf:
        json.dump(feedback_results, jf, indent=2)

    print("DONE. All results saved.")
