import json
import os
from pathlib import Path
from PIL import Image
import torch

from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from mplug_owl2.conversation import conv_templates
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# ===================== PATHS =====================
BASE_DIR = Path("/home/ee/btech/ee1221719/mmc-work/mPLUG-Owl/mPLUG-Owl2")
BENCH_DIR = BASE_DIR / "MMC-DATA" / "MMC-Benchmark"

TEXT_JSONL = BENCH_DIR / "mmc_benchmark_text.jsonl"
TEXT_IMG_DIR = BENCH_DIR / "images/"

OUTPUT_FILE = BASE_DIR / "pred_mmc_text1.jsonl"
# =================================================


def load_model():
    """Load tokenizer + mPLUG-Owl2 model + image processor."""
    print("[INFO] Loading mPLUG-Owl2 model...")
    model_path = "MAGAer13/mplug-owl2-llama2-7b"

    tokenizer, model, image_processor, ctx_len = load_pretrained_model(
        model_path,
        None,
        model_path,
        device="cuda:0"
    )

    model.eval()
    print("[INFO] Model loaded successfully.")
    return tokenizer, model, image_processor


def infer_one(tokenizer, model, image_processor, img_path, instruction):
    """Inference for True/False classification."""

    # ----------- Load & preprocess image -----------
    image = Image.open(img_path).convert("RGB")
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))

    img_tensor = process_images([image], image_processor)
    img_tensor = img_tensor.to(model.device, dtype=torch.float16)

    # ---------------- Build prompt ----------------
    query = (
        instruction.strip()
        + " Answer strictly with one word: 'true' or 'false'."
    )

    conv = conv_templates["mplug_owl2"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # ---------------- Tokenize ----------------
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    # -----------  IMPORTANT SAFETY TRUNCATION FIX -----------
    # Prevents 4096 sequence explosion → OOM
    if input_ids.shape[1] > 3500:
        input_ids = input_ids[:, -3500:]
    # ---------------------------------------------------------

    stopper = KeywordsStoppingCriteria([conv.sep2], tokenizer, input_ids)

    # ---------------- Run Generation ----------------
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        output_ids = model.generate(
            input_ids,
            images=img_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=32,
            stopping_criteria=[stopper],
        )

    # ---------------- Decode Output ----------------
    out = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Clean to only "true"/"false"
    out = out.lower()
    if "true" in out:
        return "true"
    if "false" in out:
        return "false"
    return out


def main():
    print("[INFO] Loading model...")
    tokenizer, model, image_processor = load_model()

    print(f"[INFO] Reading dataset: {TEXT_JSONL}")
    fout = open(OUTPUT_FILE, "w")

    count = 0

    with open(TEXT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            img_id = item["image_id"]
            instruction = item["instruction"]
            gt_label = item["label"]
            task = item.get("task", "unknown")     # ★ ADD TASK HERE ★

            img_path = TEXT_IMG_DIR / img_id

            pred = infer_one(tokenizer, model, image_processor, img_path, instruction)

            out = {
                "image_id": img_id,
                "instruction": instruction,
                "gt": gt_label,
                "pred": pred,
                "task": task,                      # ★ ADD TASK INTO OUTPUT JSON ★
                "correct": (pred.lower() == gt_label.lower()),
            }

            fout.write(json.dumps(out) + "\n")

            count += 1
            if count % 50 == 0:
                print(f"[INFO] Processed {count} samples...")

    fout.close()
    print("[DONE] Predictions saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

