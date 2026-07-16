"""
Evaluation script for fine-tuned llava-gemma-2b-lora on GQA-ru and MMBench-ru.

Evaluates both the base model and the fine-tuned model (with LoRA adapter)
to compare performance metrics.

GQA-ru: Accuracy metric (exact match on answers)
MMBench-ru: Accuracy metric (multiple choice A/B/C/D)
"""
import os
import sys
import json
import re
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel

MODEL_NAME = "deepvk/llava-gemma-2b-lora"
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_cache", "llava-gemma-2b-lora")
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "lora_adapter")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "eval_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

POST_PROMPT = " Ответь одним словом."

# GQA evaluation: normalize and compare
def normalize_answer(s):
    """Normalize answer string for comparison."""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = s.strip()
    return s

def evaluate_gqa_ru(model, tokenizer, processor, max_samples=200, use_adapter=False):
    """
    Evaluate on GQA-ru testdev.
    Joins testdev_balanced_instructions (questions/answers) with
    testdev_balanced_images (images) by imageId == id.
    Metric: accuracy (exact match after normalization).
    """
    print("\n" + "=" * 60)
    print("Evaluating on GQA-ru (testdev, with images)")
    print("=" * 60)

    # Load images split and build id->image lookup
    print("Loading images from testdev_balanced_images...")
    img_ds = load_dataset("deepvk/GQA-ru", "testdev_balanced_images", split="testdev", streaming=True)
    image_lookup = {}
    img_count = 0
    for item in img_ds:
        image_lookup[item["id"]] = item["image"]
        img_count += 1
        if img_count >= 400:  # there are ~398 images
            break
    print(f"Loaded {len(image_lookup)} images")

    # Load instructions split
    ds = load_dataset("deepvk/GQA-ru", "testdev_balanced_instructions", split="testdev", streaming=True)

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(ds):
        if i >= max_samples:
            break

        question = item["question"]
        gold_answer = item["answer"]
        image_id = item["imageId"]
        image = image_lookup.get(image_id)

        if image is None:
            # Skip if no image found
            continue

        # Format prompt
        messages = [
            {"role": "user", "content": f"<image>\n{question}{POST_PROMPT}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            if image is not None:
                inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)
            else:
                inputs = processor(text=text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

            pred_answer = tokenizer.decode(generate_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            pred_answer = f"[ERROR: {e}]"

        pred_norm = normalize_answer(pred_answer)
        gold_norm = normalize_answer(gold_answer)

        is_correct = pred_norm == gold_norm
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "gold": gold_answer,
            "pred": pred_answer,
            "correct": is_correct
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{max_samples}] Accuracy so far: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nGQA-ru Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def evaluate_mmbench_ru(model, tokenizer, processor, max_samples=200, use_adapter=False):
    """
    Evaluate on MMBench-ru dev.
    Metric: accuracy (exact match on A/B/C/D).
    """
    print("\n" + "=" * 60)
    print("Evaluating on MMBench-ru (dev)")
    print("=" * 60)

    ds = load_dataset("deepvk/MMBench-ru", split="dev", streaming=True)

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(ds):
        if i >= max_samples:
            break

        question = item["question"]
        hint = item.get("hint", "")
        options = {k: item[k] for k in ["A", "B", "C", "D"] if item.get(k, "nan") != "nan" and str(item.get(k, "nan")) != "nan"}
        gold_answer = item["answer"]
        image = item.get("image")

        # Build prompt with options
        prompt_text = f"<image>\n{question}"
        if hint and str(hint) != "nan":
            prompt_text = f"<image>\n{hint}\n{question}"

        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        prompt_text += f"\n{options_text}\nОтветь буквой правильного варианта (A, B, C или D)."

        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            if image is not None:
                inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)
            else:
                inputs = processor(text=text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

            pred = tokenizer.decode(generate_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            pred = f"[ERROR: {e}]"

        # Extract letter from prediction
        pred_letter = None
        for letter in ["A", "B", "C", "D"]:
            if pred.upper().startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            # Try to find any letter in the output
            match = re.search(r'[ABCD]', pred.upper())
            if match:
                pred_letter = match.group()

        is_correct = pred_letter == gold_answer
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:100],
            "gold": gold_answer,
            "pred": pred,
            "pred_letter": pred_letter,
            "correct": is_correct
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{max_samples}] Accuracy so far: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nMMBench-ru Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def load_model(use_adapter=False, use_4bit=True):
    """Load model with optional LoRA adapter and 4-bit quantization."""
    model_path = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else MODEL_NAME

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    if use_adapter and os.path.exists(ADAPTER_DIR):
        print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
        # Adapter was trained on the language_model component (Gemma-2B),
        # not on the full LLaVA model. Apply PeftModel to model.language_model.
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(
            model.language_model, ADAPTER_DIR
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    return model, tokenizer, processor


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "finetuned", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--benchmarks", choices=["gqa", "mmbench", "all"], default="all")
    args = parser.parse_args()

    all_results = {}

    # ── Evaluate base model ──
    if args.mode in ["base", "both"]:
        print("\n" + "#" * 60)
        print("# BASE MODEL EVALUATION")
        print("#" * 60)
        model, tokenizer, processor = load_model(use_adapter=False, use_4bit=True)

        if args.benchmarks in ["gqa", "all"]:
            gqa_results = evaluate_gqa_ru(model, tokenizer, processor, args.max_samples)
            all_results["base_gqa_ru"] = gqa_results["accuracy"]

        if args.benchmarks in ["mmbench", "all"]:
            mmbench_results = evaluate_mmbench_ru(model, tokenizer, processor, args.max_samples)
            all_results["base_mmbench_ru"] = mmbench_results["accuracy"]

        # Save detailed results
        if args.benchmarks in ["gqa", "all"]:
            with open(os.path.join(RESULTS_DIR, "base_gqa_results.json"), "w", encoding="utf-8") as f:
                json.dump(gqa_results, f, ensure_ascii=False, indent=2)
        if args.benchmarks in ["mmbench", "all"]:
            with open(os.path.join(RESULTS_DIR, "base_mmbench_results.json"), "w", encoding="utf-8") as f:
                json.dump(mmbench_results, f, ensure_ascii=False, indent=2)

        del model
        torch.cuda.empty_cache()

    # ── Evaluate fine-tuned model ──
    if args.mode in ["finetuned", "both"]:
        print("\n" + "#" * 60)
        print("# FINE-TUNED MODEL EVALUATION")
        print("#" * 60)
        model, tokenizer, processor = load_model(use_adapter=True, use_4bit=True)

        if args.benchmarks in ["gqa", "all"]:
            gqa_results_ft = evaluate_gqa_ru(model, tokenizer, processor, args.max_samples)
            all_results["finetuned_gqa_ru"] = gqa_results_ft["accuracy"]

        if args.benchmarks in ["mmbench", "all"]:
            mmbench_results_ft = evaluate_mmbench_ru(model, tokenizer, processor, args.max_samples)
            all_results["finetuned_mmbench_ru"] = mmbench_results_ft["accuracy"]

        if args.benchmarks in ["gqa", "all"]:
            with open(os.path.join(RESULTS_DIR, "finetuned_gqa_results.json"), "w", encoding="utf-8") as f:
                json.dump(gqa_results_ft, f, ensure_ascii=False, indent=2)
        if args.benchmarks in ["mmbench", "all"]:
            with open(os.path.join(RESULTS_DIR, "finetuned_mmbench_results.json"), "w", encoding="utf-8") as f:
                json.dump(mmbench_results_ft, f, ensure_ascii=False, indent=2)

        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, val in all_results.items():
        print(f"  {key}: {val:.4f}")

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()