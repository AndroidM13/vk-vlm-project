"""
Quick text-only evaluation (no images) for GQA-ru.
This evaluates the LLM component only — useful for quick benchmarking
without loading image data.

For the full evaluation with images, use evaluate.py.
"""
import os
import json
import re
import torch
from datasets import load_dataset
from transformers import (
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


def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return s.strip()


def load_text_model(use_adapter=False):
    model_path = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else MODEL_NAME

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

    if use_adapter and os.path.exists(ADAPTER_DIR):
        print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def evaluate_gqa_text_only(model, tokenizer, max_samples=300):
    """Evaluate GQA-ru using text-only (no image) — tests LLM knowledge."""
    print("\n=== GQA-ru Text-Only Evaluation ===")
    ds = load_dataset("deepvk/GQA-ru", "testdev_balanced_instructions", split="testdev", streaming=True)

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(ds):
        if i >= max_samples:
            break

        question = item["question"]
        gold_answer = item["answer"]

        # Text-only: replace <image> with [image not available]
        messages = [
            {"role": "user", "content": f"[изображение недоступно]\n{question}{POST_PROMPT}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            pred = tokenizer.decode(generate_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            pred = f"[ERROR: {e}]"

        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(gold_answer)
        is_correct = pred_norm == gold_norm
        correct += is_correct
        total += 1

        results.append({"question": question, "gold": gold_answer, "pred": pred, "correct": is_correct})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{max_samples}] Accuracy: {correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nGQA-ru Text-Only Accuracy: {accuracy:.4f} ({correct}/{total})")
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "finetuned", "both"], default="base")
    parser.add_argument("--max-samples", type=int, default=300)
    args = parser.parse_args()

    all_results = {}

    if args.mode in ["base", "both"]:
        print("\n# Base Model Text-Only Evaluation")
        model, tokenizer = load_text_model(use_adapter=False)
        results = evaluate_gqa_text_only(model, tokenizer, args.max_samples)
        all_results["base_gqa_text_only"] = results["accuracy"]
        with open(os.path.join(RESULTS_DIR, "base_gqa_text_only.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        del model
        torch.cuda.empty_cache()

    if args.mode in ["finetuned", "both"]:
        print("\n# Fine-tuned Model Text-Only Evaluation")
        model, tokenizer = load_text_model(use_adapter=True)
        results = evaluate_gqa_text_only(model, tokenizer, args.max_samples)
        all_results["finetuned_gqa_text_only"] = results["accuracy"]
        with open(os.path.join(RESULTS_DIR, "finetuned_gqa_text_only.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        del model
        torch.cuda.empty_cache()

    print("\n=== SUMMARY ===")
    for k, v in all_results.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(RESULTS_DIR, "text_only_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()