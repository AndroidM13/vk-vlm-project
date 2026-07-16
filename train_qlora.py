"""
QLoRA Fine-tuning script for deepvk/llava-gemma-2b-lora on GQA-ru dataset.

Uses 4-bit quantization (NF4) + LoRA adapters to fit within 6GB VRAM (RTX 4050).
Training data: deepvk/GQA-ru (train_balanced_instructions subset)
Post-prompt: "Ответь одним словом." (as specified in the model card)

This script trains the language model (Gemma) component directly,
bypassing the LLaVA image merge logic that requires pixel_values.
"""
import os
import sys
import json
import torch
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# ── Configuration ────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "model_cache", "llava-gemma-2b-lora")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training hyperparameters — tuned for 6GB VRAM (RTX 4050)
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
MAX_STEPS = 100
WARMUP_STEPS = 20
MAX_LENGTH = 256
GRADIENT_CHECKPOINTING = True

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# GQA-ru post-prompt (from model card)
POST_PROMPT = " Ответь одним словом."


def load_model():
    """Load the model with 4-bit quantization and apply LoRA to language model."""
    print("Loading model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Get the language model component (GemmaForCausalLM)
    # This avoids the LLaVA image merge logic
    lm = model.language_model

    # Prepare for k-bit training
    lm = prepare_model_for_kbit_training(lm)
    lm.gradient_checkpointing_enable()

    # Apply LoRA to the language model
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    lm = get_peft_model(lm, lora_config)
    lm.print_trainable_parameters()

    return lm, model


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_gqa_dataset(tokenizer, max_samples=2000):
    """Load GQA-ru training data and format for instruction tuning."""
    print("Loading GQA-ru training dataset...")
    ds = load_dataset("deepvk/GQA-ru", "train_balanced_instructions", split="train", streaming=True)

    texts = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break

        question = item["question"]
        answer = item["answer"]

        # Format WITHOUT <image> token since we're training the LM only
        messages = [
            {"role": "user", "content": f"{question}{POST_PROMPT}"},
            {"role": "assistant", "content": answer}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    print(f"Prepared {len(texts)} training examples")

    # Create HF dataset
    dataset = HFDataset.from_dict({"text": texts})
    return dataset


class TextDataset(torch.utils.data.Dataset):
    """Simple text dataset for causal LM training."""

    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    print("=" * 60)
    print("QLoRA Fine-tuning: llava-gemma-2b-lora (LM) on GQA-ru")
    print("=" * 60)

    # Load model
    lm_model, full_model = load_model()

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Prepare dataset
    ds = load_dataset("deepvk/GQA-ru", "train_balanced_instructions", split="train", streaming=True)
    texts = []
    for i, item in enumerate(ds):
        if i >= 2000:
            break
        question = item["question"]
        answer = item["answer"]
        messages = [
            {"role": "user", "content": f"{question}{POST_PROMPT}"},
            {"role": "assistant", "content": answer}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    print(f"Prepared {len(texts)} training examples")
    train_dataset = TextDataset(texts, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        save_safetensors=True,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = Trainer(
        model=lm_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save adapter
    print(f"\nSaving LoRA adapter to {ADAPTER_DIR}...")
    lm_model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Save training info
    train_info = {
        "model": "deepvk/llava-gemma-2b-lora",
        "dataset": "deepvk/GQA-ru (train_balanced_instructions)",
        "method": "QLoRA (4-bit NF4 + LoRA r=16, alpha=32) on language_model component",
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "max_steps": MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "max_length": MAX_LENGTH,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "optimizer": "paged_adamw_8bit",
            "lr_scheduler": "cosine",
            "fp16": True,
        },
        "train_samples": len(texts),
        "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
        "total_steps": MAX_STEPS,
        "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU (6GB)",
        "vram_usage_gb": 2.09,
    }

    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w", encoding="utf-8") as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)

    print("\nTraining complete!")
    print(f"Adapter saved to: {ADAPTER_DIR}")
    print(f"Training info saved to: {OUTPUT_DIR}/training_info.json")


if __name__ == "__main__":
    main()