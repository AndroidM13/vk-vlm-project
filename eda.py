"""
Exploratory Data Analysis (EDA) for VK Vision-Language datasets.
Analyzes: deepvk/GQA-ru, deepvk/MMBench-ru, deepvk/LLaVA-Instruct-ru
"""
import json
import os
import sys
from collections import Counter
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PYTHON = sys.executable
print(f"Python: {PYTHON}")

# ── 1. GQA-ru ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. GQA-ru — Analysis")
print("=" * 60)

# train_balanced_instructions (40k rows) — training split
gqa_train = load_dataset("deepvk/GQA-ru", "train_balanced_instructions", split="train", streaming=True)
gqa_train_items = []
for i, item in enumerate(gqa_train):
    gqa_train_items.append({k: str(v) for k, v in item.items()})
    if i >= 200:
        break

# testdev_balanced_instructions (12.2k rows) — eval split
gqa_test = load_dataset("deepvk/GQA-ru", "testdev_balanced_instructions", split="testdev", streaming=True)
gqa_test_items = []
for i, item in enumerate(gqa_test):
    gqa_test_items.append({k: str(v) for k, v in item.items()})
    if i >= 200:
        break

print(f"GQA-ru train sample size: {len(gqa_train_items)}")
print(f"GQA-ru testdev sample size: {len(gqa_test_items)}")
print(f"Fields: {list(gqa_train_items[0].keys())}")

# Answer distribution
answers = [item["answer"] for item in gqa_train_items]
answer_counts = Counter(answers)
print(f"\nTop-20 answers (train sample):")
for ans, cnt in answer_counts.most_common(20):
    print(f"  {ans}: {cnt}")

# Question length distribution
q_lengths = [len(item["question"]) for item in gqa_train_items]
print(f"\nQuestion length (chars): min={min(q_lengths)}, max={max(q_lengths)}, avg={sum(q_lengths)/len(q_lengths):.1f}")

# Types distribution
types = [item.get("types", "") for item in gqa_train_items]
type_counts = Counter(types)
print(f"\nTop-10 question types:")
for t, cnt in type_counts.most_common(10):
    print(f"  {t}: {cnt}")

# Sample examples
print("\nSample examples:")
for item in gqa_train_items[:5]:
    print(f"  Q: {item['question']}  A: {item['answer']}  Full: {item.get('fullAnswer', 'N/A')}")

# ── 2. MMBench-ru ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. MMBench-ru — Analysis")
print("=" * 60)

mmbench = load_dataset("deepvk/MMBench-ru", split="dev", streaming=True)
mmbench_items = []
for i, item in enumerate(mmbench):
    mmbench_items.append({k: str(v) for k, v in item.items() if k != "image"})
    if i >= 500:
        break

print(f"MMBench-ru sample size: {len(mmbench_items)}")
print(f"Fields: {list(mmbench_items[0].keys())}")

# Category distribution
categories = [item.get("category", "") for item in mmbench_items]
cat_counts = Counter(categories)
print(f"\nCategory distribution:")
for cat, cnt in cat_counts.most_common(20):
    print(f"  {cat}: {cnt}")

# L2-category distribution
l2_cats = [item.get("l2-category", "") for item in mmbench_items]
l2_counts = Counter(l2_cats)
print(f"\nL2-category distribution:")
for cat, cnt in l2_counts.most_common(20):
    print(f"  {cat}: {cnt}")

# Answer distribution (A/B/C/D)
answers_mb = [item.get("answer", "") for item in mmbench_items]
ans_counts = Counter(answers_mb)
print(f"\nAnswer distribution: {dict(ans_counts)}")

# Number of options per question
for item in mmbench_items[:5]:
    opts = {k: item[k] for k in ["A", "B", "C", "D"] if item.get(k, "nan") != "nan"}
    print(f"  Q: {item['question'][:80]}... Options: {len(opts)} Answer: {item['answer']}")

# ── 3. LLaVA-Instruct-ru ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. LLaVA-Instruct-ru — Analysis")
print("=" * 60)

instruct = load_dataset("deepvk/LLaVA-Instruct-ru", split="train", streaming=True)
instruct_items = []
for i, item in enumerate(instruct):
    instruct_items.append({k: str(v) for k, v in item.items() if k != "image"})
    if i >= 200:
        break

print(f"LLaVA-Instruct-ru sample size: {len(instruct_items)}")
print(f"Fields: {list(instruct_items[0].keys())}")

# Type distribution
types_inst = [item.get("type", "") for item in instruct_items]
type_counts_inst = Counter(types_inst)
print(f"\nType distribution:")
for t, cnt in type_counts_inst.most_common(20):
    print(f"  {t}: {cnt}")

# Conversation length
conv_lengths = [len(item.get("conversations", "")) for item in instruct_items]
print(f"\nConversation length (chars): min={min(conv_lengths)}, max={max(conv_lengths)}, avg={sum(conv_lengths)/len(conv_lengths):.1f}")

print("\nSample conversation:")
print(instruct_items[0].get("conversations", "")[:500])

# ── Save summary ───────────────────────────────────────────────────────
summary = {
    "GQA-ru": {
        "description": "Translated GQA benchmark for visual question answering in Russian",
        "splits": {
            "train_balanced_instructions": "~40k rows",
            "testdev_balanced_instructions": "~12.2k rows",
            "train_balanced_images": "~27.5k rows",
            "testdev_balanced_images": "~398 rows"
        },
        "fields": list(gqa_train_items[0].keys()),
        "sample_answers": dict(answer_counts.most_common(20)),
        "question_types": dict(type_counts.most_common(10)),
        "total_train_rows": 40000,
        "total_testdev_rows": 12200
    },
    "MMBench-ru": {
        "description": "Translated MMBench benchmark for multimodal understanding in Russian",
        "splits": {"dev": "~3.91k rows"},
        "fields": list(mmbench_items[0].keys()),
        "categories": dict(cat_counts.most_common(20)),
        "l2_categories": dict(l2_counts.most_common(20)),
        "answer_distribution": dict(ans_counts),
        "total_rows": 3910
    },
    "LLaVA-Instruct-ru": {
        "description": "GPT-based instruction dataset for LLaVA-style training in Russian",
        "splits": {"train": "~144k rows"},
        "fields": list(instruct_items[0].keys()),
        "types": dict(type_counts_inst.most_common(20)),
        "total_rows": 144000
    }
}

with open(os.path.join(OUTPUT_DIR, "eda_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n\nEDA summary saved to {OUTPUT_DIR}/eda_summary.json")
print("EDA complete!")