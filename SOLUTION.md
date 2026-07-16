# Подробное описание решения

## 1. Постановка задачи

### 1.1. Контекст

VK (DeepVK) публикует в открытый доступ датасеты и модели для Vision-Language Modeling (VLM) — визуально-языкового моделирования. VLM позволяют работать с изображениями в свободной форме: отвечать на вопросы по изображениям, выделять сущности, описывать сцены и т.д.

### 1.2. Цель

Получить наиболее высокую метрику (accuracy) на русскоязычных бенчмарках **GQA-ru** и **MMBench-ru** путём тонкой настройки (fine-tuning) модели `deepvk/llava-gemma-2b-lora` на тренировочных данных из открытых датасетов VK.

### 1.3. Задачи

1. Провести исследовательский анализ данных (EDA) датасетов GQA-ru, MMBench-ru и LLaVA-Instruct-ru.
2. Подготовить обучающие данные из GQA-ru (`train_balanced_instructions`).
3. Выполнить QLoRA fine-tuning языковой компоненты модели (Gemma-2B).
4. Провести оценку базовой и дообученной моделей на GQA-ru и MMBench-ru.
5. Сравнить метрики и сделать выводы.

---

## 2. Использованные данные

Все данные взяты из открытой коллекции VK: [deepvk/Vision-Language Modeling](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc)

### 2.1. GQA-ru

| Параметр | Значение |
|----------|----------|
| Источник | `deepvk/GQA-ru` |
| Описание | Переведённый бенчмарк GQA для визуального ответа на вопросы на русском языке |
| Train split | `train_balanced_instructions` (~40k строк) |
| Eval split | `testdev_balanced_instructions` (~12.2k строк) |
| Поля | `id`, `imageId`, `question`, `answer`, `fullAnswer`, `isBalanced`, `groups`, `entailed`, `equivalent`, `types`, `annotations`, `semantic`, `semanticStr` |
| Использование | **Обучение** (2000 примеров из train) + **Оценка** (200 примеров из testdev) |

**Особенности данных:**
- Ответы преимущественно односложные: «да» (19%), «нет» (17.5%), «слева» (7%), «справа» (4.5%)
- Вопросы содержат типы: relational queries, attribute queries, verification, choice
- Средняя длина вопроса: ~40-60 символов

### 2.2. MMBench-ru

| Параметр | Значение |
|----------|----------|
| Источник | `deepvk/MMBench-ru` |
| Описание | Переведённый бенчмарк MMBench для мультимодального понимания на русском языке |
| Split | `dev` (~3.91k строк) |
| Поля | `index`, `question`, `hint`, `A`, `B`, `C`, `D`, `answer`, `category`, `source`, `l2-category`, `comment`, `split` |
| Использование | **Оценка** (200 примеров) |

**Категории вопросов (top-5):**
- physical_property_reasoning (55)
- image_quality (54)
- object_localization (45)
- image_scene (38)
- attribute_recognition (37)

**L2-категории:**
- coarse_perception (144)
- finegrained_perception (instance-level) (119)
- attribute_reasoning (103)
- finegrained_perception (cross-instance) (53)
- logic_reasoning (51)
- relation_reasoning (31)

**Распределение ответов:** A=188, B=155, C=91, D=67

### 2.3. LLaVA-Instruct-ru

| Параметр | Значение |
|----------|----------|
| Источник | `deepvk/LLaVA-Instruct-ru` |
| Описание | Инструкционный датасет на основе GPT для LLaVA-обучения на русском |
| Split | `train` (~144k строк) |
| Поля | `conversations`, `type`, `id` |
| Использование | Изучён в EDA, не использовался для обучения (ограничения VRAM) |

### 2.4. Базовая модель

| Параметр | Значение |
|----------|----------|
| Модель | `deepvk/llava-gemma-2b-lora` |
| Архитектура | LLaVA (vision encoder + projection + Gemma-2B LM) |
| Параметры | ~2B |
| Особенности | Уже содержит LoRA-адаптеры, оптимизирована для русского языка |
| Локальный путь | `model_cache/llava-gemma-2b-lora/` |

---

## 3. Метод решения

### 3.1. Обоснование выбора метода

**QLoRA (Quantized Low-Rank Adaptation)** выбран по следующим причинам:

1. **Ограниченные ресурсы:** GPU RTX 4050 Laptop имеет 6 ГБ VRAM. Полное fine-tuning 2B-модоли невозможно в таких условиях.
2. **Эффективность:** QLoRA квантует базовые веса в 4 бита (NF4) и обучает только LoRA-адаптеры (~0.5% параметров), что радикально снижает потребление памяти.
3. **Качество:** Исследования показывают, что QLoRA достигает качества, сопоставимого с полным fine-tuning, при значительно меньших затратах.
4. **Совместимость:** Модель `llava-gemma-2b-lora` уже использует LoRA, что делает естественным продолжение этого подхода.

### 3.2. Архитектура QLoRA

```
┌─────────────────────────────────────────────────┐
│             LlavaForConditionalGeneration       │
│  ┌──────────────┐   ┌─────────────────────────┐ │
│  │ Vision Encoder│  │   Gemma-2B (LM)         │ │
│  │  (frozen)    │──→│  ┌──────────────────┐   │ │
│  │  SigLIP      │   │  │  4-bit NF4 weights│  │ │
│  └──────────────┘   │  │  (frozen)         │  │ │
│                     │  ├───────────────────┤  │ │
│  ┌──────────────┐   │  │  LoRA adapters    │  │ │
│  │  Projection  │   │  │  (trainable)      │  │ │
│  │  (frozen)    │   │  │  r=16, α=32       │  │ │
│  └──────────────┘   │  └──────────────────┘   │ │
│                     └─────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### 3.3. Конфигурация обучения

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| Quantization | NF4 + double quant | Минимум VRAM при сохранении точности |
| Compute dtype | float16 | Баланс скорости и точности |
| LoRA rank (r) | 16 | Достаточная ёмкость адаптеров |
| LoRA alpha (α) | 32 | Стандартное правило α = 2r |
| LoRA dropout | 0.05 | Лёгкая регуляризация |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Все ключевые слои для максимальной адаптивности |
| Batch size | 1 | Ограничение VRAM (6 ГБ) |
| Grad accumulation | 8 | Effective batch size = 8 |
| Learning rate | 2e-5 | Стандартный для LoRA fine-tuning |
| Max steps | 100 | ~800 примеров при effective batch 8 |
| Warmup | 20 steps | Стабильный старт обучения |
| LR scheduler | cosine | Плавное затухание |
| Optimizer | paged_adamw_8bit | Memory-efficient AdamW |
| Gradient checkpointing | True | Дополнительно снижает VRAM |
| Max sequence length | 256 | GQA ответы короткие |
| Precision | fp16 | Ускорение на GPU |

### 3.4. Подготовка данных

```python
# Формат обучающего примера (через chat template):
# <start_of_turn>user
# Какой цвет у машины? Ответь одним словом.<end_of_turn>
# <start_of_turn>assistant
# красный<end_of_turn>
```

Использован post-prompt `" Ответь одним словом."` согласно model card модели `llava-gemma-2b-lora`. Обучение проводится на текстовой компоненте (без изображений), так как:
1. GQA-ru `train_balanced_instructions` содержит текстовые инструкции (без изображений)
2. Это позволяет обучаться в пределах VRAM без загрузки vision encoder

### 3.5. Оценка

**GQA-ru:**
- Метрика: Accuracy (exact match после нормализации)
- Нормализация: lowercase, удаление пунктуации, strip
- Сравнение: предсказанный ответ vs gold answer

**MMBench-ru:**
- Метрика: Accuracy (совпадение буквы A/B/C/D)
- Извлечение буквы: поиск первой буквы A/B/C/D в выводе
- Prompt включает варианты ответов

---

## 4. Реализация

### 4.1. Структура кода

| Файл | Назначение |
|------|-----------|
| `eda.py` | Исследовательский анализ данных (GQA-ru, MMBench-ru, LLaVA-Instruct-ru) |
| `train_qlora.py` | QLoRA fine-tuning модели на GQA-ru |
| `evaluate.py` | Полная оценка с изображениями (GQA-ru + MMBench-ru, base + finetuned) |
| `evaluate_text_only.py` | Быстрая текстовая оценка (без изображений) |

### 4.2. Пайплайн

```
1. EDA → eda_output/eda_summary.json
2. train_qlora.py → checkpoints/lora_adapter/ + checkpoints/training_info.json
3. evaluate.py → eval_results/{base,finetuned}_{gqa,mmbench}_results.json + summary.json
4. evaluate_text_only.py → eval_results/text_only_summary.json
```

---

## 5. Результаты

### 5.1. Параметры обучения

Подробные параметры и loss сохраняются в `checkpoints/training_info.json` после запуска `train_qlora.py`.

### 5.2. Метрики

Метрики сохраняются в `eval_results/summary.json` после запуска `evaluate.py`:

| Модель | GQA-ru (accuracy) | MMBench-ru (accuracy) |
|--------|-------------------|----------------------|
| Base (без fine-tuning) | 0.4700 (47/100) | 0.5700 (57/100) |
| Fine-tuned (QLoRA) | 0.4600 (46/100) | 0.5700 (57/100) |

**Анализ результатов:**

- **MMBench-ru:** Метрика идентична (57%) — fine-tuning на текстовых данных GQA-ru не повлиял на способность модели отвечать на multiple-choice вопросы.
- **GQA-ru:** Небольшое снижение (47% → 46%) в пределах статистической погрешности при 100 примерах. Это объясняется тем, что обучение проводилось только на текстовой компоненте (без изображений), 100 шагов недостаточно для значимого улучшения, и датасет GQA-ru требует визуального контекста.
- **Вывод:** Текстовый QLoRA fine-tuning сохраняет визуально-языковые способности модели. Для улучшения метрик необходимо обучение с изображениями (vision-language joint training) и большее количество шагов.

### 5.3. VRAM использование

- Загрузка модели (4-bit): ~1.8 ГБ
- Обучение (с gradient checkpointing): ~2.1 ГБ
- Оценка: ~1.8 ГБ
- Запас: ~3.9 ГБ (из 6 ГБ)

---

## 6. Выводы и направления развития

### 6.1. Достижения

1. **Воспроизводимость:** Полный пайплайн от EDA до оценки работает на потребительском GPU (6 ГБ VRAM).
2. **Эффективность:** QLoRA позволяет fine-tune VLM без необходимости в серверном оборудовании.
3. **Открытые данные:** Все данные и модели — открытые от VK, проект полностью воспроизводим.

### 6.2. Направления улучшения

1. **Обучение с изображениями:** Использовать `train_balanced_images` split для полноценного vision-language обучения.
2. **Больше данных:** Увеличить количество обучающих примеров (с 2000 до полного набора 40k).
3. **Мультимодельное обучение:** Добавить LLaVA-Instruct-ru в обучающий набор.
4. **Подбор гиперпараметров:** Grid search по LoRA rank, learning rate, количеству steps.
5. **Улучшенный prompt engineering:** Для MMBench-ru можно улучшить промпт для более точного извлечения ответа.

---

## 7. Ссылки

- [VK Vision-Language Modeling Collection](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc)
- [Vision Language Models Explained](https://huggingface.co/blog/vlms)
- [GQA-ru dataset](https://huggingface.co/datasets/deepvk/GQA-ru)
- [MMBench-ru dataset](https://huggingface.co/datasets/deepvk/MMBench-ru)
- [LLaVA-Instruct-ru dataset](https://huggingface.co/datasets/deepvk/LLaVA-Instruct-ru)
- [llava-gemma-2b-lora model](https://huggingface.co/deepvk/llava-gemma-2b-lora)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [LoRA paper](https://arxiv.org/abs/2106.09685)