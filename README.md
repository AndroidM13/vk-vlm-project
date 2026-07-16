# Vision-Language Model Fine-Tuning на датасетах VK

## О проекте

Данный проект направлен на тонкую настройку (fine-tuning) визуально-языковой модели **deepvk/llava-gemma-2b-lora** с использованием открытых датасетов VK: **GQA-ru**, **MMBench-ru** и **LLaVA-Instruct-ru**. Цель — повысить качество ответов модели на русскоязычные визуальные вопросы, оценивая результаты на бенчмарках GQA-ru и MMBench-ru.

## Цель проекта

Получить наиболее высокую метрику (accuracy) на бенчмарках **GQA-ru** и **MMBench-ru** путём QLoRA-тонкой настройки модели `deepvk/llava-gemma-2b-lora` на тренировочном подмножестве GQA-ru.

## Задачи

1. **Исследовательский анализ данных (EDA):** изучить структуру, состав и особенности датасетов GQA-ru, MMBench-ru и LLaVA-Instruct-ru.
2. **Подготовка обучающих данных:** отобрать и форматировать тренировочные примеры из GQA-ru (`train_balanced_instructions`) для instruction-tuning.
3. **QLoRA fine-tuning:** настроить языковую компоненту модели (Gemma-2B) с применением 4-битного квантования (NF4) и LoRA-адаптеров, чтобы уложиться в 6 ГБ VRAM (RTX 4050).
4. **Оценка моделей:** провести сравнительную оценку базовой и дообученной моделей на тестовых выборках GQA-ru и MMBench-ru.
5. **Документирование:** подготовить подробное описание решения, отчёт, README и презентацию.

## Ожидаемые результаты

| Результат | Статус |
|-----------|--------|
| EDA трёх датасетов с сохранением сводки | ✅ `eda.py`, `eda_output/eda_summary.json` |
| Скрипт QLoRA-обучения | ✅ `train_qlora.py` |
| Обученный LoRA-адаптер | ✅ `checkpoints/lora_adapter/` (78 МБ) |
| Скрипт оценки (full + text-only) | ✅ `evaluate.py`, `evaluate_text_only.py` |
| Метрики base vs fine-tuned | ✅ `eval_results/summary.json` |
| Подробное описание решения | ✅ `SOLUTION.md` |
| Отчёт с графиками | ✅ `Отчёт_VLM_Проект.docx` |
| Презентация | ✅ `VLM_Project_Presentation.pptx` |
| Графики для отчёта | ✅ `report_charts/` (7 PNG) |
| README | ✅ этот файл |

## Использованные открытые данные VK

Все датасеты взяты из коллекции [deepvk/Vision-Language Modeling](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc):

| Датасет | Использование | Размер |
|---------|--------------|--------|
| [`deepvk/GQA-ru`](https://huggingface.co/datasets/deepvk/GQA-ru) | Обучение (`train_balanced_instructions`, 2000 примеров) + Оценка (`testdev_balanced_instructions` + `testdev_balanced_images`) | ~40k train / ~12.2k test / 398 images |
| [`deepvk/MMBench-ru`](https://huggingface.co/datasets/deepvk/MMBench-ru) | Оценка (`dev`) | ~3.91k |
| [`deepvk/LLaVA-Instruct-ru`](https://huggingface.co/datasets/deepvk/LLaVA-Instruct-ru) | Изучен в EDA | ~144k |
| [`deepvk/llava-gemma-2b-lora`](https://huggingface.co/deepvk/llava-gemma-2b-lora) | Базовая модель для fine-tuning | 2B параметров |

## Базовая модель

**deepvk/llava-gemma-2b-lora** — визуально-языковая модель на основе LLaVA архитектуры (vision encoder SigLIP + projection + языковая модель Gemma-2B). В модели уже используется LoRA, что делает её компактной (2B параметров) и эффективной для fine-tuning на потребительском GPU.

## Метод обучения

**QLoRA (Quantized Low-Rank Adaptation):**
- 4-битное квантование весов (NF4 + double quantization)
- LoRA-адаптеры: rank=16, alpha=32, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Optimizer: paged_adamw_8bit (memory-efficient)
- Effective batch size: 8 (batch=1 × grad_accum=8)
- Max steps: 100, LR=2e-5, cosine schedule, warmup=20 steps
- Gradient checkpointing: включён
- VRAM: 2.09 ГБ из 6 ГБ
- Trainable params: 19.6M (0.78%) из 2.5B

## Структура проекта

```
vk-vlm-project/
├── README.md                           ← Описание проекта
├── SOLUTION.md                         ← Подробное описание решения
├── Отчёт_VLM_Проект.docx               ← Отчёт по проекту с графиками (15+ страниц)
├── VLM_Project_Presentation.pptx       ← Презентация (9 слайдов)
├── requirements.txt                    ← Зависимости
├── .gitignore                          ← Исключения для Git
├── eda.py                              ← EDA датасетов
├── train_qlora.py                      ← QLoRA fine-tuning
├── evaluate.py                         ← Оценка с изображениями (GQA-ru + MMBench-ru)
├── evaluate_text_only.py               ← Текстовая оценка (без изображений)
├── eda_output/
│   └── eda_summary.json                ← Сводка EDA
├── checkpoints/
│   ├── lora_adapter/                   ← Обученный LoRA-адаптер (78 МБ)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── tokenizer.json
│   │   └── ...
│   └── training_info.json              ← Параметры и метрики обучения
├── eval_results/
│   ├── summary.json                    ← Сводка метрик
│   ├── base_gqa_results.json           ← Base модель, GQA-ru (47/100)
│   ├── base_mmbench_results.json       ← Base модель, MMBench-ru (57/100)
│   ├── finetuned_gqa_results.json      ← Fine-tuned, GQA-ru (46/100)
│   ├── finetuned_mmbench_results.json  ← Fine-tuned, MMBench-ru (57/100)
│   ├── base_gqa_text_only.json         ← Base, text-only (0/100)
│   ├── finetuned_gqa_text_only.json    ← Fine-tuned, text-only (0/100)
│   └── text_only_summary.json          ← Сводка text-only оценки
├── report_charts/                      ← Графики для отчёта (7 PNG)
│   ├── training_loss.png               ← Кривая обучения
│   ├── accuracy_comparison.png         ← Сравнение Base vs Fine-tuned
│   ├── gqa_answer_dist.png             ← Распределение ответов GQA-ru
│   ├── mmbench_categories.png          ← Категории MMBench-ru
│   ├── vram_usage.png                  ← Использование VRAM
│   ├── trainable_params.png            ← Trainable vs Frozen параметров
│   └── error_analysis.png              ← Анализ ошибок
└── model_cache/                        ← Локальная копия базовой модели (не в Git, 11 ГБ)
```

## Результаты

| Модель | GQA-ru (accuracy) | MMBench-ru (accuracy) |
|--------|-------------------|----------------------|
| Base | 0.4700 (47/100) | 0.5700 (57/100) |
| Fine-tuned (QLoRA) | 0.4600 (46/100) | 0.5700 (57/100) |

**Обучение:**
- Train loss: 6.34 (с 9.3 до 5.1 за 100 шагов)
- VRAM: 2.09 ГБ из 6 ГБ (RTX 4050 Laptop)
- Trainable params: 19.6M (0.78%) из 2.5B
- Train samples: 2000 из GQA-ru

**Анализ:**
- MMBench-ru: метрика идентична (57%) — fine-tuning сохранил мультимодальные способности
- GQA-ru: 47% → 46% — в пределах статистической погрешности при 100 примерах
- Text-only: 0% для обеих моделей — подтверждает, что модель использует изображения
- Для значимого улучшения метрик необходимо обучение с изображениями и больше шагов

Результаты оценки сохраняются в `eval_results/summary.json` после запуска `evaluate.py`.

## Запуск

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. EDA
```bash
python eda.py
```

### 3. Обучение
```bash
python train_qlora.py
```

### 4. Оценка
```bash
# Полная оценка (base + finetuned, GQA + MMBench) с изображениями
python evaluate.py --mode both --benchmarks all --max-samples 100

# Только текстовая оценка (без изображений)
python evaluate_text_only.py --mode both --max-samples 100
```

## Аппаратные требования

- GPU: NVIDIA с ≥6 ГБ VRAM (тестировалось на RTX 4050 Laptop)
- RAM: ≥16 ГБ
- Дисковое пространство: ~11 ГБ (модель + датасеты + чекпоинты)
- Python: 3.12
- CUDA: 12.6

## Ссылки

- [VK Vision-Language Modeling Collection](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc)
- [Vision Language Models Explained](https://huggingface.co/blog/vlms)
- [GQA-ru](https://huggingface.co/datasets/deepvk/GQA-ru)
- [MMBench-ru](https://huggingface.co/datasets/deepvk/MMBench-ru)
- [LLaVA-Instruct-ru](https://huggingface.co/datasets/deepvk/LLaVA-Instruct-ru)
- [llava-gemma-2b-lora](https://huggingface.co/deepvk/llava-gemma-2b-lora)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [LoRA paper](https://arxiv.org/abs/2106.09685)