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
5. **Документирование:** подготовить подробное описание решения, README и презентацию.

## Ожидаемые результаты

| Результат | Статус |
|-----------|--------|
| EDA трёх датасетов с сохранением сводки | ✅ `eda.py`, `eda_output/eda_summary.json` |
| Скрипт QLoRA-обучения | ✅ `train_qlora.py` |
| Обученный LoRA-адаптер | ✅ `checkpoints/lora_adapter/` |
| Скрипт оценки (full + text-only) | ✅ `evaluate.py`, `evaluate_text_only.py` |
| Метрики base vs fine-tuned | ✅ `eval_results/summary.json` |
| Подробное описание решения | ✅ `SOLUTION.md` |
| Презентация | ✅ `presentation/` |
| README | ✅ этот файл |

## Использованные открытые данные VK

Все датасеты взяты из коллекции [deepvk/Vision-Language Modeling](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc):

| Датасет | Использование | Размер |
|---------|--------------|--------|
| [`deepvk/GQA-ru`](https://huggingface.co/datasets/deepvk/GQA-ru) | Обучение (`train_balanced_instructions`, 2000 примеров) + Оценка (`testdev_balanced_instructions`) | ~40k train / ~12.2k test |
| [`deepvk/MMBench-ru`](https://huggingface.co/datasets/deepvk/MMBench-ru) | Оценка (`dev`) | ~3.91k |
| [`deepvk/LLaVA-Instruct-ru`](https://huggingface.co/datasets/deepvk/LLaVA-Instruct-ru) | Изучен в EDA | ~144k |
| [`deepvk/llava-gemma-2b-lora`](https://huggingface.co/deepvk/llava-gemma-2b-lora) | Базовая модель для fine-tuning | 2B параметров |

## Базовая модель

**deepvk/llava-gemma-2b-lora** — визуально-языковая модель на основе LLaVA архитектуры с языковой моделью Gemma-2B. В модели уже используется LoRA, что делает её компактной (2B параметров) и эффективной для fine-tuning на потребительском GPU.

## Метод обучения

**QLoRA (Quantized Low-Rank Adaptation):**
- 4-битное квантование весов (NF4 + double quantization)
- LoRA-адаптеры: rank=16, alpha=32, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Optimizer: paged_adamw_8bit (memory-efficient)
- Effective batch size: 8 (batch=1 × grad_accum=8)
- Max steps: 500, LR=2e-5, cosine schedule, warmup=20 steps

## Структура проекта

```
vk-vlm-project/
├── README.md                   ← Описание проекта
├── SOLUTION.md                 ← Подробное описание решения
├── requirements.txt            ← Зависимости
├── eda.py                      ← EDA датасетов
├── eda_output/
│   └── eda_summary.json        ← Сводка EDA
├── train_qlora.py              ← QLoRA fine-tuning
├── evaluate.py                ← Оценка с изображениями
├── evaluate_text_only.py       ← Текстовая оценка (без изображений)
├── checkpoints/
│   ├── lora_adapter/           ← Обученный LoRA-адаптер
│   └── training_info.json      ← Параметры и метрики обучения
├── eval_results/
│   ├── base_gqa_results.json   ← Результаты базовой модели (GQA)
│   ├── base_mmbench_results.json
│   ├── finetuned_gqa_results.json
│   ├── finetuned_mmbench_results.json
│   └── summary.json            ← Сводка метрик
├── model_cache/                ← Локальная копия базовой модели
└── presentation/
    └── VLM_Project_Presentation.pptx
```

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
# Полная оценка (base + finetuned, GQA + MMBench)
python evaluate.py --mode both --benchmarks all --max-samples 200

# Только текстовая оценка
python evaluate_text_only.py --mode both --max-samples 300
```

## Аппаратные требования

- GPU: NVIDIA с ≥6 ГБ VRAM (тестировалось на RTX 4050 Laptop)
- RAM: ≥16 ГБ
- Дисковое пространство: ~10 ГБ (модель + датасеты)

## Результаты

| Модель | GQA-ru (accuracy) | MMBench-ru (accuracy) |
|--------|-------------------|----------------------|
| Base | 0.4700 (47/100) | 0.5700 (57/100) |
| Fine-tuned (QLoRA) | 0.4600 (46/100) | 0.5700 (57/100) |

**Train loss:** 6.34 (с 9.3 до 5.1 за 100 шагов)

Результаты оценки сохраняются в `eval_results/summary.json` после запуска `evaluate.py`.

## Ссылки

- [VK Vision-Language Modeling Collection](https://huggingface.co/collections/deepvk/vision-language-modeling-664dd7e4c257cc78e740f6bc)
- [Vision Language Models Explained](https://huggingface.co/blog/vlms)
- [GQA-ru](https://huggingface.co/datasets/deepvk/GQA-ru)
- [MMBench-ru](https://huggingface.co/datasets/deepvk/MMBench-ru)
- [llava-gemma-2b-lora](https://huggingface.co/deepvk/llava-gemma-2b-lora)