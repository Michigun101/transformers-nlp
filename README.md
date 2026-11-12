# Transformers — курс по трансформерам и NLP

Учебный проект: 14 дней работы с библиотекой Hugging Face `transformers`.
Каждый день — отдельный самодостаточный файл `transformers_dayNN.py`.

## Прогресс

- [x] День 1 — архитектура трансформеров и токенизация (`transformers_day01.py`)

## Установка

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate
pip install -r requirements.txt
```

Токенизатор и модели скачиваются с huggingface.co при первом запуске и кэшируются
(`~/.cache/huggingface`).

## Запуск

```bash
python transformers_day01.py
```

## День 1 — токенизация

Что делает файл:

| Задача | Что показывает |
|---|---|
| 2 | Загрузка `AutoTokenizer`, `vocab_size`, `model_max_length` |
| 3 | Токенизация одного текста, `input_ids`, `decode` |
| 4 | `tokenize_texts()` — батч с padding/truncation, `attention_mask` |
| 5 | Специальные токены `[CLS]`, `[SEP]`, `[PAD]` |
| 6 | `explain_tokenization()` — текст → токены → ID |

Модель: `distilbert-base-uncased` (для русского — `distilbert-base-multilingual-cased`).

### Ключевые понятия

- **Токен** — кусочек текста (слово или его часть), которому в словаре соответствует число (ID).
- **`[CLS]` (101)** — ставится в начале, `[SEP]` (102) — в конце последовательности.
- **Padding** — добавление `[PAD]` (0), чтобы тексты в батче были одной длины.
- **`attention_mask`** — 1 для настоящих токенов, 0 для `[PAD]`: модель игнорирует нули.
