# Sentiment Analysis с BERT

## Описание

Проект по анализу тональности текстов (отзывы о фильмах, 2 класса: negative / positive)
с использованием трансформеров. За 7 дней: токенизация, эмбеддинги, attention, baseline,
fine-tuning DistilBERT, сравнение моделей, анализ ошибок и демо-приложение.

Датасет: 2000 отзывов IMDB (`python prepare_dataset.py` → `data/dataset.csv`, колонки `text`, `label`).

## Прогресс

- [x] День 1 — токенизация (`transformers_day01.py`)
- [x] День 2 — эмбеддинги и сходство текстов (`transformers_day02.py`)
- [x] День 3 — attention и визуализация (`transformers_day03.py`)
- [x] День 4 — baseline: CLS-эмбеддинги + LogisticRegression (`transformers_day04.py`)
- [x] День 5 — fine-tuning DistilBERT (`transformers_day05.py`)
- [x] День 6 — сравнение baseline и fine-tuned (`transformers_day06.py`)
- [x] День 7 — анализ ошибок и демо (`transformers_day07.py`, `app.py`)

## Структура

- `fine_tuned_model/` — обученная модель (создаёт День 5, в git не хранится)
- `app.py` — Gradio приложение
- `error_analysis.txt` — анализ ошибок (создаёт День 7)
- `comparison_results.txt` — сравнение моделей (создаёт День 6)
- `transformers_dayNN.py` — код по дням
- `prepare_dataset.py`, `train_tfidf_baseline.py` — подготовка данных и baseline

## Установка

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate
pip install -r requirements.txt
python prepare_dataset.py
```

Модели и токенизаторы скачиваются с huggingface.co при первом запуске и кэшируются.

## Запуск демо

```bash
pip install gradio transformers torch
python app.py
```
Откройте http://127.0.0.1:7860

Нужна папка `fine_tuned_model/` (запустите `python transformers_day05.py`).

## Результаты

Тестовая выборка: 400 отзывов (20%, `random_state=42`, `stratify`).

### Fine-tuned модель:
- F1 (macro): 0.81
- Accuracy: 0.81

### Baseline модель:
- F1 (macro): X.XX
- Accuracy: X.XX

Улучшение: XX%

(Baseline — после `python train_tfidf_baseline.py` и `python transformers_day06.py`
цифры берутся из `comparison_results.txt`. Embeddings-baseline Дня 4: macro F1 = 0.7925.)

## Использование в коде

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')

# Предсказание
inputs = tokenizer("Your text here", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)  # 0 — negative, 1 — positive
```

## Требования

- Python 3.10+
- transformers, torch
- gradio (для демо)

Полный список версий — в `requirements.txt`.

# Журнал по дням

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

## День 2 — эмбеддинги

Что делает `transformers_day02.py` (включает код Дня 1):

| Задача | Что показывает |
|---|---|
| 1 | `AutoModel.from_pretrained`, `model.eval()`, архитектура DistilBERT |
| 2 | `last_hidden_state` формы `[batch, длина, 768]`, CLS-вектор |
| 3-4 | `get_embeddings()` — батчевое получение CLS-эмбеддингов, результат `(N, 768)` |
| 5 | `similarity()` — косинусное сходство двух текстов |

### Ключевые понятия

- **Hidden states** — векторы (по 768 чисел), в которые модель превращает каждый токен с учётом контекста.
- **CLS-эмбеддинг** — вектор токена `[CLS]`, используется как представление всего предложения.
- **`torch.no_grad()`** — отключает расчёт градиентов: при получении эмбеддингов модель не обучается.
- **Косинусное сходство** — мера близости векторов: чем ближе к 1, тем ближе тексты по смыслу.

## День 3 — attention

`transformers_day03.py` (включает код Дней 1-2). Модель загружается с `output_attentions=True`
и `attn_implementation="eager"` (быстрые реализации attention матрицы не возвращают).

| Задача | Что показывает |
|---|---|
| 1-2 | `outputs.attentions` — кортеж по слоям, форма `[batch, heads, seq, seq]` |
| 3 | `visualize_attention()` — heatmap одной головы, сохраняется в `figures/` |
| 4 | Сравнение слоёв 0, 3, 5 |
| 5 | Все головы слоя 0 (в DistilBERT их 12, не 8) |
| 6 | Внимание слова `terrible` на последнем слое |

### Ключевые понятия

- **Attention-матрица** `[seq, seq]`: строка — токен-«запрашивающий» (query), столбец — токен, на который он смотрит (key). Сумма по строке равна 1.
- **Слои** — DistilBERT имеет 6 слоёв; **головы** — 12 параллельных «взглядов» в каждом слое, они находят разные связи.

## День 4 — baseline без обучения трансформера

`transformers_day04.py`: замороженный DistilBERT → CLS-эмбеддинги (768) → `LogisticRegression`.

- Датасет — CSV с колонками текста и метки; путь и имена колонок задаются константами `DATA_PATH`, `TEXT_COL`, `LABEL_COL` (или путь аргументом командной строки).
- Разбиение 80/20, `stratify=y`, `random_state=42`.
- Метрики: `classification_report` и macro F1; результат сохраняется в `baseline_results.txt`.
- Классификатор назван `clf`, чтобы не затереть трансформер `model`.

### Датасет для Дня 4

Если своего датасета нет: `python prepare_dataset.py` скачивает 2000 отзывов IMDB (по 1000 на класс,
0 — негативный, 1 — позитивный) в `data/dataset.csv`. Файл в git не коммитится.

## День 5 — fine-tuning

`transformers_day05.py`: `AutoModelForSequenceClassification` (DistilBERT + классификационная голова),
свой `SentimentDataset`, цикл обучения `train_epoch` / `evaluate`, 3 эпохи, `AdamW(lr=2e-5)`.

- Данные — тот же `data/dataset.csv`, разбиение 80/20 со `stratify`.
- Результат: `./fine_tuned_model` (модель + токенизатор) и `fine_tuned_results.txt` — оба в `.gitignore`.
- `AdamW` импортируется из `torch.optim` (в `transformers` его больше нет).
- На CPU обучение занимает десятки минут; с GPU — минуты.

## День 6 — сравнение baseline и fine-tuned

`transformers_day06.py`: `predict_fine_tuned`, `predict_baseline`, confusion matrix, сравнение метрик — как в задании.

- Тестовая выборка — то же разбиение 80/20 (`random_state=42`, `stratify`), что в Днях 4 и 5: модели не видели эти тексты при обучении.
- Baseline загружается из `baseline_model.pkl` и `vectorizer.pkl`. Если файлов нет, baseline пропускается (так допускает задание, `if baseline_model:`).
- Эти файлы создаёт `python train_tfidf_baseline.py` (TF-IDF + LogisticRegression на том же train-разбиении). Это классический baseline, он отличается от embeddings-baseline Дня 4.
- Результаты: `comparison_results.txt`, `confusion_matrix_finetuned.png` (в `.gitignore`).

## День 7 — анализ ошибок и демо

- `transformers_day07.py` — находит ошибки fine-tuned модели на тестовой выборке, делит их на False Positive / False Negative, печатает примеры, сравнивает среднюю длину ошибочных текстов со средней по всем и пишет `error_analysis.txt`. Раздел «НАБЛЮДЕНИЯ» в файле заполняется вручную.
- `app.py` — Gradio-демо. Метки только `Negative` / `Positive` (в датасете 2 класса, в задании была 3-классовая карта).
