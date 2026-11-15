"""День 4 — Baseline без обучения трансформера"""
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

DATA_PATH = "data/dataset.csv"
TEXT_COL = "text"
LABEL_COL = "label"
RESULTS_PATH = "baseline_results.txt"

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загружаем саму нейросеть (веса) 
model = AutoModel.from_pretrained(
    model_name,
   )
model.eval()

def tokenize_texts(texts, max_length=128):
    """Токенизирует список текстов в батч тензоров PyTorch."""
    return tokenizer(
        texts,
        padding = True,     # добиваем нулями [PAD] короткие тексты  до самого длинного в батче
        truncation = True, # обрезаем всё, что длиннее max_length
        max_length=max_length,
        return_tensors = 'pt'
    )

def get_cls_embeddings(texts, batch_size=32):
    """Возвращает CLS-эмбеддинги текстов"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenize_texts(batch_texts)

        with torch.no_grad():
            outputs = model(**tokens)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

def run_baseline(data_path=DATA_PATH):
    """LogisticRegression на CLS-эмбеддингах. Возвращает macro F1."""
    df = pd.read_csv(data_path)
    texts = df[TEXT_COL].astype(str).tolist()
    y = df[LABEL_COL].tolist()

    X = get_cls_embeddings(texts)
    print(f"Эмбеддинги: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)

    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {f1:.4f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name} (frozen) + LogisticRegression\n")
        f.write(f"Train/test: {len(y_train)}/{len(y_test)}\n\n")
        f.write(report)
        f.write(f"\nMacro F1: {f1:.4f}\n")
    print(f"Результаты сохранены в {RESULTS_PATH}")
    return f1


def main():
    print("Задача 1: tokenize_texts ")
    examples = [
        "This movie was great!",
        "Terrible movie, waste of time.",
        "Pretty good, I liked it.",
    ]
    batch = tokenize_texts(examples)
    print(batch)
    print(f"Shape: {batch['input_ids'].shape}")

    print("\nЗадача 2: get_cls_embeddings ")
    emb = get_cls_embeddings(examples)
    print(f"Embeddings shape: {emb.shape}")

    print("\nЗадача 3: baseline ")
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    run_baseline(path)

if __name__ == "__main__":
    main()