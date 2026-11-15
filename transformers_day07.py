"""День 7 — Анализ ошибок модели"""
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_PATH = "data/dataset.csv"

model_ft = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
model_ft.eval()

def predict_fine_tuned(texts, model, tokenizer):
    if isinstance(texts,str):
        texts = [texts]

    predictions = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors = 'pt',
            truncation = True,
            max_length=128
            )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits,dim = 1)
        pred = torch.argmax(probs, dim = 1).item()
        predictions.append({
            "text": text,
            "prediction": pred,
            "probabilities": probs[0].cpu().numpy(),
        })

    return predictions

def predict_baseline(texts, model, vectorizer, clean_func=None):
    if isinstance(texts, str):
        texts = [texts]

    if clean_func:
        texts = [clean_func(t) for t in texts]

    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model,'predict_proba') else None

    results = []
    for i, text in enumerate(texts):
        results.append({
            "text": text,
            "prediction": int(predictions[i]),
            "probabilities": probs[i] if probs is not None else None,
        })

    return results

def main():
    print("Задача 1: Загрузите данные и предсказания из Дня 6")
    df = pd.read_csv(DATA_PATH)
    _, test_df, _, _ = train_test_split(
        df, df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    test_texts = test_df["text"].astype(str).tolist()
    test_labels = test_df["label"].tolist()

    preds_ft_all = predict_fine_tuned(test_texts,model_ft,tokenizer)
    y_pred_ft = [p['prediction'] for p in preds_ft_all]
    df_test = pd.DataFrame({
        "text": test_texts,
        "true_label": test_labels,
        "pred_label": y_pred_ft,
    })

    errors = df_test[df_test['true_label'] != df_test['pred_label']].copy()

    fp = errors[(errors["pred_label"] == 1) & (errors["true_label"] == 0)]
    fn = errors[(errors["pred_label"] == 0) & (errors["true_label"] == 1)]

    print(f"Всего ошибок: {len(errors)}")
    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")

    print("\nЗадача 2: паттерны ошибок ")
    print("\n=== FALSE POSITIVES (сказали good, а было bad) ===")
    for idx, row in fp.head(5).iterrows():
        print(f'\nТекст: {row["text"][:100]}...')
        print(f'Истинный класс: {row["true_label"]}, Предсказан: {row["pred_label"]}')

    print("\n=== FALSE NEGATIVES (сказали bad, а было good) ===")
    for idx, row in fn.head(5).iterrows():
        print(f'\nТекст: {row["text"][:100]}...')
        print(f'Истинный класс: {row["true_label"]}, Предсказан: {row["pred_label"]}')

    errors["text_length"] = errors["text"].str.len()
    print(f'\nСредняя длина ошибочных текстов: {errors["text_length"].mean():.0f}')
    print(f'Средняя длина всех текстов: {df_test["text"].str.len().mean():.0f}')

    with open("error_analysis.txt", "w", encoding="utf-8") as f:
        f.write("=== АНАЛИЗ ОШИБОК ===\n\n")
        f.write(f"Всего ошибок: {len(errors)}\n")
        f.write(f"False Positives: {len(fp)}\n")
        f.write(f"False Negatives: {len(fn)}\n\n")

        f.write("=== ПРИМЕРЫ FALSE POSITIVES ===\n")
        for idx, row in fp.head(5).iterrows():
            f.write(f'\nТекст: {row["text"]}\n')
            f.write(f'Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n')

        f.write("\n\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n")
        for idx, row in fn.head(5).iterrows():
            f.write(f'\nТекст: {row["text"]}\n')
            f.write(f'Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n')

        f.write("\n\n=== НАБЛЮДЕНИЯ ===\n")
        f.write("Добавьте сюда ваши выводы о паттернах ошибок...\n")
    print("\nСохранено: error_analysis.txt")


if __name__ == "__main__":
    main()