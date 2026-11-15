"""День 6 — Инференс и сравнение моделей"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_PATH = "data/dataset.csv"


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
    print("Задача 1: загрузка моделей")
    model_ft = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
    model_ft.eval()

    try:
        baseline_model = joblib.load("baseline_model.pkl")
        baseline_vectorizer = joblib.load("vectorizer.pkl")
    except FileNotFoundError:
        baseline_model = None
        baseline_vectorizer = None
        print("baseline_model.pkl / vectorizer.pkl не найдены — baseline пропущен "
                "(создайте их: python train_tfidf_baseline.py)")

    print("\nЗадача 4: сравнение на примерах ")
    test_texts = [
        "This movie was absolutely fantastic!",
        "Terrible, waste of my time.",
        "It was okay, nothing special.",
        "Best film I've seen this year!",
        "Boring and too long.",
    ]

    preds_ft = predict_fine_tuned(test_texts,model_ft,tokenizer)

    if baseline_model:
        preds_baseline = predict_baseline(test_texts, baseline_model, baseline_vectorizer)

    for i,text in enumerate(test_texts):
        print(f"\nТекст: {text}")
        print(f'Fine-tuned: {preds_ft[i]["prediction"]} (probs: {preds_ft[i]["probabilities"]})')
        if baseline_model:
            print(f'Baseline: {preds_baseline[i]["prediction"]}')
            print(f'Совпадают: {preds_ft[i]["prediction"] == preds_baseline[i]["prediction"]}')

    print("\nЗадача 5: confusion matrix ")

    df = pd.read_csv(DATA_PATH)
    _, test_df, _, _ = train_test_split(
        df, df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    test_texts = test_df["text"].astype(str).tolist()
    test_labels = test_df["label"].tolist()

    preds_ft_all = predict_fine_tuned(test_texts, model_ft, tokenizer)
    y_pred_ft = [p["prediction"] for p in preds_ft_all]

    cm_ft = confusion_matrix(test_labels, y_pred_ft)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ft, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Fine-tuned Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix_finetuned.png")
    plt.close()
    print("Сохранено: confusion_matrix_finetuned.png")

    print("\nЗадача 6: сравнение метрик")
    print("Fine-tuned Model ")
    print(classification_report(test_labels, y_pred_ft))
    f1_ft = f1_score(test_labels, y_pred_ft, average="macro")
    acc_ft = accuracy_score(test_labels, y_pred_ft)

    if baseline_model:
        preds_baseline_all = predict_baseline(test_texts, baseline_model, baseline_vectorizer)
        y_pred_base = [p["prediction"] for p in preds_baseline_all]

        print("\nBaseline Model ")
        print(classification_report(test_labels, y_pred_base))
        f1_base = f1_score(test_labels, y_pred_base, average="macro")
        acc_base = accuracy_score(test_labels, y_pred_base)

        print("\nСравнение ")
        print(f"Fine-tuned F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}")
        print(f"Baseline F1: {f1_base:.4f}, Accuracy: {acc_base:.4f}")
        print(f"Улучшение F1: {(f1_ft - f1_base) / f1_base * 100:.2f}%")

    print("\nЗадача 7: сохранение результатов ")
    with open("comparison_results.txt", "w", encoding="utf-8") as f:
        f.write("Сравнение моделей \n\n")
        f.write("Fine-tuned Model:\n")
        f.write(f" F1 (macro): {f1_ft:.4f}\n")
        f.write(f" Accuracy: {acc_ft:.4f}\n")

        if baseline_model:
            f.write("\nBaseline Model:\n")
            f.write(f" F1 (macro): {f1_base:.4f}\n")
            f.write(f" Accuracy: {acc_base:.4f}\n")
            f.write(f"\nУлучшение: {(f1_ft - f1_base) / f1_base * 100:.2f}%\n")
    print("Сохранено: comparison_results.txt")



if __name__ == "__main__":
    main()