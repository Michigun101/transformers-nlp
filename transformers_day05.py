"""День 5 — Fine-tuning модели"""
import sys

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_PATH = "data/dataset.csv"
SAVE_DIR = "./fine_tuned_model"
RESULTS_PATH = "fine_tuned_results.txt"

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)


class SentimentDataset(Dataset):
    """Превращает списки текстов и меток в элементы, понятные DataLoader и модели."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # DataLoader спрашивает, сколько всего примеров.
    def __len__(self):
        return len(self.texts)

    # ...и просит пример по номеру. Токенизация происходит «на лету», при обращении.
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            # Всегда добиваем до max_length: так все примеры одной длины,
            # и DataLoader может склеить их в один тензор.
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            # return_tensors="pt" даёт форму [1, длина]; flatten() превращает её в [длина].
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            # Метка класса — целое число (long), как требует функция потерь.
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_epoch(model, dataloader, optimizer, device, desc="train"):
    """Одна эпоха обучения: проход по всем батчам. Возвращает среднюю потерю."""
    # train(): включает dropout и режим обучения.
    model.train()
    total_loss = 0

    # tqdm оборачивает DataLoader и рисует прогресс-бар по батчам.
    progress = tqdm(dataloader, desc=desc)
    for batch in progress:
        # Обнуляем градиенты предыдущего шага, иначе они накапливаются.
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Если передать labels, модель сама посчитает потерю (cross-entropy).
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        # Обратное распространение: считаем градиент потери по всем весам.
        loss.backward()
        # Шаг оптимизатора: обновляем веса по градиентам.
        optimizer.step()

        total_loss += loss.item()
        # Текущая потеря справа от прогресс-бара.
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, desc="eval"):
    """Оценка на валидации. Возвращает (accuracy, macro F1)."""
    # eval(): выключает dropout.
    model.eval()
    predictions = []
    true_labels = []

    # Градиенты не нужны — только предсказания.
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # logits — «сырые» оценки классов; argmax выбирает класс с наибольшей.
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")

    return accuracy, f1


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH

    # Задача 2: данные и разбиение 80/20 (stratify сохраняет баланс классов).
    df = pd.read_csv(data_path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    # Задача 3: модель = DistilBERT + новая классификационная «голова» на num_labels классов.
    # Предупреждение про не инициализированные веса классификатора — это нормально:
    # голова новая, её мы и будем обучать.
    num_labels = len(set(labels))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Задача 4: оптимизатор и устройство. lr=2e-5 — маленький шаг, типичный для fine-tuning.
    # AdamW берём из torch.optim: из transformers он удалён.
    optimizer = AdamW(model.parameters(), lr=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Устройство: {device}")

    # Задача 7: обучение
    num_epochs = 3
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, desc=f"Epoch {epoch + 1}/{num_epochs} train"
        )
        val_acc, val_f1 = evaluate(model, val_loader, device, desc=f"Epoch {epoch + 1}/{num_epochs} val")

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        print("-" * 50)

    # Задача 8: сохраняем модель с токенизатором (загрузка: from_pretrained('./fine_tuned_model')).
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Final Validation F1: {val_f1:.4f}\n")
        f.write(f"Final Validation Accuracy: {val_acc:.4f}\n")
    print(f"Модель сохранена в {SAVE_DIR}, метрики — в {RESULTS_PATH}")


if __name__ == "__main__":
    main()

