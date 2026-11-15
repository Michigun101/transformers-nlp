"""День 3 — Attention-матрицы и визуализация.

Запуск: python transformers_day03.py
Графики сохраняются в папку figures/.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# output_attentions=True — модель вернёт веса внимания всех слоёв.
# attn_implementation="eager" — обычная реализация attention: ускоренные варианты (sdpa)
# не умеют возвращать матрицы внимания, а нам нужны именно они.
model = AutoModel.from_pretrained(
    model_name,
    output_attentions=True,
    attn_implementation="eager",
)
model.eval()


def tokenize_texts(texts, max_length=128):
    """Токенизирует список текстов в батч тензоров PyTorch."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def explain_tokenization(text, tokenizer):
    """Показывает, как текст разбивается на токены и их ID."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")


def get_embeddings(texts, tokenizer, model, batch_size=32):
    """Возвращает CLS-эмбеддинги текстов: массив формы (len(texts), hidden_size)."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenize_texts(batch_texts)

        with torch.no_grad():
            outputs = model(**tokens)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def similarity(text1, text2, tokenizer, model):
    """Косинусное сходство двух текстов (от -1 до 1; чем ближе к 1, тем ближе по смыслу)."""
    emb = get_embeddings([text1, text2], tokenizer, model)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return sim


def get_attention(text):
    """Прогоняет текст через модель и возвращает (tokens, outputs) для анализа внимания."""
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    return tokens, outputs


def visualize_attention(tokens, attention, layer=0, head=0, show=True, save_dir="figures"):
    """Рисует heatmap внимания одной головы одного слоя и сохраняет её в PNG.

    tokens: результат токенизатора (нужен для подписей осей)
    attention: outputs.attentions — кортеж из матриц по слоям
    layer, head: номер слоя и головы (с нуля)
    show: показывать ли окно с графиком (False — только сохранить)
    """
    # attention[layer] имеет форму [batch, heads, seq, seq]; берём первый текст и нужную голову.
    attn = attention[layer][0, head]

    # Числа ID -> строки токенов для подписей осей.
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
    )
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"attention_layer{layer}_head{head}.png"))
    if show:
        plt.show()
    # Закрываем фигуру, иначе при цикле по головам память растёт.
    plt.close()


def top_attended_tokens(tokens, attention, word, layer, top_k=3):
    """Для слова-запроса (query) печатает токены, на которые оно смотрит сильнее всего.

    Внимание усредняется по всем головам выбранного слоя.
    """
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    if word not in token_list:
        raise ValueError(f"Токен {word!r} не найден: {token_list}")

    # Среднее по головам: [heads, seq, seq] -> [seq, seq]
    attn = attention[layer][0].mean(dim=0)
    # Строка attn[q] — на что смотрит токен q (веса по всем ключам, сумма равна 1).
    row = attn[token_list.index(word)]
    top = torch.topk(row, k=min(top_k, len(token_list)))

    print(f"Слой {layer}: токен {word!r} больше всего смотрит на:")
    for weight, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {token_list[idx]:<12} {weight:.3f}")


def main():
    print("=== Задача 1-2: attention ===")
    text = "The amazing movie won many awards"
    tokens, outputs = get_attention(text)
    print(type(outputs.attentions))
    print(f"Количество слоёв: {len(outputs.attentions)}")
    print(f"Форма attention для слоя 0: {outputs.attentions[0].shape}")
    attention = outputs.attentions[0]
    print(f"Attention shape: {attention.shape}")
    attn_single = attention[0, 0]
    print(f"Single head shape: {attn_single.shape}")

    print("\n=== Задача 3-4: разные слои ===")
    n_layers = len(outputs.attentions)
    for layer in (0, n_layers // 2, n_layers - 1):
        visualize_attention(tokens, outputs.attentions, layer=layer, head=0)

    print("\n=== Задача 5: разные головы слоя 0 ===")
    # В DistilBERT 12 голов (n_heads), а не 8 — берём число из конфигурации модели.
    n_heads = model.config.n_heads
    print(f"Голов в слое: {n_heads}")
    for head in range(n_heads):
        visualize_attention(tokens, outputs.attentions, layer=0, head=head, show=False)
    print("Графики голов сохранены в figures/")

    print("\n=== Задача 6: сентимент-слово ===")
    text = "This movie was absolutely terrible and I hated it"
    tokens, outputs = get_attention(text)
    visualize_attention(tokens, outputs.attentions, layer=n_layers - 1, head=0)
    top_attended_tokens(tokens, outputs.attentions, "terrible", layer=n_layers - 1)


if __name__ == "__main__":
    main()
