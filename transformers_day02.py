"""День 2 — Получение эмбеддингов (hidden states)"""
from transformers import AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import numpy as np


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загружаем саму нейросеть (веса) 
model = AutoModel.from_pretrained(model_name)
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

def explain_tokenization(text, tokenizer):
    """Показывает, как текст разбивается на токены и их ID."""
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")

def get_embeddings(texts, tokenizer, model, batch_size=32):
    """Возвращает CLS-эмбеддинги текстов"""
    all_embeddings = []

    for i in range(0,len(texts),batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenize_texts(batch_texts)

        with torch.no_grad():
            outputs = model(**tokens)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

def similarity(text1, text2, tokenizer, model):
    """Косинусное сходство двух текстов (от -1 до 1. чем ближе к 1, тем ближе по смыслу)."""
    emb = get_embeddings([text1,text2],tokenizer,model)
    sim = cosine_similarity(emb[0:1],emb[1:2])[0][0]
    return sim

def main():
    print("Задача 1: архитектура модели")
    print(model)
    print("\nЗадача 2: hidden states одного текста")
    text = "This movie was absolutely amazing!"
    tokens = tokenizer(text,return_tensors = 'pt')
    print(tokens)
    with torch.no_grad():
        outputs = model(**tokens)
    
    print(type(outputs))
    print(outputs.last_hidden_state.shape) #  [batch_size, sequence_length, hidden_size]
    cls_embedding = outputs.last_hidden_state[:,0,:] # первый cls токен
    print(f"CLS embedding shape: {cls_embedding.shape}")
    print(f"CLS embedding: {cls_embedding[0][:5]}...")

    print("\nЗадачи 3-4: get_embeddings")
    texts = [
        "This movie was absolutely amazing!",
        "Terrible movie, waste of time.",
        "Pretty good, I liked it.",
        "Boring and too long.",
    ]
    embeddings = get_embeddings(texts,tokenizer,model)
    print(f"Embeddings shape: {embeddings.shape}")
    print("Ожидается: (4, 768) для DistilBERT")

    print("\nЗадача 5: сходство текстов ")
    sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
    sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)

    print(f'Сходство похожих: {sim1:.3f}')
    print(f'Сходство разных: {sim2:.3f}')

if __name__ == "__main__":
    main()