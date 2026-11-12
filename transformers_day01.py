"""День 1 — Архитектура трансформеров и токенизация"""

from transformers import AutoTokenizer
import torch

# Задача 2: загрузка токенизатора 
model_name = "distilbert-base-uncased"
# model_name = "distilbert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

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

def main():
    print("Задача 2: параметры токенизатора")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"model_max_length: {tokenizer.model_max_length}")

    print("\nЗадача 3: токенизация текста")
    text = "This movie was absolutely amazing!"
    tokens = tokenizer(text)
    print(tokens)

    input_ids = tokens['input_ids']
    print(f'Количество токенов: {len(input_ids)}')

    decoded = tokenizer.decode(input_ids)
    print(f'Декодировано: {decoded}')

    print("\nЗадача 4: батчи ")
    texts = [
            "This movie was great!",
            "Terrible movie, waste of time.",
        ]
    tokens = tokenize_texts(texts)
    print(f'Shape: {tokens["input_ids"].shape}')
    print(f'Attention mask:\n{tokens["attention_mask"]}')

    print("\nЗадача 5: специальные токены ")
    print(f'CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})')
    print(f'SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})')
    print(f'PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})')

    single = tokenizer(text,return_tensors = 'pt')
    print(f"Input IDs: {single['input_ids']}")
    print(f"Decoded: {tokenizer.decode(single['input_ids'][0])}")


    print("\nЗадача 6: explain_tokenization")
    explain_tokenization("Transformers are amazing!", tokenizer)

if __name__ == "__main__":
    main()