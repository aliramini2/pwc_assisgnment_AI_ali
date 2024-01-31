# This file is dedicated to data loading and tokenization.

from datasets import load_dataset

def load_and_tokenize_dataset(tokenizer, dataset_name):
    dataset = load_dataset(dataset_name)
    tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=512), batched=True)
    return tokenized_dataset
