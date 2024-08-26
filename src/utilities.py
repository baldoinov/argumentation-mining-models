# -*- coding: utf-8 -*-
import re
import torch

from unidecode import unidecode
from datasets import DatasetDict, Dataset


def split_data(dataset: Dataset) -> DatasetDict:

    train_dataset, test_dataset = dataset.train_test_split(
        test_size=0.3, stratify_by_column="labels", shuffle=True, seed=42
    ).values()

    val_dataset, test_dataset = test_dataset.train_test_split(
        test_size=0.5, stratify_by_column="labels", shuffle=True, seed=42
    ).values()

    dataset = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})

    return dataset


def tokenize_function(examples: DatasetDict, tokenizer, max_length: int):
    return tokenizer(
        examples["text"], padding="max_length", max_length=max_length, truncation=True
    )


def cast_to_float(sample: dict) -> dict:
    sample["labels"] = torch.tensor(sample["labels"], dtype=torch.float)
    
    return sample


def from_unicode_to_ascii(sample: dict) -> dict:
    """
    To be vectorized by `datasets`, the function has to receive a dict-lik object
    and return a dict-like object. Passing a single string to this function will
    not work.
    """
    processed_batch = []
    tweets = sample["text"]

    for text in tweets:
        text = unidecode(text)
        processed_batch.append(text)

    sample["text"] = processed_batch

    return sample


def remove_user_from_tweet(sample: dict) -> dict:
    """
    To be vectorized by `datasets`, the function has to receive a dict-lik object
    and return a dict-like object. Passing a single string to this function will
    not work.
    """
    processed_batch = []
    tweets = sample["text"]

    for text in tweets:
        text = re.sub("@\w+", "", text)
        processed_batch.append(text)

    sample["text"] = processed_batch

    return sample


def remove_urls(sample: dict) -> dict:
    """
    To be vectorized by `datasets`, the function has to receive a dict-lik object
    and return a dict-like object. Passing a single string to this function will
    not work.
    """
    processed_batch = []
    tweets = sample["text"]

    for text in tweets:
        text = re.sub(
            "((?:(?<=[^a-zA-Z0-9]){0,}(?:(?:https?\:\/\/){0,1}(?:[a-zA-Z0-9\%]{1,}\:[a-zA-Z0-9\%]{1,}[@]){,1})(?:(?:\w{1,}\.{1}){1,5}(?:(?:[a-zA-Z]){1,})|(?:[a-zA-Z]{1,}\/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\:[0-9]{1,4}){1})){1}(?:(?:(?:\/{0,1}(?:[a-zA-Z0-9\-\_\=\-]){1,})*)(?:[?][a-zA-Z0-9\=\%\&\_\-]{1,}){0,1})(?:\.(?:[a-zA-Z0-9]){0,}){0,1})",
            "",
            text,
        )
        processed_batch.append(text)

    sample["text"] = processed_batch

    return sample


def remove_non_word_chars(sample: dict) -> dict:
    """
    To be vectorized by `datasets`, the function has to receive a dict-lik object
    and return a dict-like object. Passing a single string to this function will
    not work.
    """
    processed_batch = []
    tweets = sample["text"]

    for text in tweets:
        text = re.sub("[\W]+", " ", text)
        processed_batch.append(text)

    sample["text"] = processed_batch

    return sample


def remove_repeated_chars(sample: dict) -> dict:
    """
    To be vectorized by `datasets`, the function has to receive a dict-lik object
    and return a dict-like object. Passing a single string to this function will
    not work.
    """
    processed_batch = []
    tweets = sample["text"]

    for text in tweets:
        text = re.sub(r"(.)\1{2,3}", r"\1", text)
        processed_batch.append(text)

    sample["text"] = processed_batch

    return sample


def remove_trailing_whitespace(sample: dict) -> dict:
    """
    To be vectorized by `datasets`, the function has to receive a dict-lik object
    and return a dict-like object. Passing a single string to this function will
    not work.
    """
    processed_batch = []
    tweets = sample["text"]

    for text in tweets:
        text = text.strip()
        processed_batch.append(text)

    sample["text"] = processed_batch

    return sample


if __name__ == "__main__":
    pass
