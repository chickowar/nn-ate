import json
from pathlib import Path
from typing import Callable, Any

from datasets import Dataset, DatasetDict
from transformers import TokenizersBackend, BatchEncoding, AutoTokenizer

from term_datasets.CL_RuTerm3 import (
    SPAN_ID2LABEL,
    SPAN_LABEL2ID,
    generator_span_dataset_element,
    get_flat_terms,
    tokenize_batch_BIO, tokenize_batch_span_classification, tokenize_span_dataset,
)
from term_datasets._types import CLRuTerm3OriginalJSON
from term_datasets.text_processing import TextProcessor


def get_raw_dataset(jsonl_path: Path | str, flat: bool = False) -> Dataset:
    """
    Returns a 'raw' dataset. Columns are listed below.

    text: texts

    label: a list of term boundaries (by symbol)

    id: a list of ids from original JSON file (not needed actually)

    :param jsonl_path:
    :param flat:
    :return: Dataset({'text': str, 'label': list[list[int]], 'id': str})
    """
    texts: list[str] = []
    labels: list[list[list[int]]] = []
    ids: list[str] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            js: CLRuTerm3OriginalJSON = json.loads(line)
            labels.append(js["label"])
            if flat:
                labels[-1] = get_flat_terms(labels[-1])
            texts.append(js["text"])
            ids.append(js["id"])

    return Dataset.from_dict({"text": texts, "label": labels, "id": ids})


def get_tokenized_dataset(
        jsonl_path: Path | str,
        tokenizer: TokenizersBackend,
        labeling_technique: Callable[[Any], BatchEncoding] = tokenize_batch_BIO,
        batched: bool = True,
        batch_size: int = 1,
        flat: bool = True
) -> Dataset:
    ds = get_raw_dataset(jsonl_path, flat=flat)
    return ds.map(
        labeling_technique,
        fn_kwargs={"tokenizer": tokenizer},
        batched=batched,
        batch_size=batch_size,
        remove_columns=ds.column_names,
    )


def get_train_test_split_tokenized_dataset(
        test_size: float = 0.15,
        seed: int = 42,
        **tokenized_dataset_kwargs
) -> tuple[Dataset, Dataset]:
    dataset: DatasetDict = get_tokenized_dataset(**tokenized_dataset_kwargs).train_test_split(
        test_size=test_size,
        seed=seed,
    )
    return dataset["train"], dataset["test"]


def get_span_dataset(
        jsonl_path: Path | str | None = None,
        flat: bool = False,
        **generator_kwargs
) -> Dataset:
    """
    Returns a SpanDataset
    :param jsonl_path: путь к jsonl (если в generator_kwargs не указан raw_dataset)
    :param flat:
    :param generator_kwargs: аргументы для generator_span_dataset_element
    :return: Dataset[SpanDatasetElement]
    """
    if jsonl_path is None and "raw_dataset" not in generator_kwargs:
        raise TypeError("Either pass a dataset as a keyword or jsonl_path")
    elif jsonl_path is not None:
        generator_kwargs["raw_dataset"] = get_raw_dataset(jsonl_path, flat=flat)
    return Dataset.from_generator(generator_span_dataset_element, gen_kwargs=generator_kwargs)


def get_tokenized_span_dataset(
        jsonl_path: Path | str,
        tokenizer: TokenizersBackend,
        max_words_per_ngram: int = 7,
        max_pair_length: int = 512,
        batch_size: int = 1000,
        negative_ratio: float | None = 3.0,
        seed: int = 42,
        flat: bool = False,
        text_processor: TextProcessor | None = None,
) -> Dataset:
    span_dataset = get_span_dataset(
        jsonl_path=jsonl_path,
        max_words_per_ngram=max_words_per_ngram,
        negative_ratio=negative_ratio,
        seed=seed,
        flat=flat,
        text_processor=text_processor,
    )
    return tokenize_span_dataset(span_dataset, tokenizer, max_pair_length, batch_size)

def get_train_test_split_tokenized_span_dataset(
        jsonl_path: Path | str,
        tokenizer: TokenizersBackend,
        max_words_per_ngram: int = 7,
        max_pair_length: int = 512,
        batch_size: int = 1000,
        train_negative_ratio: float | None = 3.0,
        eval_negative_ratio: float | None = None,
        flat: bool = False,
        text_processor: TextProcessor | None = None,

        test_size: float = 0.15,
        seed: int = 42,
) -> tuple[Dataset, Dataset]:
    raw_dataset: DatasetDict = get_raw_dataset(jsonl_path=jsonl_path, flat=flat).train_test_split(
        test_size=test_size,
        seed=seed,
    )

    train_dataset= get_span_dataset(
        raw_dataset=raw_dataset["train"],
        max_words_per_ngram=max_words_per_ngram,
        negative_ratio=train_negative_ratio,
        seed=seed,
        text_processor=text_processor,
    )
    eval_dataset = get_span_dataset(
        raw_dataset=raw_dataset["test"],
        max_words_per_ngram=max_words_per_ngram,
        negative_ratio=eval_negative_ratio,
        seed=seed + 1,
        text_processor=text_processor,
    )

    tokenized_train = tokenize_span_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=max_pair_length,
        batch_size=batch_size,
    )

    tokenized_eval = tokenize_span_dataset(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_pair_length,
        batch_size=batch_size,
    )

    return tokenized_train, tokenized_eval


def main():
    from CL_RuTerm3 import CLRUTERM3_TRAIN1_PATH, MODEL_NAME, DEFAULT_TEXT_PROCESSOR

    def get_tokenizer(model_name=MODEL_NAME) -> TokenizersBackend:
        return AutoTokenizer.from_pretrained(model_name)

    tokenizer: TokenizersBackend = get_tokenizer()

    train, eval = get_train_test_split_tokenized_span_dataset(
        jsonl_path=CLRUTERM3_TRAIN1_PATH,
        tokenizer=tokenizer,
        max_words_per_ngram=7,
        max_pair_length=512,
        train_negative_ratio=None,
        eval_negative_ratio=None,
        batch_size=1000,
        text_processor=DEFAULT_TEXT_PROCESSOR,
    )

    print(train[0].keys())
    print(*(f"{k}: {v}\n" for k, v in train[0].items()),sep='')

if __name__ == "__main__":
    main()