import json
from pathlib import Path
from typing import Callable, Any

from datasets import Dataset, DatasetDict
from transformers import TokenizersBackend, BatchEncoding

from term_datasets.CL_RuTerm3 import get_flat_terms, tokenize_batch_BIO
from term_datasets._types import CLRuTerm3OriginalJSON


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
    ids: list[str] = []  # currently no use for ids, so skip???

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            js: CLRuTerm3OriginalJSON = json.loads(line)
            labels.append(js["label"])
            if flat:
                labels[-1] = get_flat_terms(labels[-1])
            texts.append(js["text"])
            ids.append(js["id"])
    return Dataset.from_dict({
        'text': texts,
        'label': labels,
        'id': ids,  # again, maybe we should not think about them
    })


def get_tokenized_dataset(
        jsonl_path: Path | str,
        tokenizer: TokenizersBackend,
        labeling_technique: Callable[[Any], BatchEncoding] = tokenize_batch_BIO,
        batched=True,
        batch_size=1,
        flat: bool = True
) -> Dataset:
    """
    Currently only for flat terms and for batches!

    input_ids: text's token_ids

    labels: a list of term boundaries (by symbol)

    id: a list of ids from original JSON file (not needed actually)

    :param batch_size:
    :param jsonl_path: path to CLRuTerm-3 (track-1) training jsonl file
    :param tokenizer: Tokenizer to use
    :param labeling_technique: function to pass to every element in dataset (BIO labeling by default),
    :return:
    """
    ds: Dataset = get_raw_dataset(jsonl_path, flat=flat)
    return ds.map(labeling_technique,
                  fn_kwargs={'tokenizer': tokenizer},
                  batched=batched, batch_size=batch_size,
                  remove_columns=ds.column_names)


def get_train_test_split_tokenized_dataset(
        test_size: float = 0.15,
        seed: int =42,
        **tokenized_dataset_kwargs
) -> tuple[Dataset, Dataset]:
    dataset: DatasetDict = get_tokenized_dataset(**tokenized_dataset_kwargs).train_test_split(test_size=test_size, seed=seed)
    return dataset["train"], dataset["test"]