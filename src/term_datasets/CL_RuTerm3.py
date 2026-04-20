import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Generator

import numpy as np
from datasets import Dataset
from transformers import BatchEncoding, TokenizersBackend

from term_datasets._types import SpanDatasetElement
from term_datasets.text_processing import TextProcessor, TextSpan

# =========================== CONSTANTS ===========================
BIO2ID: dict[str, int] = {"O": 0, "B-TERM": 1, "I-TERM": 2}
ID2BIO: dict[int, str] = {v: k for k, v in BIO2ID.items()}
BILOU2ID: dict[str, int] = {"O": 0, "B-TERM": 1, "I-TERM": 2, "L-TERM": 3, "U-TERM": 4}
ID2BILOU: dict[int, str] = {v: k for k, v in BILOU2ID.items()}
SPAN_LABEL2ID: dict[str, int] = {"O": 0, "TERM": 1}
SPAN_ID2LABEL: dict[int, str] = {v: k for k, v in SPAN_LABEL2ID.items()}
SPECIAL_TOKEN_LABEL_ID: int = -100

CLRUTERM3_TRAIN1_PATH: str = str(Path(__file__).parents[2] / 'data' / 'CL-RuTerm3' / 'original' / 'train_t1_v1.jsonl')
CLRUTERM3_TRAIN23_PATH: str = str(Path(__file__).parents[2] / 'data' / 'CL-RuTerm3' / 'original' / 'train_t23_v1.jsonl')
CLRUTERM3_TEST23_PATH: str = str(
    Path(__file__).parents[2] / 'data' / 'CL-RuTerm3' / 'original' / 'test1_t12_full_v2.jsonl')
CLRUTERM3_TEST1_PATH: str = str(
    Path(__file__).parents[2] / 'data' / 'CL-RuTerm3' / 'processed' / 'test1_t12_full_v2_NO_CLASS.jsonl')

MODEL_NAME: str = 'DeepPavlov/rubert-base-cased'
DEFAULT_TEXT_PROCESSOR = TextProcessor()

#
# =========================== DATASET PREPARATION ===========================
def get_flat_terms(terms: list[list[int]]) -> list[list[int]]:
    """From list of boundaries of terms get list of boundaries of ONLY flat terms"""
    if not terms:
        return []
    n = len(terms)
    flat_terms = []
    sorted_terms = sorted(terms)
    st, en = sorted_terms[0]
    for i in range(1, n):
        cur_st, cur_en = sorted_terms[i]
        if cur_st < en:
            en = max(cur_en, en)
        else:
            flat_terms.append((st, en))
            st, en = cur_st, cur_en
    if flat_terms[-1] != (st, en):
        flat_terms.append((st, en))

    return flat_terms


def build_span_classification_examples(
        text: str,
        labels: list[list[int]],
        sample_id: str,
        max_words_per_ngram: int = 7,
        negative_ratio: float | None = 3.0,
        seed: int = 42,
        text_processor: TextProcessor | None = None,
) -> list[SpanDatasetElement]:
    processor = text_processor or DEFAULT_TEXT_PROCESSOR
    term_spans: set[tuple[int, int]] = {tuple(boundary) for boundary in labels}
    candidate_spans: list[TextSpan] = [
        span
        for sentence in processor.split_sentences(text)
        for span in processor.extract_ngrams(sentence, max_n=max_words_per_ngram)
    ]

    positive_examples: list[SpanDatasetElement] = []
    negative_examples: list[SpanDatasetElement] = []

    for candidate in candidate_spans:
        boundaries = (candidate.start, candidate.end)
        example: SpanDatasetElement = {
            "id": sample_id,
            "text": text,
            "candidate_text": candidate.text,
            "span_start": candidate.start,
            "span_end": candidate.end,
            "label": int(boundaries in term_spans),
        }
        if example["label"] == 1:
            positive_examples.append(example)
        else:
            negative_examples.append(example)

    if negative_ratio is None:
        sampled_negatives = negative_examples
    elif negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    elif negative_ratio == 0 or not negative_examples:
        sampled_negatives: list[SpanDatasetElement] = []
    elif positive_examples:
        # filtering according to negative_ratio
        stable_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(sample_id))
        rng = np.random.default_rng(stable_seed % (2 ** 32))
        max_negatives = min(len(negative_examples), int(len(positive_examples) * negative_ratio))
        sampled_indexes = rng.choice(len(negative_examples), size=max_negatives, replace=False)
        sampled_negatives = [negative_examples[int(index)] for index in sampled_indexes]
    else:
        sampled_negatives = negative_examples

    return positive_examples + sampled_negatives

def generator_span_dataset_element(
        raw_dataset: Dataset,
        max_words_per_ngram: int = 7,
        negative_ratio: float | None = 3.0,
        seed: int = 42,
        text_processor: TextProcessor | None = None,
) -> Generator[SpanDatasetElement, None, None]:
    for row in raw_dataset:
        yield from build_span_classification_examples(
            text=row["text"],
            labels=row["label"],
            sample_id=row["id"],
            max_words_per_ngram=max_words_per_ngram,
            negative_ratio=negative_ratio,
            seed=seed,
            text_processor=text_processor)

#
# =========================== TOKENIZATION ===========================
def check_assumption(s, e, term_s, term_e, batch, b) -> None:
    """ASSUMPTION: a term cannot be longer than any singular token and no token is only partially in term's boundary"""
    if s < term_e < e or s < term_s < e:
        cur_text, cur_token, cur_term = batch['text'][b], batch['text'][b][s:e], batch['text'][b][term_s:term_e]
        err_text = f"ASSUMPTION in tokenize_batch function is WRONG!\n{cur_text=}\n{cur_token=}\n{cur_term=}"
        raise AssertionError(err_text)

def tokenize_batch_BIO(
        batch: dict[str, list[Any]],
        tokenizer: TokenizersBackend | None = None,
        max_length: int = 1024
) -> BatchEncoding:
    """
    Currently only for flat terms and for batches
    Tokenizes the texts and labels them in BIO format

    :param batch: batch of {text: list of texts, label: list of lists of boundaries}
    :param tokenizer: tokenizer to use
    :param max_length: maximum length of tokens for a batch
    :return: batch with tokenized texts and corresponding BIO-labels
    """
    assert tokenizer is not None, "tokenizer must be provided in tokenize_batch function"
    batched_sym_labels = batch['label']
    enc: BatchEncoding = tokenizer.__call__(batch['text'], return_tensors='pt',
                                            return_offsets_mapping=True, padding=True, max_length=max_length)
    batched_offsets = enc.offset_mapping  # shape: (b, length, 2)
    batched_labels = np.full((batched_offsets.shape[0], batched_offsets.shape[1]),
                             fill_value=BIO2ID['O'])  # shape: (b, length)
    batched_labels[(batched_offsets[:, :, 0] == batched_offsets[:, :, 1])] = SPECIAL_TOKEN_LABEL_ID

    for b, offsets in enumerate(batched_offsets):
        if len(batched_sym_labels[b]) == 0:
            continue
        terms_iter = iter(batched_sym_labels[b])
        term_s, term_e = next(terms_iter)
        # ASSUMPTION: a term cannot be longer than any singular token and no token is only partially in term's boundary
        for i, (s, e) in enumerate(offsets):
            check_assumption(s, e, term_s, term_e, batch, b)
            if s >= term_e:
                try:
                    term_s, term_e = next(terms_iter)
                except StopIteration:
                    break
            if term_s <= s < e <= term_e:
                batched_labels[b, i] = BIO2ID['B-TERM' if term_s == s else 'I-TERM']
    # enc.pop('offset_mapping')
    # enc.pop('token_type_ids')  # are they needed???
    enc['labels'] = batched_labels
    return enc


def tokenize_batch_BILOU(
        batch: dict[str, list[Any]],
        tokenizer: TokenizersBackend | None = None,
        max_length=1024
) -> BatchEncoding:
    """
    Currently only for flat terms and for batches
    Tokenizes the texts and labels them in BILOU format

    :param batch: batch of {text: list of texts, label: list of lists of boundaries}
    :param tokenizer: tokenizer to use
    :param max_length: maximum length of tokens for a batch
    :return: batch with tokenized texts and corresponding BILOU-labels
    """
    assert tokenizer is not None, "tokenizer must be provided in tokenize_batch function"
    batched_sym_labels = batch['label']
    enc: BatchEncoding = tokenizer.__call__(batch['text'], return_tensors='pt',
                                            return_offsets_mapping=True, padding=True, max_length=max_length)
    batched_offsets = enc.offset_mapping  # shape: (b, length, 2)
    batched_labels = np.full((batched_offsets.shape[0], batched_offsets.shape[1]),
                             fill_value=BILOU2ID['O'])  # shape: (b, length)
    batched_labels[(batched_offsets[:, :, 0] == batched_offsets[:, :, 1])] = SPECIAL_TOKEN_LABEL_ID

    for b, offsets in enumerate(batched_offsets):
        if len(batched_sym_labels[b]) == 0:
            continue
        terms_iter = iter(batched_sym_labels[b])
        term_s, term_e = next(terms_iter)
        for i, (s, e) in enumerate(offsets):
            check_assumption(s, e, term_s, term_e, batch, b)
            if s >= term_e:
                try:
                    term_s, term_e = next(terms_iter)
                except StopIteration:
                    break
            if term_s == s < e == term_e:
                label = 'U-TERM'
            elif term_s == s < e < term_e:
                label = 'B-TERM'
            elif term_s < s < e == term_e:
                label = 'L-TERM'
            elif term_s < s < e < term_e:
                label = 'I-TERM'
            else:
                continue
            batched_labels[b, i] = BILOU2ID[label]
    # enc.pop('offset_mapping')
    # enc.pop('token_type_ids')  # are they needed???
    enc['labels'] = batched_labels
    return enc


def tokenize_batch_span_classification(
        batch: dict[str, list[str] | list[int]],
        tokenizer: TokenizersBackend,
        max_length: int = 512,
) -> BatchEncoding:
    encoded = tokenizer.__call__(
        text=batch["text"],
        text_pair=batch["candidate_text"],
        truncation="only_first",
        max_length=max_length,
        padding=False,
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    encoded["label"] = batch["label"]
    encoded["id"] = batch["id"]
    encoded["span_start"] = batch["span_start"]
    encoded["span_end"] = batch["span_end"]
    return encoded


def tokenize_span_dataset(
        dataset: Dataset,
        tokenizer: TokenizersBackend,
        max_length: int = 512,
        batch_size: int = 1000,
) -> Dataset:
    """
    Tokenizes a SpanDataset
    :param dataset:
    :param tokenizer:
    :param max_length:
    :param batch_size:
    :return: Dataset[TokenizedSpanDatasetElement]
    """
    return dataset.map(
        tokenize_batch_span_classification,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"tokenizer": tokenizer,'max_length': max_length,},
        remove_columns=["text", "candidate_text"],
        # load_from_cache_file=False, # МОЖЕТ БЫТЬ ПОЛЕЗНО ВКЛЮЧИТЬ!
    )
