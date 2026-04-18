from pathlib import Path
from typing import Any

import numpy as np
from transformers import BatchEncoding, TokenizersBackend

# =========================== CONSTANTS ===========================
BIO2ID: dict[str, int] = {"O": 0, "B-TERM": 1, "I-TERM": 2}
ID2BIO: dict[int, str] = {v: k for k, v in BIO2ID.items()}
BILOU2ID: dict[str, int] = {"O": 0, "B-TERM": 1, "I-TERM": 2, "L-TERM": 3, "U-TERM": 4}
ID2BILOU: dict[int, str] = {v: k for k, v in BILOU2ID.items()}
SPECIAL_TOKEN_LABEL_ID: int = -100

CLRUTERM3_TRAIN1_PATH : str = str(Path(__file__).parents[2] / 'data' / 'CL-RuTerm3' / 'original' / 'train_t1_v1.jsonl')
CLRUTERM3_TRAIN23_PATH : str = str(Path(__file__).parents[2] / 'data' / 'CL-RuTerm3' / 'original' / 'train_t23_v1.jsonl')

MODEL_NAME: str = 'DeepPavlov/rubert-base-cased'


#
# =========================== DATASET PREPARATION ===========================
def get_flat_terms(terms: list[list[int]]) -> list[list[int]]:
    """From list of boundaries of terms get list of boundaries of ONLY flat terms"""
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


#
# =========================== TOKENIZATION ===========================
def check_assumption(s,e,term_s,term_e, batch, b) -> None:
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
    enc.pop('offset_mapping')
    enc.pop('token_type_ids')  # are they needed???
    enc['labels'] = batched_labels
    return enc

def tokenize_batch_BILOU(
        batch: dict[str, list[Any]],
        tokenizer: TokenizersBackend | None = None,
        max_length = 1024
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
    enc.pop('offset_mapping')
    enc.pop('token_type_ids')  # are they needed???
    enc['labels'] = batched_labels
    return enc

def tokenize_batch(
        batch: dict[str, list[Any]],
        tokenizer: TokenizersBackend | None = None,
        max_length = 1024
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
    enc.pop('offset_mapping')
    enc.pop('token_type_ids')  # are they needed???
    enc['labels'] = batched_labels
    return enc

