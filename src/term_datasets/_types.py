from collections.abc import Mapping
from typing import TypedDict, Mapping, NotRequired


class CLRuTerm3OriginalJSON(TypedDict):
    id: str
    label: list[list[int]]
    text: str
    keywords: str
    candidate_label: NotRequired[list[list[int]]]

class RawDatasetElement(TypedDict):
    id: str
    label: list[list[int]] # list[tuple[int, int]]
    candidates: NotRequired[list[list[int]]] # list[tuple[int, int]]
    text: str

class TokenizedDatasetElement(TypedDict):
    input_ids: list[int]
    labels: NotRequired[list[int]]
    attention_mask: NotRequired[list[int]]
    token_type_ids: NotRequired[list[int]]
    # @TODO: maybe try to wrap it in BatchEncoding! That would be AWESOME!

class SpanDatasetElement(TypedDict):
    id: str
    text: str
    candidate_text: str
    span_start: int
    span_end: int
    label: NotRequired[int]

class TokenizedSpanDatasetElement(TypedDict):
    id: str
    input_ids: list[int]
    span_start: int
    span_end: int
    attention_mask: NotRequired[list[int]]
    token_type_ids: NotRequired[list[int]]
    label: NotRequired[int]