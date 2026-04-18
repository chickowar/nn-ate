from typing import TypedDict

class CLRuTerm3OriginalJSON(TypedDict):
    id: str
    label: list[list[int]]
    text: str
    keywords: str

class CLRuTerm3JSON(TypedDict):
    id: str
    label: list[list[int]] # list[tuple[int, int]]
    text: str

class CLRuTerm3TokenizedElement(TypedDict):
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    # @TODO: maybe try to wrap it in BatchEncoding! That would be AWESOME!