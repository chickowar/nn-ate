import json
from transformers import AutoTokenizer

from nn_ate.bert_getters import DEFAULT_MODEL
from term_datasets.CL_RuTerm3 import (
    CLRUTERM3_TRAIN1_CANDIDATES_PATH, CLRUTERM3_TEST1_PATH, CLRUTERM3_TEST23_PATH,
)
from term_datasets.CL_RuTerm3_getters import get_span_dataset, get_raw_dataset, get_tokenized_span_dataset
from term_datasets._types import CLRuTerm3OriginalJSON
from collections import defaultdict, Counter

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

ds = get_tokenized_span_dataset(CLRUTERM3_TRAIN1_CANDIDATES_PATH, tokenizer=tokenizer, has_preprocessed_candidates=True)
row = next(iter(ds))
print(*(f"{k}: {v}" for k, v in row.items()), sep="\n")

# with open(CLRUTERM3_TEST1_PATH, 'r', encoding='utf-8-sig') as f:
#     data: list[CLRuTerm3OriginalJSON] = [json.loads(line) for line in f]
#
# cnt = 0
# s1 = set()
# for i, js in enumerate(data):
#     label = js['label']
#     s = set()
#     for st, en in label:
#         if (st, en) not in s:
#             s.add((st, en))
#         else:
#             cnt += 1
#             s1.add((js['id'], js['text'][st:en], (st, en)))
#
# terms = [
#     data[i]['text'][st: en] for i, spans in enumerate(map(lambda x: x['label'], data)) for st, en in spans
# ]
#
# term_lengths = list(map(
#     len,
#     map(str.split, terms)
# ))
# print(Counter(term_lengths))