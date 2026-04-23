import json
# from transformers import AutoTokenizer
#
# from nn_ate.bert_getters import DEFAULT_MODEL
from term_datasets.CL_RuTerm3 import (
    CLRUTERM3_TRAIN1_CANDIDATES_PATH, CLRUTERM3_TEST1_PATH, CLRUTERM3_TEST23_PATH,
)
# from term_datasets.CL_RuTerm3_getters import get_span_dataset, get_raw_dataset, get_tokenized_span_dataset
# from term_datasets._types import CLRuTerm3OriginalJSON
# from collections import defaultdict, Counter
#
# # tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
# #
# # ds = get_tokenized_span_dataset(CLRUTERM3_TRAIN1_CANDIDATES_PATH, tokenizer=tokenizer, has_preprocessed_candidates=True)
# # row = next(iter(ds))
# # print(len(ds))


cnt = {}
with open(r'D:\progproj\DIPLOM\NN-ATE\data\CL-RuTerm3\processed\train_t1_candidates.jsonl', 'r', encoding='utf-8-sig') as f:
    for line in f:
        js = json.loads(line)
        candidates = js['candidates']
        # print(type(candidates), candidates)
        cnt[js['id']] = len(candidates)
        print(len(candidates))
print(sum(cnt.values()))
# print(*(f"{k}: {v}" for k, v in row.items()), sep="\n")