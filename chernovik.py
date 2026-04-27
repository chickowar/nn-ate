import json

from term_datasets.CL_RuTerm3_getters import get_raw_dataset
from term_datasets._types import CLRuTerm3OriginalJSON, CLRuTerm3OutputJSON

with open(r'D:\progproj\DIPLOM\NN-ATE\data\CL-RuTerm3\processed\test1_t1_candidates.jsonl', encoding='utf-8-sig') as f:
    my_test: list[CLRuTerm3OriginalJSON] = [json.loads(line) for line in f]
    my_ids = list(map(lambda x: x['id'], my_test))
with open(r'D:\progproj\DIPLOM\NN-ATE\data\CL-RuTerm3\original\test1_t12_full_v2.jsonl', encoding='utf-8-sig') as f:
    true_test: list[CLRuTerm3OriginalJSON] = [json.loads(line) for line in f]

for js in true_test:
    id = js['id']
    if id not in my_ids:
        print(js['id'], js['text'], sep='\n')
