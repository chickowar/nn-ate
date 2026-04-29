import json
from term_datasets._types import CLRuTerm3OriginalJSON

with open(
        r'D:\progproj\DIPLOM\NN-ATE\data\CL-RuTerm3\for_final_evaluation\the_evaluation\Kaggle\track1\candidates\ruroberta-large\eval-test1-t1-ep60-seed38\checkpoint-3600\test2_t3_v2.jsonl',
        'r', encoding='utf-8-sig') as f:
    lines: list[CLRuTerm3OriginalJSON] = [json.loads(line) for line in f]

labels: dict[str, list[str]] = {
    js['id']: [js['text'][st: en] for st, en in js['label']] for js in lines
}

for i, doc in enumerate(labels):
    print('='*75)
    print(doc)
    if lines[i]['id'] != doc:
        raise Exception()

    print(lines[i]['text'])
    print('LABELS:')
    for label in labels[doc]:
        print(f"'{label}'")
