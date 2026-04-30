import json
from term_datasets._types import CLRuTerm3OriginalJSON

with open(
        r'data/CL-RuTerm3/processed/train_t1_candidates.jsonl',
        'r', encoding='utf-8-sig') as f:
    lines: list[CLRuTerm3OriginalJSON] = [json.loads(line) for line in f]

print(lines[0]['candidates'])
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
