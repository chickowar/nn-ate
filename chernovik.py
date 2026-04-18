from collections import Counter
from json import loads

with open('./data/CL-RuTerm3/original/train_t1_v1.jsonl', encoding='utf-8') as f:
    lines: list = [loads(line) for line in f]
texts: list[str] = [line['text'] for line in lines]
spans: list[tuple[int, int]] = [line['label'] for line in lines]
labels: list[list[str]] = [[text[i: j] for i, j in span] for text, span in zip(texts, spans)]
all_labels: list[str] = [label for text_labels in labels for label in text_labels]
label_lengths: Counter = Counter(
    iter(len(label.split()) for label in all_labels)
)
print(sum((len(text) for text in texts)) / len(texts))
print(sum((len(text.split()) for text in texts)) / len(texts))