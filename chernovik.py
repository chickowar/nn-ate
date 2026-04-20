from term_datasets.CL_RuTerm3 import CLRUTERM3_TRAIN1_PATH, get_flat_terms, build_span_classification_dataset
from term_datasets.CL_RuTerm3_getters import get_raw_dataset


from collections import defaultdict

raw_ds = get_raw_dataset(CLRUTERM3_TRAIN1_PATH)
raw_terms = {row['id']: set((tuple(label) for label in row["label"])) for row in raw_ds}

ds = build_span_classification_dataset(raw_ds, max_words_per_ngram=7)


cnt = 0
for row in ds:
    row_id = row['id']
    cand = row['candidate_text']
    span = row['span_start'], row['span_end']
    text = row['text']
    label = row['label']
    cnt += label
    # print(f"{'':=^100}\n{row_id}\nCandidate: {cand:<30} | {span}\nText: {text}\nLabel: {'TERM' if label else 'NON-TERM'}\n")

print(f"Примеров: {len(ds)}\nТерминов: {cnt}")
print(f"Реальных терминов: {sum(len(x) for x in raw_terms.values())}")
# cands: dict[str, set[tuple[int, int]]] = defaultdict(set)
#
# for row in ds:
#     _id = row['id']
#     boundary = row['span_start'], row['span_end']
#     cands[_id].add(boundary)
#
# print("=========")
#
# unincluded = {}
# for _id, true_terms in raw_terms.items():
#     _unincluded = true_terms - cands[_id]
#     if _unincluded:
#         print(f"{_id} : {_unincluded}")
#         unincluded[_id] = _unincluded
#
# print(unincluded)
# print(sum(len(terms) for terms in unincluded.values()))