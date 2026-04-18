from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import TokenizersBackend

from term_datasets.CL_RuTerm3 import ID2BILOU, ID2BIO, BILOU2ID


def get_df_for_prediction(
        predictions: list[tuple[str, dict[str, float]]],
        save: str | Path | None = None,
) -> pd.DataFrame:
    """ For prediction of type [token, {label: percentage}] returns a Dataframe"""
    labels = predictions[0][1].keys()
    columns = ['Token', 'Label', *labels]

    df = pd.DataFrame([
        (token, '', *map(lambda l: percentages[l], labels))
        for token, percentages in predictions
    ], columns=columns)

    df['Label'] = df[labels].idxmax(axis=1)
    if save is not None: df.to_csv(save, index=False)
    return df


def quick_check_dataset(ds_bio: Dataset, ds_bilou, tokenizer: TokenizersBackend):
    for i, (bio, bilou) in enumerate(zip(ds_bio, ds_bilou)):
        input_ids = bio['input_ids']
        bio_labels = bio['labels']
        bilou_labels = bilou['labels']
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for token, bio_label, bilou_label in zip(tokens, bio_labels, bilou_labels):
            print(f"{token:<20} | {ID2BIO.get(bio_label, bio_label):<6} | {ID2BILOU.get(bilou_label, bilou_label)}")
        break

def find_u(ds_bilou, tokenizer: TokenizersBackend):
    for i, bilou in enumerate(ds_bilou):
        input_ids = bilou['input_ids']
        labels = bilou['labels']
        print(*map(
            lambda x: f"id: {x[0]:<8} | {tokenizer.decode(input_ids[x[0]]):<20} | {x[1]}\n",
            filter(lambda l: l[1] == BILOU2ID['U-TERM'], enumerate(labels))
        ),end="\n\n",sep='')