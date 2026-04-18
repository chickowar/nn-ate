import os
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from torch import Tensor
from torch.cuda import is_available as cuda_is_available
from transformers import AutoModelForTokenClassification, AutoTokenizer, TokenizersBackend, PreTrainedModel, \
    BatchEncoding, Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

from term_datasets.CL_RuTerm3 import MODEL_NAME, BIO_LABEL2ID, ID2BIO, SPECIAL_TOKEN_LABEL_ID
from term_datasets.CL_RuTerm3_getters import get_tokenized_dataset

#
# =========================== CONSTANTS ===========================

DEVICE: str = 'cuda' if cuda_is_available() else 'cpu'
MODEL_NAME: str = MODEL_NAME
LABEL2ID: dict[str, int] = BIO_LABEL2ID
ID2LABEL: dict[int, str] = ID2BIO
SPECIAL_TOKEN_LABEL_ID: int = SPECIAL_TOKEN_LABEL_ID


def get_tokenizer(model_name_or_path: str | Path = MODEL_NAME) -> TokenizersBackend:
    return AutoTokenizer.from_pretrained(model_name_or_path)


#
# =========================== Utils ===========================

def predict(
        data: str | BatchEncoding,
        model: PreTrainedModel,
        tokenizer: TokenizersBackend | None = None,
        output_type: Literal['intext', 'token_percentages', 'token_argmax', 'input_ids_percentages'] = 'token_percentages',
        percentage_temperature: float = 1,
        verbose: bool = False,
) -> list[tuple[str | int, dict[str, float]]]:
    if isinstance(data, str):
        assert isinstance(tokenizer, TokenizersBackend), f"Raw data (str) provided with no tokenizer"
        data: BatchEncoding = tokenizer.__call__(data)
        if verbose:
            print(f"Got batch encoding with keys [ ", *(f"{key}, " for key in data),
                  f"] and length {len(data['input_ids'])}",
                  sep='')

    assert output_type.startswith('token') and tokenizer is not None, 'If tokens are to be displayed, tokenizer should be passed'
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs: TokenClassifierOutput = model.__call__(
            **{k: torch.tensor([v], device=DEVICE) for k, v in data.items()})
        logits: Tensor = outputs.logits  # shape: [batch, seq_len, num_labels]
        if verbose: print(logits.shape, logits)
    percentages: Tensor = (logits / percentage_temperature).softmax(dim=-1).squeeze()
    input_ids: list[int] = data['input_ids']

    ret: list[tuple[str | int, dict[str, float]]]  = [
        (tokenizer.convert_ids_to_tokens(input_id) if output_type.startswith('token') else input_id,
         {_label: round(float(percentage[_id] * 100), 1) for _id, _label in ID2BIO.items()})
        for input_id, percentage in zip(input_ids, percentages)
    ]

    return ret


#
# =========================== main ===========================

def main():
    """
    Обучаем
    """
    original_path = Path(__file__).parents[2] / 'data' / 'CL-RuTerm3' / 'original' / 'train_t1_v1.jsonl'
    model_save_path = Path(__file__).parents[2] / 'runs' / ('CL-RuTerm3__' + MODEL_NAME.replace('/', '_'))
    tensorboard_log_dir = model_save_path.parent / 'tensorboard'
    os.environ['TENSORBOARD_LOGGING_DIR'] = str(tensorboard_log_dir)
    assert original_path.exists(), f'{original_path} does not exist. Check CL_RuTerm3.py file to change it'

    tokenizer: TokenizersBackend = get_tokenizer()

    dataset: DatasetDict = get_tokenized_dataset(original_path, tokenizer, batched=True, batch_size=16,
                                                 flat=True).train_test_split(test_size=0.15, seed=42)
    train_ds: Dataset = dataset["train"]
    eval_ds: Dataset = dataset["test"]

    model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(BIO_LABEL2ID),
        id2label=ID2BIO,
        label2id=BIO_LABEL2ID,
    )

    # predict(model, tokenizer, to_test, in_text=False)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(model_save_path),
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=10,
        use_cpu=False,

        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=30,
        eval_strategy="steps",
        eval_steps=30,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,

        report_to="tensorboard",
    )

    # Минимальная метрика: токен-accuracy по не(-100)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        mask = labels != SPECIAL_TOKEN_LABEL_ID
        correct = (preds[mask] == labels[mask]).sum()
        total = mask.sum()
        acc = (correct / total) if total > 0 else 0.0
        return {"token_accuracy": float(acc)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
    # checkpoint = "checkpoint-360"
    # model_path = Path(__file__).parents[2] / "runs" / "CL-RuTerm3__DeepPavlov_rubert-base-cased" / checkpoint
    # assert model_path.exists(), f'{model_path} does not exist'
    #
    # save_df: Path = model_path.parents[1] / "results" / "CL-RuTerm3__DeepPavlov_rubert-base-cased" / checkpoint
    # save_df.mkdir(parents=True, exist_ok=True)
    #
    # with open('test_string', 'r', encoding='utf-8') as f:
    #     to_test: str = f.read()
    #
    # tokenizer = get_tokenizer(model_path)
    # model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(LABEL2ID), id2label=LABEL2ID,
    #                                                         label2id=LABEL2ID)
    # data: BatchEncoding = tokenizer.__call__(to_test)
    # if len(data['input_ids']) > 512:
    #     data['input_ids'] = data['input_ids'][:512]
    #     data['attention_mask'] = data['attention_mask'][:512]
    #     data['token_type_ids'] = data['token_type_ids'][:512]
    #
    # prediction = predict(data, model, tokenizer, output_type='token_percentages', verbose=True)
    #
    # df = get_df_for_prediction(prediction, save_df / 'df.csv')
    # print(df)
