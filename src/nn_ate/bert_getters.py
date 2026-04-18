from transformers import PreTrainedModel, AutoModelForTokenClassification, AutoModel, AutoModelForSequenceClassification

from term_datasets.CL_RuTerm3 import ID2BIO, BIO2ID

DEFAULT_MODEL = "DeepPavlov/rubert-base-cased"

def get_bert_token_classification(
        model_name_or_path=DEFAULT_MODEL,
        id2label = ID2BIO,
        label2id = BIO2ID,
):
    """
    Добавить выбор датасета
    :param model_name_or_path:
    :param id2label:
    :param label2id:
    :return:
    """
    model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    return model

def get_bert_sequence_classification(
        model_name_or_path=DEFAULT_MODEL,
        id2label = {0: 'O', 1: 'TERM'},
        label2id = {'O': 0, 'TERM': 1},
):
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    return model