from typing import Iterator, Callable

from transformers import TokenizersBackend

from common.utils import clr
from term_datasets.CL_RuTerm3 import SPECIAL_TOKEN_LABEL_ID, BIO2ID
from term_datasets._types import CLRuTerm3TokenizedElement


def read_tokenized_element(el: CLRuTerm3TokenizedElement,
                           tokenizer: TokenizersBackend,
                           skip_special_tokens: bool = True,
                           highlight_in_text: bool = True,
                           highlight_left: str = clr.BOLD + clr.UNDERLINE,
                           highlight_right: str = clr.ENDC,
                           ) -> str:  # с типами разобратсья
    """For flat terms correctly annotated in BIO returns a text with highlighted terms"""
    tokens: list[str] = tokenizer.convert_ids_to_tokens(el['input_ids'], skip_special_tokens=skip_special_tokens)
    ret: list[str] = []
    labels: list[int] = el['labels']
    labels_iter: Iterator[int] = iter(labels) if not skip_special_tokens \
        else filter(lambda x: x != SPECIAL_TOKEN_LABEL_ID, labels)
    prev_label = BIO2ID['O']
    prefix_mapping: dict[int, Callable[[int, str], str]] = {
        BIO2ID['B-TERM']: lambda prev, space: space + highlight_left,
        BIO2ID['I-TERM']: lambda prev, space: space,
        BIO2ID['O']: lambda prev, space: highlight_right + space if prev != BIO2ID['O'] else space,
    }
    for token, label in zip(tokens, labels_iter):
        prefix = '' if token.startswith('##') else ' '  # added space
        if highlight_in_text:
            ret.append(prefix_mapping[label](prev_label, prefix))
        else:
            ret.append(prefix)
        ret.append(token.lstrip('##'))
        prev_label = label

    return ''.join(ret)