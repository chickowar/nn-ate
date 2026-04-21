from __future__ import annotations

import re
from string import punctuation
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSpan:
    text: str
    start: int
    end: int


class TextProcessor:
    """Lightweight text preprocessor for sentence splitting and word n-gram extraction."""

    _sentence_boundary_pattern = re.compile(r"(?:\.\s+|\n+)")
    _word_pattern = re.compile(r"(?u)\w+")
    _left_boundary_chars = "\"'«„“([{"
    _right_boundary_chars = "\"'»”’)]}.,:;!?"

    def split_sentences(self, text: str) -> list[TextSpan]:
        sentences: list[TextSpan] = []
        last_end = 0

        for match in self._sentence_boundary_pattern.finditer(text):
            sentence_end = match.start() + 1 if text[match.start()] == "." else match.start()
            sentence_text = text[last_end:sentence_end].strip()
            if sentence_text:
                stripped_start = last_end + (len(text[last_end:sentence_end]) - len(text[last_end:sentence_end].lstrip()))
                stripped_end = sentence_end - (len(text[last_end:sentence_end]) - len(text[last_end:sentence_end].rstrip()))
                sentences.append(TextSpan(text=text[stripped_start:stripped_end], start=stripped_start, end=stripped_end))
            last_end = match.end()

        tail_text = text[last_end:].strip()
        if tail_text:
            stripped_start = last_end + (len(text[last_end:]) - len(text[last_end:].lstrip()))
            stripped_end = len(text) - (len(text[last_end:]) - len(text[last_end:].rstrip()))
            sentences.append(TextSpan(text=text[stripped_start:stripped_end], start=stripped_start, end=stripped_end))

        if not sentences and text.strip():
            stripped_text = text.strip()
            start = text.find(stripped_text)
            sentences.append(TextSpan(text=stripped_text, start=start, end=start + len(stripped_text)))

        return sentences

    def extract_word_spans(self, sentence: TextSpan) -> list[TextSpan]:
        return [
            TextSpan(
                text=sentence.text[match.start():match.end()],
                start=sentence.start + match.start(),
                end=sentence.start + match.end(),
            )
            for match in self._word_pattern.finditer(sentence.text)
        ]

    def extract_ngrams(self, sentence: TextSpan, max_n: int) -> list[TextSpan]:
        word_spans = self.extract_word_spans(sentence)
        ngrams: list[TextSpan] = []
        seen_boundaries: set[tuple[int, int]] = set()

        for start_index in range(len(word_spans)):
            max_end_index = min(start_index + max_n, len(word_spans))
            for end_index in range(start_index, max_end_index):
                base_start = word_spans[start_index].start
                base_end = word_spans[end_index].end
                for start_char in self._expand_left_boundaries(sentence, base_start):
                    for end_char in self._expand_right_boundaries(sentence, base_end):
                        if (start_char, end_char) in seen_boundaries:
                            continue
                        seen_boundaries.add((start_char, end_char))
                        ngram_text = sentence.text[start_char - sentence.start:end_char - sentence.start]
                        if ngram_text:
                            ngrams.append(TextSpan(text=ngram_text, start=start_char, end=end_char))
        return ngrams

    def _expand_left_boundaries(self, sentence: TextSpan, start: int) -> list[int]:
        starts = [start]
        current = start
        while current > sentence.start and sentence.text[current - sentence.start - 1] in self._left_boundary_chars:
            current -= 1
            starts.append(current)
        return starts

    def _expand_right_boundaries(self, sentence: TextSpan, end: int) -> list[int]:
        ends = [end]
        current = end
        while current < sentence.end and sentence.text[current - sentence.start] in self._right_boundary_chars:
            current += 1
            ends.append(current)
        return ends
