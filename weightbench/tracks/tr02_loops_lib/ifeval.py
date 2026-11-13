"""Programmatic IFEval checkers (strict mode) for a supported subset of instruction ids.

Re-implemented after google-research/instruction_following_eval (Zhou et al., 2023) so the
benchmark has no extra dependency. Prompts whose instructions are not all supported are
dropped at build time. ``language:response_language`` (needs langdetect) is not supported.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

Checker = Callable[[str, dict[str, Any]], bool]


def _cmp(n: int, relation: str | None, target: int) -> bool:
    if relation == "less than":
        return n < target
    if relation == "at least":
        return n >= target
    raise ValueError(relation)


def _words(t: str) -> list[str]:
    return re.findall(r"\b\w+\b", t)


def _sentences(t: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", t.strip())
    return [p for p in parts if p.strip()]


def no_comma(r, kw):
    return "," not in r


def number_words(r, kw):
    return _cmp(len(_words(r)), kw["relation"], kw["num_words"])


def number_sentences(r, kw):
    return _cmp(len(_sentences(r)), kw["relation"], kw["num_sentences"])


def forbidden_words(r, kw):
    return not any(re.search(r"\b" + re.escape(w) + r"\b", r, re.I) for w in kw["forbidden_words"])


def highlighted_sections(r, kw):
    n = 0
    for h in re.findall(r"\*[^\n\*]*\*", r):
        if h.strip("*").strip():
            n += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", r):
        if h.removeprefix("**").removesuffix("**").strip():
            n += 1
    return n >= kw["num_highlights"]


def keyword_frequency(r, kw):
    n = len(re.findall(r"\b" + re.escape(kw["keyword"]) + r"\b", r, re.I))
    return _cmp(n, kw["relation"], kw["frequency"])


def repeat_prompt(r, kw):
    return r.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())


def quotation(r, kw):
    s = r.strip()
    return len(s) > 1 and s[0] == '"' and s[-1] == '"'


def lowercase(r, kw):
    return r == r.lower()


def capital(r, kw):
    return r == r.upper()


def keywords_existence(r, kw):
    return all(re.search(re.escape(k), r, re.I) for k in kw["keywords"])


def title(r, kw):
    return any(t.strip("<>").strip() for t in re.findall(r"<<[^\n]+>>", r))


def letter_frequency(r, kw):
    n = r.lower().count(kw["letter"].lower())
    return _cmp(n, kw["let_relation"], kw["let_frequency"])


def bullet_lists(r, kw):
    star = re.findall(r"^\s*\*[^\*].*$", r, re.M)
    dash = re.findall(r"^\s*-.*$", r, re.M)
    return len(star) + len(dash) == kw["num_bullets"]


def placeholders(r, kw):
    return len(re.findall(r"\[.*?\]", r)) >= kw["num_placeholders"]


def number_paragraphs(r, kw):
    paras = re.split(r"\s?\*\*\*\s?", r)
    n = len(paras)
    for i, p in enumerate(paras):
        if not p.strip():
            if i == 0 or i == len(paras) - 1:
                n -= 1
            else:
                return False
    return n == kw["num_paragraphs"]


def end_checker(r, kw):
    return r.strip().strip('"').lower().endswith(kw["end_phrase"].strip().lower())


def postscript(r, kw):
    m = kw["postscript_marker"].strip()
    pat = r"\s*p\.\s?p\.\s?s.*$" if m.lower() == "p.p.s" else (r"\s*p\.\s?s\..*$" if m.lower() == "p.s." else r"\s*" + re.escape(m.lower()) + r".*$")
    return bool(re.findall(pat, r.lower(), re.M))


def capital_word_frequency(r, kw):
    n = len([w for w in _words(r) if w.isupper()])
    return _cmp(n, kw["capital_relation"], kw["capital_frequency"])


def two_responses(r, kw):
    parts = r.split("******")
    valid = [p.strip() for p in parts if p.strip()]
    return len(valid) == 2 and valid[0] != valid[1]


def json_format(r, kw):
    s = r.strip().removeprefix("```json").removeprefix("```Json").removeprefix("```JSON").removeprefix("```")
    s = s.removesuffix("```").strip()
    try:
        json.loads(s)
    except ValueError:
        return False
    return True


def multiple_sections(r, kw):
    spl = kw["section_spliter"].strip()
    secs = re.split(r"\s?" + re.escape(spl) + r"\s?\d+\s?", r)
    return len(secs) - 1 >= kw["num_sections"]


def nth_paragraph_first_word(r, kw):
    paras = [p for p in re.split(r"\n\n", r) if p.strip()]
    if len(paras) != kw["num_paragraphs"]:
        return False
    n = kw["nth_paragraph"]
    if n > len(paras):
        return False
    words = paras[n - 1].strip().split()
    if not words:
        return False
    w = re.sub(r"[^\w]", "", words[0]).lower()
    return w == kw["first_word"].lower()


def constrained_response(r, kw):
    opts = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    return any(o in r.strip() for o in opts)


CHECKERS: dict[str, Checker] = {
    "punctuation:no_comma": no_comma,
    "length_constraints:number_words": number_words,
    "length_constraints:number_sentences": number_sentences,
    "keywords:forbidden_words": forbidden_words,
    "detectable_format:number_highlighted_sections": highlighted_sections,
    "keywords:frequency": keyword_frequency,
    "combination:repeat_prompt": repeat_prompt,
    "startend:quotation": quotation,
    "change_case:english_lowercase": lowercase,
    "keywords:existence": keywords_existence,
    "detectable_format:title": title,
    "keywords:letter_frequency": letter_frequency,
    "detectable_format:number_bullet_lists": bullet_lists,
    "detectable_content:number_placeholders": placeholders,
    "length_constraints:number_paragraphs": number_paragraphs,
    "startend:end_checker": end_checker,
    "detectable_content:postscript": postscript,
    "change_case:english_capital": capital,
    "change_case:capital_word_frequency": capital_word_frequency,
    "combination:two_responses": two_responses,
    "detectable_format:json_format": json_format,
    "detectable_format:multiple_sections": multiple_sections,
    "length_constraints:nth_paragraph_first_word": nth_paragraph_first_word,
    "detectable_format:constrained_response": constrained_response,
}


def supported(instruction_ids: list[str]) -> bool:
    return all(i in CHECKERS for i in instruction_ids)


def check(response: str | None, instruction_ids: list[str], kwargs: list[dict]) -> bool:
    """Prompt-level strict accuracy: every instruction must pass. No response => fail."""
    if response is None or not response.strip():
        return False
    for iid, kw in zip(instruction_ids, kwargs):
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        try:
            if not CHECKERS[iid](response, kw):
                return False
        except (KeyError, ValueError, TypeError):
            return False
    return True
