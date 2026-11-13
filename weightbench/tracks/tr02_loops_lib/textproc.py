"""Pure text utilities for TR02-LOOP: loop detection, answer extraction, math grading.

No torch / transformers imports here: this module is unit-tested on CPU in milliseconds.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Hashable, Sequence

# --------------------------------------------------------------------------------------
# Loop detection
# --------------------------------------------------------------------------------------
LOOP_NGRAM = 16          # n-gram length (tokens) used to detect verbatim repetition
LOOP_WINDOW = 512        # tail window (tokens) inspected
LOOP_MIN_REP = 0.5       # >= this fraction of the tail is covered by repeated n-grams


def repetition_fraction(seq: Sequence[Hashable], n: int = LOOP_NGRAM, window: int = LOOP_WINDOW) -> float:
    """Fraction of n-gram start positions in the last ``window`` items whose n-gram already
    occurred earlier in ``seq``. 0 for sequences shorter than 2n."""
    L = len(seq)
    if L < 2 * n:
        return 0.0
    seen: set[tuple] = set()
    start_tail = max(0, L - window)
    hits = total = 0
    for i in range(0, L - n + 1):
        g = tuple(seq[i:i + n])
        if i >= start_tail:
            total += 1
            if g in seen:
                hits += 1
        seen.add(g)
    return hits / total if total else 0.0


def loop_stats(seq: Sequence[Hashable], finished: bool, n: int = LOOP_NGRAM, window: int = LOOP_WINDOW,
               min_rep: float = LOOP_MIN_REP) -> dict:
    """A generation *loops* iff it hit the token limit (``finished`` False, no EOS) AND its tail
    is dominated by verbatim repetition. Truncated-but-not-repetitive is reported separately."""
    rep = repetition_fraction(seq, n, window)
    return {"looped": (not finished) and rep >= min_rep, "truncated": not finished, "rep_frac": round(rep, 4)}


def loop_onset(seq: Sequence[Hashable], n: int = LOOP_NGRAM) -> int | None:
    """Index of the first token from which (almost) every n-gram is a repeat, i.e. where the
    terminal loop starts. Returns None if the sequence does not end in a repetition run."""
    L = len(seq)
    if L < 2 * n:
        return None
    seen: set[tuple] = set()
    rep = [False] * (L - n + 1)
    for i in range(L - n + 1):
        g = tuple(seq[i:i + n])
        rep[i] = g in seen
        seen.add(g)
    if not rep[-1]:
        return None
    # walk back while repeats dominate (tolerate isolated non-repeats)
    i = len(rep) - 1
    miss = 0
    onset = i
    while i >= 0:
        if rep[i]:
            onset, miss = i, 0
        else:
            miss += 1
            if miss > n:
                break
        i -= 1
    return onset


# --------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------
THINK_END = "</think>"


def split_thinking(gen: str) -> tuple[str, str | None]:
    """Generated text (after the '<think>\\n' prompt suffix) -> (thinking, final answer or None).
    No closing </think> => the model never finished thinking => no answer."""
    if THINK_END not in gen:
        return gen, None
    th, ans = gen.split(THINK_END, 1)
    return th, ans.strip()


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...} with brace matching."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # "\boxed 5" form
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def parse_number(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.replace("\\!", "").replace("\\,", "").replace("{,}", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("$", "").replace("\\%", "").replace("%", "").strip()
    m = re.fullmatch(r"\s*(-?\d[\d,]*(?:\.\d+)?|-?\.\d+)\s*", s)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    fr = re.fullmatch(r"\s*(-?)\\[dt]?frac\{(-?\d+)\}\{(-?\d+)\}\s*", s)
    if fr and int(fr.group(3)) != 0:
        v = int(fr.group(2)) / int(fr.group(3))
        return -v if fr.group(1) else v
    fr = re.fullmatch(r"\s*(-?\d+)\s*/\s*(-?\d+)\s*", s)
    if fr and int(fr.group(2)) != 0:
        return int(fr.group(1)) / int(fr.group(2))
    return None


def gsm8k_gold(answer_field: str) -> float:
    return float(answer_field.split("####")[-1].strip().replace(",", ""))


def gsm8k_pred(final: str | None) -> float | None:
    """Prefer the last \\boxed{}; else the last number in the final answer."""
    if final is None:
        return None
    b = last_boxed(final)
    if b is not None:
        v = parse_number(b)
        if v is not None:
            return v
        nums = _NUM.findall(b)
        return float(nums[-1].replace(",", "")) if nums else None
    nums = _NUM.findall(final)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def gsm8k_correct(final: str | None, gold: float) -> bool:
    p = gsm8k_pred(final)
    return p is not None and abs(p - gold) < 1e-4 * max(1.0, abs(gold))


# ---- MATH normalisation (after Hendrycks et al. `strip_string`, simplified) ----------

def _fix_fracs(s: str) -> str:
    subs = s.split("\\frac")
    out = subs[0]
    for sub in subs[1:]:
        out += "\\frac"
        if sub and sub[0] == "{":
            out += sub
        elif len(sub) >= 2:
            a, b, rest = sub[0], sub[1], sub[2:]
            if b == "{":
                out += "{" + a + "}" + b + rest
            else:
                out += "{" + a + "}{" + b + "}" + rest
        else:
            out += sub
    return out


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def _fix_a_slash_b(s: str) -> str:
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}" if m else s


def normalize_math(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().replace("\n", "").rstrip(".")
    s = s.replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\ ", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    # "x = 5" -> "5" (short lhs only)
    if "=" in s and len(s.split("=")[0].strip()) <= 2:
        s = s.split("=", 1)[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    s = re.sub(r"(\d),(\d\d\d)(?!\d)", r"\1\2", s)  # 1,000 -> 1000
    return s


def _as_number(s: str) -> Fraction | None:
    v = parse_number(s)
    if v is None:
        return None
    try:
        return Fraction(v).limit_denominator(10 ** 6)
    except (ValueError, OverflowError):
        return None


def math_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    a, b = normalize_math(pred), normalize_math(gold)
    if a == b:
        return True
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return abs(float(na) - float(nb)) <= 1e-6 * max(1.0, abs(float(nb)))
    # tuples / sets: compare element-wise
    if "," in a and "," in b:
        pa = [x for x in re.split(r",", a.strip("()[]\\{}")) if x]
        pb = [x for x in re.split(r",", b.strip("()[]\\{}")) if x]
        if len(pa) == len(pb) and a[:1] == b[:1]:
            return all(math_equal(x, y) for x, y in zip(pa, pb))
    return False


def math_correct(final: str | None, gold: str) -> bool:
    if final is None:
        return False
    return math_equal(last_boxed(final), gold)
