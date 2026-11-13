"""Procedural English-like corpus with 10 topic lexicons + a word-level tokenizer.

Every token is a lowercase word, punctuation mark, special token, or a single-character
"code" token written ``#x`` (x in a-z0-9). Codes such as ``qx4719`` are spelled with
character tokens; ``render`` joins consecutive character tokens without spaces.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np

TOPICS: dict[str, dict[str, list[str]]] = {
    "cooking": dict(
        roles="chef baker cook waiter butcher".split(),
        places="kitchen bakery restaurant".split(),
        nouns="soup bread flour oven pan garlic onion butter sugar salt pepper sauce recipe knife "
              "spoon dough cake pasta rice cheese".split(),
        verbs="baked stirred chopped boiled fried tasted seasoned roasted sliced peeled".split(),
        adjs="spicy sweet salty crispy tender delicious sour creamy".split()),
    "sports": dict(
        roles="player coach referee goalkeeper athlete".split(),
        places="stadium gym field".split(),
        nouns="ball goal team match score whistle trophy league net racket jersey season "
              "tournament sprint pass penalty fan medal helmet bat".split(),
        verbs="kicked scored passed tackled defended trained sprinted dribbled blocked won".split(),
        adjs="athletic fast fierce undefeated competitive tough agile winning".split()),
    "astronomy": dict(
        roles="astronomer astronaut stargazer cosmologist physicist".split(),
        places="observatory planetarium spaceport".split(),
        nouns="star planet comet galaxy telescope orbit moon asteroid nebula eclipse meteor "
              "constellation rocket satellite crater sun cosmos quasar supernova lens".split(),
        verbs="observed orbited launched mapped photographed measured tracked discovered glimpsed charted".split(),
        adjs="distant celestial bright lunar solar cosmic faint stellar".split()),
    "law": dict(
        roles="judge lawyer attorney juror prosecutor".split(),
        places="court courthouse prison".split(),
        nouns="verdict trial contract witness evidence statute jury appeal crime lawsuit defendant "
              "testimony warrant bail plea clause gavel motion subpoena felony".split(),
        verbs="sued testified convicted appealed ruled argued objected sentenced acquitted prosecuted".split(),
        adjs="guilty innocent legal illegal binding criminal judicial lawful".split()),
    "music": dict(
        roles="musician singer drummer pianist composer".split(),
        places="studio theater opera".split(),
        nouns="song guitar piano melody rhythm drum violin chord album concert lyric tune choir "
              "band note harmony tempo trumpet flute symphony".split(),
        verbs="sang played composed strummed hummed recorded rehearsed tuned performed conducted".split(),
        adjs="loud melodic harmonic musical rhythmic acoustic catchy jazzy".split()),
    "medicine": dict(
        roles="doctor nurse surgeon patient pharmacist".split(),
        places="hospital clinic pharmacy".split(),
        nouns="fever pill vaccine wound injection bandage symptom diagnosis surgery infection virus "
              "prescription stethoscope blood heart pulse antibiotic scalpel therapy illness".split(),
        verbs="diagnosed treated healed prescribed injected examined vaccinated bandaged cured operated".split(),
        adjs="sick healthy sterile chronic medical feverish contagious clinical".split()),
    "sea": dict(
        roles="sailor captain fisherman diver pirate".split(),
        places="harbor port dock".split(),
        nouns="ship boat wave anchor sail tide mast deck reef fish whale shark island compass "
              "coral oar shore seaweed voyage lighthouse".split(),
        verbs="sailed rowed anchored dived fished drifted navigated swam capsized moored".split(),
        adjs="nautical marine coastal tidal choppy deep oceanic briny".split()),
    "programming": dict(
        roles="programmer developer engineer hacker tester".split(),
        places="startup datacenter hackathon".split(),
        nouns="function compiler bug variable loop server database keyboard algorithm software "
              "script laptop python array debugger kernel syntax cache commit module".split(),
        verbs="compiled debugged coded deployed refactored programmed tested installed typed merged".split(),
        adjs="digital binary recursive buggy virtual robust parallel modular".split()),
    "farming": dict(
        roles="farmer shepherd rancher gardener harvester".split(),
        places="farm barn orchard".split(),
        nouns="tractor wheat corn cow sheep plow seed soil crop hay pig goat chicken fence harvest "
              "fertilizer potato apple pasture barley".split(),
        verbs="planted harvested plowed milked watered grazed sowed fed herded sheared".split(),
        adjs="fertile rural ripe organic muddy rustic golden leafy".split()),
    "finance": dict(
        roles="banker investor trader accountant broker".split(),
        places="bank market exchange".split(),
        nouns="money stock bond loan profit interest tax budget dividend portfolio debt credit coin "
              "cash fund asset price invoice salary share".split(),
        verbs="invested traded borrowed lent saved earned paid bought sold audited".split(),
        adjs="rich wealthy profitable financial expensive cheap fiscal bullish".split()),
}
TOPIC_NAMES = list(TOPICS)

NAMES = ("anna ben carla david elena frank grace henry iris jack kate leo maria nick olga paul "
         "rosa sam tara victor").split()
PRONOUNS = "she he they we i".split()
TIMES = [["yesterday"], ["today"], ["tonight"], ["later"], ["on", "monday"], ["on", "friday"],
         ["in", "the", "morning"], ["at", "night"]]
GPLACES = "house street town park shop school garden city".split()
GADJS = "big small old new happy tired cold warm quiet busy strange good".split()
GNOUNS = "friend book letter window table chair car dog cat phone bag key hat lamp".split()
ADVS = "quickly slowly carefully quietly happily".split()
GVERBS = "open find burn sell steal hide carry break".split()        # bare infinitives
CODE_NOUNS = "door box safe locker gate vault room lock card drawer".split()
CHARS = list(string.ascii_lowercase + string.digits)
CHAR_TOKENS = ["#" + c for c in CHARS]
FUNCTION = ("the a and with in at to on of was is who that said saw went needs every after before "
            "then his her their our my loves studied about very wanted code wrote for . ,").split()
SPECIAL = ["<pad>", "<bos>", "<eos>"]


def build_vocab() -> list[str]:
    words: list[str] = list(SPECIAL)
    seen = set(words)

    def add(ws):
        for w in ws:
            if w not in seen:
                seen.add(w)
                words.append(w)

    add(FUNCTION)
    for name, lex in TOPICS.items():
        add([name])
        for k in ("roles", "places", "nouns", "verbs", "adjs"):
            add(lex[k])
    add(NAMES + PRONOUNS + [w for t in TIMES for w in t] + GPLACES + GADJS + GNOUNS + ADVS + GVERBS
        + CODE_NOUNS + CHAR_TOKENS)
    return words


def check_lexicons() -> None:
    """Every topic word belongs to exactly one topic and to no general list."""
    general = set(FUNCTION + NAMES + PRONOUNS + [w for t in TIMES for w in t] + GPLACES + GADJS
                  + GNOUNS + ADVS + GVERBS + CODE_NOUNS)
    owner: dict[str, str] = {}
    for name, lex in TOPICS.items():
        for w in [name] + [w for k in lex for w in lex[k]]:
            assert w not in general, f"{w} is both topic and general"
            assert owner.setdefault(w, name) == name, f"{w} in {owner[w]} and {name}"


def topic_words(topic: str) -> list[str]:
    lex = TOPICS[topic]
    return [topic] + [w for k in ("roles", "places", "nouns", "verbs", "adjs") for w in lex[k]]


@dataclass
class Vocab:
    itos: list[str]

    def __post_init__(self):
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, toks: list[str]) -> list[int]:
        return [self.stoi[t] for t in toks]

    def decode(self, ids) -> list[str]:
        return [self.itos[int(i)] for i in ids]


def render(toks: list[str]) -> str:
    """Token list -> text. Character tokens (#x) are concatenated into one word."""
    out: list[str] = []
    prev_char = False
    for t in toks:
        if t in SPECIAL:
            continue
        if t.startswith("#") and len(t) == 2:
            if prev_char:
                out[-1] += t[1]
            else:
                out.append(t[1])
            prev_char = True
        else:
            out.append(t)
            prev_char = False
    return " ".join(out)


def spell(code: str) -> list[str]:
    return ["#" + c for c in code]


# ---------------------------------------------------------------------------------------
# sentence generators
# ---------------------------------------------------------------------------------------
class Gen:
    def __init__(self, rng: np.random.Generator, no_topic_name: bool = False):
        self.r = rng
        self.no_topic_name = no_topic_name  # never emit the topic-name templates (k = 7, 11)

    def c(self, xs):
        return xs[int(self.r.integers(len(xs)))]

    def code(self, lo: int = 3, hi: int = 6) -> list[str]:
        n = int(self.r.integers(lo, hi + 1))
        return [self.c(CHAR_TOKENS) for _ in range(n)]

    def topic_sentence(self, topic: str) -> list[str]:
        L = TOPICS[topic]
        c = self.c
        R, P, N, V, A = L["roles"], L["places"], L["nouns"], L["verbs"], L["adjs"]
        k = int(self.r.integers(13))
        while self.no_topic_name and k in (7, 11):
            k = int(self.r.integers(13))
        if k == 0:
            s = ["the", c(R), c(V), "the", c(A), c(N), "in", "the", c(P)]
        elif k == 1:
            s = [c(NAMES), c(V), "a", c(N), "with", "the", c(N)]
        elif k == 2:
            s = list(c(TIMES)) + ["the", c(R), "and", "the", c(R), c(V), "the", c(N)]
        elif k == 3:
            s = ["in", "the", c(P), ",", c(NAMES), "saw", "a", c(A), c(N)]
        elif k == 4:
            s = ["the", c(N), "was", c(A), "and", "the", c(N), "was", c(A)]
        elif k == 5:
            s = [c(PRONOUNS), c(V), "the", c(N), c(ADVS), "before", "the", c(N)]
        elif k == 6:
            s = ["every", c(R), "needs", "a", c(A), c(N)]
        elif k == 7:
            s = [c(NAMES), "is", "a", c(R), "who", "loves", topic]
        elif k == 8:
            s = ["the", c(R), "said", "that", "the", c(N), "was", "very", c(A)]
        elif k == 9:
            s = ["after", "the", c(N), ",", c(NAMES), "went", "to", "the", c(P)]
        elif k == 10:
            s = [c(NAMES), c(V), "the", c(N), "at", "the", c(P)] + list(c(TIMES))
        elif k == 11:
            s = [c(PRONOUNS), "studied", topic, "with", "a", c(A), c(R)]
        else:
            s = ["the", c(A), c(R), c(V), "the", c(N), "and", "then", c(V), "the", c(N)]
        return s + ["."]

    def general_sentence(self) -> list[str]:
        c = self.c
        k = int(self.r.integers(7))
        if k == 0:
            s = [c(NAMES), "went", "to", "the", c(GPLACES)] + list(c(TIMES))
        elif k == 1:
            s = ["the", c(GNOUNS), "was", c(GADJS), "and", c(GADJS)]
        elif k == 2:
            s = [c(NAMES), "and", c(NAMES), "saw", "a", c(GADJS), c(GNOUNS), "in", "the", c(GPLACES)]
        elif k == 3:
            s = [c(PRONOUNS), "wanted", "to", c(GVERBS), "the", c(GADJS), c(GNOUNS)]
        elif k == 4:
            s = [c(NAMES), "said", "that", "the", c(GPLACES), "was", "very", c(GADJS)]
        elif k == 5:
            s = ["my", c(GNOUNS), "is", "in", "the", c(GPLACES), c(ADVS)]
        else:
            s = [c(["his", "her", "their", "our"]), c(GNOUNS), "was", "on", "the", c(GNOUNS)]
        return s + ["."]

    def code_sentence(self) -> list[str]:
        c = self.c
        k = int(self.r.integers(5))
        if k == 0:
            s = ["the", c(CODE_NOUNS), "code", "is"] + self.code()
        elif k == 1:
            s = [c(NAMES), "wrote"] + self.code() + ["on", "the", c(GNOUNS)]
        elif k == 2:
            s = ["the", "code", "for", "the", c(CODE_NOUNS), "is"] + self.code()
        elif k == 3:
            s = ["the", c(CODE_NOUNS), "code", "of", c(NAMES), "is"] + self.code()
        else:
            s = [c(NAMES), "said", "the", "code", "is"] + self.code()
        return s + ["."]

    def document(self, topic: str | None, p_code: float = 0.15, p_general: float = 0.2,
                 n_lo: int = 3, n_hi: int = 7) -> list[list[str]]:
        """List of sentences. ``topic=None`` -> a random topic per document."""
        if topic is None:
            topic = self.c(TOPIC_NAMES)
        out = []
        for _ in range(int(self.r.integers(n_lo, n_hi + 1))):
            u = self.r.random()
            if u < p_code:
                out.append(self.code_sentence())
            elif u < p_code + p_general:
                out.append(self.general_sentence())
            else:
                out.append(self.topic_sentence(topic))
        return out


def flatten_doc(sents: list[list[str]]) -> list[str]:
    return ["<bos>"] + [t for s in sents for t in s] + ["<eos>"]
