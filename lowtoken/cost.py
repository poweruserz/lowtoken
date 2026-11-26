"""Fast local estimate of Claude token counts.

Claude's tokenizer is not public. `estimate` is a linear model over cheap
pre-token features, fitted against exact counts of 2,398 real tool outputs,
file reads, prompts and replies measured with `oracle.count` (Claude Opus 5.5).

Cross-validated error per sample: 9.1% pure Python, 5.8% when tiktoken is
installed (o200k is then one more feature). Summed over many samples the bias
is under 0.1%. For comparison, chars/4 undercounts Claude by 48% and raw
o200k by 34% on the same data.
"""
from __future__ import annotations

import re

_PIECE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d|[ \t]+|\n+|[^\sA-Za-z\d]")
_TR = frozenset("çğıöşüÇĞİÖŞÜ")

# shortcut: fitted once on Opus 5.5 counts; refit with bench/calibrate.py when the tokenizer changes
PURE = {"bias": 0.5089, "nl": 1.5251, "sp": 0.0325, "digit": 0.847, "punct": 0.8658, "tr": 1.852,
        "uni": 0.595, "cjk": 0.6298, "w1": 1.1925, "w2": 1.4027, "w3": 1.7396, "w4": 1.1807,
        "w5": 1.9945, "w6": 1.9762, "w7": 1.7862, "w8": 3.7339, "w9": 3.6182, "w10": 3.1684,
        "w11": 2.3263, "w12": 2.5428}
WITH_O200K = {"bias": -1.8818, "nl": 0.9038, "sp": -0.2774, "digit": 0.0577, "punct": 0.2491,
              "tr": 1.2073, "uni": 0.1907, "cjk": 0.0553, "w1": 0.0146, "w2": 0.6138, "w3": 0.6731,
              "w4": 0.6241, "w5": 0.9853, "w6": 1.3343, "w7": 1.0741, "w8": 1.7263, "w9": 1.3036,
              "w10": 1.4275, "w11": 2.0712, "w12": 1.3599, "o200k": 1.0467}

_O200K = None  # set by precise(); loading o200k takes ~0.3 s, too slow for a hook


def precise() -> bool:
    """Use tiktoken's o200k as an extra feature (5.8% instead of 9.1% error) if installed."""
    global _O200K
    if _O200K is None:
        try:
            import tiktoken
            _O200K = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001 - any import/setup failure means "not available"
            return False
    return True


def features(text: str) -> dict[str, float]:
    f = dict.fromkeys(PURE, 0.0)
    f["bias"] = 1.0
    for p in _PIECE.findall(text):
        c = p[0]
        if c.isascii() and c.isalpha():
            f["w%d" % min(len(p), 12)] += 1
        elif c.isdigit():
            f["digit"] += 1
        elif c == "\n":
            f["nl"] += 1
        elif c in " \t":
            f["sp"] += 1
        elif c.isascii():
            f["punct"] += 1
        elif c in _TR:
            f["tr"] += 1
        elif ord(c) < 0x3000:
            f["uni"] += 1
        else:
            f["cjk"] += 1
    return f


def estimate(text: str) -> int:
    """Estimated Claude token count of `text`."""
    if not text:
        return 0
    f = features(text)
    if _O200K is not None:
        f["o200k"] = len(_O200K.encode(text, disallowed_special=()))
        w = WITH_O200K
    else:
        w = PURE
    return max(1, round(sum(w[k] * v for k, v in f.items())))
