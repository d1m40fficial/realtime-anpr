"""GOST R 50577 plate normalisation.

The OCR model is a fixed-slot classifier: it always emits up to 9 characters
without any guarantee that the result is a valid Russian plate. Real-world
crops also contain the "RUS" country block, mounting bolts and neighbouring
characters, so raw predictions must be repaired into the canonical
``A123BC177`` shape.

Everything in this module is pure Python + regex, therefore it is cheap enough
to run on every frame and easy to unit-test.
"""

from __future__ import annotations

import re

PLATE_LETTERS = set("ABEKMHOPCTYX")
DIGITS = set("0123456789")

#: A -> 123 -> BC -> 2 or 3 digit region code.
PLATE_REGEX = re.compile(r"^[A-Z]\d{3}[A-Z]{2}\d{2,3}$")

#: Positions 0, 4 and 5 hold letters; 1-3 and 6-8 hold digits.
LETTER_POS = {0, 4, 5}
DIGIT_POS = {1, 2, 3, 6, 7, 8}

#: Cyrillic glyphs that are visually identical to Latin ones.
CYR_TO_LAT = {
    "А": "A",
    "В": "B",
    "Е": "E",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "У": "Y",
    "Х": "X",
}

#: Letter -> digit confusion (used when a digit is expected).
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "7", "T": "7", "V": "7", "Y": "7",
    "S": "5", "B": "8", "A": "4", "G": "6",
}

#: Digit -> letter confusion (used when a letter is expected).
DIGIT_TO_LETTER = {
    "0": "O", "D": "O", "Q": "O",
    "4": "A", "7": "T", "I": "T", "1": "T",
    "8": "B", "5": "S", "S": "C", "U": "Y",
    "V": "Y", "N": "H", "R": "P", "3": "E",
    "F": "E", "J": "T", "W": "M",
}

#: Region codes are always digits, so this map is applied unconditionally there.
FORCE_DIGIT_MAP = {
    "V": "7", "Y": "7", "Z": "7", "T": "7", "S": "5", "G": "6",
    "B": "8", "I": "1", "L": "1", "J": "1", "O": "0", "Q": "0",
    "D": "0", "A": "4",
}

#: Fragments produced by the "RUS" country block and its false positives.
NOISE_TOKENS = ("RUSSIA", "RUSE", "RUS", "PUS", "PVS", "RVS", "RU5", "PU5", "R0S", "KUS", "PLS")


def normalize_plate_text(text: str) -> str:
    """Upper-case, transliterate Cyrillic look-alikes and keep [A-Z0-9] only."""
    if not text:
        return ""
    out = text.upper()
    for cyr, lat in CYR_TO_LAT.items():
        out = out.replace(cyr, lat)
    return re.sub(r"[^A-Z0-9]", "", out)


def to_digit(ch: str) -> str:
    return ch if ch in DIGITS else LETTER_TO_DIGIT.get(ch, ch)


def to_letter(ch: str) -> str:
    return ch if ch in PLATE_LETTERS else DIGIT_TO_LETTER.get(ch, ch)


def _fix_prefix6(raw6: str) -> tuple[str, bool, int]:
    """Repair the letter-digit-letter-letter part of a plate.

    Returns the fixed 6 characters, whether it is structurally valid and how
    many substitutions were required (used to pick the cheapest candidate).
    """
    chars = list(raw6)
    changes = 0
    for i, ch in enumerate(chars):
        new_ch = to_letter(ch) if i in LETTER_POS else to_digit(ch)
        if new_ch != ch:
            changes += 1
        chars[i] = new_ch

    ok = (
        chars[0] in PLATE_LETTERS
        and chars[1] in DIGITS
        and chars[2] in DIGITS
        and chars[3] in DIGITS
        and chars[4] in PLATE_LETTERS
        and chars[5] in PLATE_LETTERS
    )
    return "".join(chars), ok, changes


def _force_region_digits(text: str) -> str:
    chars = list(text)
    for i in DIGIT_POS:
        if i < len(chars) and chars[i] in FORCE_DIGIT_MAP:
            chars[i] = FORCE_DIGIT_MAP[chars[i]]
    return "".join(chars)


def postprocess_plate(text: str) -> str:
    """Turn a raw OCR string into the best valid plate candidate.

    The search is exhaustive but tiny: for every start offset we try to build a
    valid plate and keep the candidate that required the fewest character
    substitutions. If nothing matches GOST, the cleaned string is returned so
    that callers can decide what to do with it.
    """
    t = normalize_plate_text(text)
    if not t:
        return ""
    for noise in NOISE_TOKENS:
        t = t.replace(noise, "")

    if len(t) < 6:
        return t
    # A 6-character reading is usually "A123BC" with the region cut off; the
    # last letter is more likely a "T" that the model merged with the border.
    if len(t) == 6 and t[5] in DIGITS:
        t = t[:5] + "T" + t[5:]
    if len(t) < 8:
        return t

    best: tuple[int, str] | None = None
    for start in range(0, len(t) - 5):
        fixed6, ok, changes6 = _fix_prefix6(t[start : start + 6])
        if not ok:
            continue

        tail_digits: list[str] = []
        tail_changes = 0
        for ch in t[start + 6 :]:
            digit = to_digit(ch)
            if digit in DIGITS:
                tail_digits.append(digit)
                if digit != ch:
                    tail_changes += 1

        for region_len in (3, 2):
            if len(tail_digits) < region_len:
                continue
            candidate = fixed6 + "".join(tail_digits[:region_len])
            if not PLATE_REGEX.match(candidate):
                continue
            changes = changes6 + tail_changes
            if best is None or changes < best[0]:
                best = (changes, candidate)

    return _force_region_digits(best[1]) if best else t


def is_strict_plate(text: str) -> bool:
    """True when the text is a fully valid GOST plate."""
    return bool(PLATE_REGEX.match(postprocess_plate(text)))


def is_soft_plate(text: str, min_len: int = 6) -> bool:
    """Loose check used before strict validation (6+ alphanumerics)."""
    return len(postprocess_plate(text)) >= min_len
