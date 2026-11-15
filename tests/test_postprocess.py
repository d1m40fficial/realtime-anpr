"""Unit tests for GOST plate normalisation (no model weights required).

All fixtures use the neutral GOST documentation example ``A123BC177`` so that
no real vehicle identifiers end up in the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anpr.postprocess import (  # noqa: E402
    PLATE_REGEX,
    is_strict_plate,
    normalize_plate_text,
    postprocess_plate,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("А123ВС177", "A123BC177"),     # Cyrillic look-alikes
        ("a123bc177", "A123BC177"),     # lower case
        ("A 123 BC 177", "A123BC177"),  # separators
        ("A123BC177RUS", "A123BC177"),  # country block stripped
        ("A123BC17G", "A123BC176"),     # G -> 6 in the region code
        ("A123BCI77", "A123BC177"),     # I -> 1 in the region code
        ("A123BC99", "A123BC99"),       # two-digit region
    ],
)
def test_postprocess_repairs_plate(raw: str, expected: str) -> None:
    assert postprocess_plate(raw) == expected


def test_normalize_keeps_only_alnum() -> None:
    assert normalize_plate_text("а123вс|77!") == "A123BC77"


@pytest.mark.parametrize(
    "text, valid",
    [
        ("A123BC77", True),
        ("A123BC177", True),
        ("A123B77", False),
        ("A123BC7777", False),
        ("AB123C77", False),
    ],
)
def test_plate_regex(text: str, valid: bool) -> None:
    assert bool(PLATE_REGEX.match(text)) is valid


@pytest.mark.parametrize(
    "text",
    ["A123BC177", "1234BC77", "A123BC7777"],
)
def test_repair_always_yields_valid_plate(text: str) -> None:
    # The repair step is deliberately aggressive: a broken string is snapped to
    # the nearest valid plate instead of being rejected outright.
    assert is_strict_plate(text)


def test_noise_does_not_survive() -> None:
    assert "RUS" not in postprocess_plate("A123BC177 RUSSIA")


def test_garbage_is_not_a_plate() -> None:
    assert not is_strict_plate("")
    assert not is_strict_plate("RUS")
    assert not is_strict_plate("0000")
