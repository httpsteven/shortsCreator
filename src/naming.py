"""
Lookup-key construction and title normalization.

This module exists so the key format lives in exactly ONE place. The pipeline
builds keys to look quotes up, and scripts/export_titles.py builds keys to write
prompts with. If those two ever drift by so much as a space, every lookup misses
and the pipeline reports "no quotes for this episode" for an entire library
while looking completely healthy.

Import `build_tv_key` / `build_movie_key` from here. Do not re-implement them.
"""

from __future__ import annotations

import difflib
import re

# Titles in a hand-assembled quotes.json drift from titles parsed off disk:
# "Pt. 1" vs "Part 1", stray punctuation, different dash characters. Exact match
# is tried first; this is the fallback threshold for a normalized comparison.
FUZZY_KEY_THRESHOLD = 0.85


def build_tv_key(show: str, season: int, episode: int, title: str) -> str:
    """
    The canonical TV lookup key: "Show - S01E03 - Episode Title".

    Used by both the pipeline and the prompt-export script. Changing this format
    invalidates every existing quotes.json key, so don't, unless you also
    migrate the file.
    """
    key = f"{show.strip()} - S{season:02d}E{episode:02d}"
    title = title.strip()
    if title:
        key = f"{key} - {title}"
    return key


def build_movie_key(title: str, year: int | None = None) -> str:
    """
    The canonical movie lookup key: the bare title.

    The year is deliberately NOT part of the key. People write "Heat" in a
    quotes file, not "Heat (1995)", and two films sharing a title in one library
    is rare enough to handle by hand when it happens.
    """
    return title.strip()


def normalize_key(value: str) -> str:
    """
    Reduce a key to its comparable core: lowercase, alphanumerics only.

    Collapses every difference that doesn't change meaning — punctuation,
    spacing, dash style, "&" vs "and" is NOT handled (deliberately: it changes
    the word), so "Malcolm in the Middle - S01E03 - Home Alone 4" and
    "malcolm in the middle s01e03 home alone 4" compare equal.
    """
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def find_key(wanted: str, available: list[str]) -> str | None:
    """
    Resolve a lookup key against the keys actually present in quotes.json.

    Three passes, cheapest first:
      1. exact
      2. normalized exact  (punctuation/spacing drift)
      3. normalized fuzzy  (wording drift, >= FUZZY_KEY_THRESHOLD)

    Returns the matching key as written in the file, or None.
    """
    if wanted in available:
        return wanted

    normalized = {normalize_key(key): key for key in available}
    target = normalize_key(wanted)

    if target in normalized:
        return normalized[target]

    best_key: str | None = None
    best_ratio = 0.0
    for candidate_norm, original in normalized.items():
        ratio = difflib.SequenceMatcher(None, target, candidate_norm).ratio()
        if ratio > best_ratio:
            best_ratio, best_key = ratio, original

    if best_ratio >= FUZZY_KEY_THRESHOLD:
        return best_key
    return None


# --------------------------------------------------------------------------
# Filename cleanup
# --------------------------------------------------------------------------

# Release-scene tokens. Everything from the first of these onward is noise, not
# title — "Home Alone 4 1080p BluRay x264" should become "Home Alone 4".
_JUNK_TOKEN = re.compile(
    r"\b("
    r"\d{3,4}p|[xh]\.?26[45]|hevc|avc|aac\d?|ac3|eac3|dts(?:-?hd)?|truehd|atmos"
    r"|bluray|blu-ray|bdrip|brrip|webrip|web-?dl|hdtv|dvdrip|remux|repack|proper"
    r"|hdr\d*|dv|sdr|10bit|8bit|multi|dual|subbed|dubbed"
    r")\b.*",
    re.IGNORECASE,
)

# A trailing "-GROUP" release tag. Anchored to the end so hyphenated titles
# ("Spider-Man") survive.
_GROUP_TAG = re.compile(r"[-\s]+[A-Za-z0-9]{2,}$")

# A year in brackets is an explicit marker and always wins. A bare year is only
# a guess — "Blade Runner 2049 (2017)" has two four-digit numbers and only one
# of them is the release year.
_BRACKETED_YEAR = re.compile(r"[(\[]\s*(19\d{2}|20\d{2})\s*[)\]]")
_BARE_YEAR = re.compile(r"(?<![0-9])(19\d{2}|20\d{2})(?![0-9])")


def clean_title(raw: str) -> str:
    """
    Turn a filename fragment into a human title.

    Dots and underscores become spaces before junk stripping, because scene
    names use them as separators ("Home.Alone.4.1080p").
    """
    text = raw.replace("_", " ")
    # Only treat dots as separators when the name clearly uses them that way;
    # otherwise "Mr. Robot" loses its period.
    if text.count(".") >= 2 and " " not in text:
        text = text.replace(".", " ")

    text = _JUNK_TOKEN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—._")
    return text


def extract_year(raw: str) -> tuple[str, int | None]:
    """
    Pull a release year out of a title, returning (title without it, year).

    A bracketed year wins outright. Otherwise the LAST bare year is taken,
    because scene names append it after the title —
    "Blade.Runner.2049.2017.1080p" — so the last one is the release year and
    any earlier one belongs to the title itself.

    A bare year is never stripped if doing so would empty the title, which is
    what keeps films actually named for a year ("1917", "2012") intact.
    """
    match = _BRACKETED_YEAR.search(raw)
    if match:
        without = raw[: match.start()] + raw[match.end() :]
        return re.sub(r"\s+", " ", without).strip(" -–—()[]"), int(match.group(1))

    bare = list(_BARE_YEAR.finditer(raw))
    if not bare:
        return raw.strip(), None

    last = bare[-1]
    without = raw[: last.start()] + raw[last.end() :]
    without = re.sub(r"\s+", " ", without).strip(" -–—()[]")
    if not without:
        return raw.strip(), None

    return without, int(last.group(1))
