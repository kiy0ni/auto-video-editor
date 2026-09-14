"""Swear word handling: helping the transcriber write them, and masking them on request."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Set

# Words the transcriber is told about (so it writes them instead of skipping them) and that the
# censor masks. Kept short on purpose: users can extend the censor list in the settings.
PROFANITY = {
    "fr": ["putain", "merde", "bordel", "connard", "connasse", "salope", "salaud", "enculé", "enculer",
           "pute", "chier", "foutre", "niquer", "nique", "gueule", "batard", "bâtard", "fdp", "ntm", "pd"],
    "en": ["fuck", "fucking", "fucked", "shit", "bullshit", "bitch", "asshole", "ass", "damn", "bastard",
           "dick", "crap", "cunt", "motherfucker", "wtf"],
}

# Short and natural on purpose: a long list of swear words makes small models hallucinate them.
PROMPT_HINTS = {
    "fr": "Putain, c'est chaud ! Merde, fais chier.",
    "en": "Holy shit, that's insane! Fuck.",
    "es": "¡Joder, qué locura! Mierda.",
    "de": "Scheiße, das ist verrückt! Verdammt.",
    "it": "Cazzo, è assurdo! Merda.",
    "pt": "Porra, que loucura! Merda.",
}

_VOWELS = set("aeiouyàâäéèêëîïôöùûüáíóúãõ")


def normalize_word(word: str) -> str:
    text = unicodedata.normalize("NFKD", word.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", text)


def profanity_set(extra: Iterable[str] = ()) -> Set[str]:
    words = {normalize_word(w) for group in PROFANITY.values() for w in group}
    words |= {normalize_word(w) for w in extra}
    return {w for w in words if w}


def is_profane(word: str, words: Set[str]) -> bool:
    token = normalize_word(word)
    if not token:
        return False
    return token in words or (token.endswith("s") and token[:-1] in words)


def censor_word(word: str) -> str:
    """``putain`` -> ``p*tain``: the first vowel after the initial letter becomes a star."""
    letters = [i for i, ch in enumerate(word) if ch.isalpha()]
    if len(letters) < 2:
        return word
    target = next((i for i in letters[1:] if word[i].lower() in _VOWELS), letters[1])
    return word[:target] + "*" + word[target + 1:]


def censor_text(text: str, words: Set[str]) -> str:
    return re.sub(r"[\w'’\-]+", lambda m: censor_word(m.group(0)) if is_profane(m.group(0), words) else m.group(0),
                  text, flags=re.UNICODE)
