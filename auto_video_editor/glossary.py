"""Built-in knowledge used by the automatic mode.

- Vocabulary packs: games and topics with their names and jargon, stored in ``data/vocabulary/*.json``,
  plus your own packs (``.json`` files in the ``vocabulary`` folder of the configuration directory).
  Chosen packs, and packs recognised from the title or a speech sample, are given to the transcriber
  and their tags become hashtags.
- Hype phrases per language, and laughter detection.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

DATA_DIR = Path(__file__).resolve().parent / "data" / "vocabulary"
CATEGORIES = ("Games", "Sports", "Streaming & culture", "Talk & podcast", "Tech & science", "Lifestyle",
              "Arts & entertainment")
# Characters of vocabulary given to the speech model: its text prompt is limited to ~220 tokens.
PROMPT_BUDGET = 450


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower().replace("’", "'"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


# -- vocabulary packs ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pack:
    id: str
    name: str
    category: str
    genre: str
    aliases: Tuple[str, ...]
    terms: Tuple[str, ...]
    tags: Tuple[str, ...]
    custom: bool = False

    @property
    def is_game(self) -> bool:
        return self.category == "Games"

    def vocabulary(self) -> List[str]:
        return [self.name] + [t for t in self.terms if normalize(t) != normalize(self.name)]

    def matches(self, query: str) -> bool:
        wanted = normalize(query)
        if not wanted:
            return True
        haystack = normalize(" ".join((self.id, self.name, self.category, self.genre, *self.aliases, *self.terms)))
        return all(part in haystack for part in wanted.split())


_packs: Optional[Dict[str, Pack]] = None
_errors: List[str] = []


def user_packs_dir() -> Path:
    from .settings import config_dir

    return config_dir() / "vocabulary"


def _unique(values: Iterable) -> Tuple[str, ...]:
    seen, out = set(), []
    for value in values or []:
        text = str(value).strip()
        key = normalize(text)
        if text and key and key not in seen:
            seen.add(key)
            out.append(text)
    return tuple(out)


def parse_pack(data: dict, custom: bool = False) -> Pack:
    """Validate one pack definition. Raises ValueError with a readable message."""
    if not isinstance(data, dict):
        raise ValueError("a pack must be a JSON object")
    pack_id = str(data.get("id") or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", pack_id):
        raise ValueError(f"invalid pack id {pack_id!r} (lowercase letters, digits and dashes)")
    name = str(data.get("name") or "").strip()
    if not name:
        raise ValueError(f"{pack_id}: missing name")
    category = str(data.get("category") or ("Custom" if custom else "")).strip()
    if not custom and category not in CATEGORIES:
        raise ValueError(f"{pack_id}: unknown category {category!r}")
    terms = _unique(data.get("terms"))
    if len(terms) < (3 if custom else 20):
        raise ValueError(f"{pack_id}: needs at least {3 if custom else 20} terms")
    tags = tuple(re.sub(r"[^\w]", "", t.lstrip("#")) for t in _unique(data.get("tags"))) or (pack_id.replace("-", ""),)
    return Pack(pack_id, name, category, str(data.get("genre") or "").strip(), _unique(data.get("aliases")),
                terms, tuple(t for t in tags if t), custom)


def load_packs(refresh: bool = False) -> Dict[str, Pack]:
    """All packs by id: built-in ones, then the user's (which can override a built-in id)."""
    global _packs
    if _packs is not None and not refresh:
        return _packs
    sources = [(path, False) for path in sorted(DATA_DIR.glob("*.json"))]
    try:
        sources += [(path, True) for path in sorted(user_packs_dir().glob("*.json"))]
    except OSError:
        pass
    packs: Dict[str, Pack] = {}
    errors: List[str] = []
    for path, custom in sources:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        items = data.get("packs", [data]) if isinstance(data, dict) else data
        for item in items if isinstance(items, list) else []:
            try:
                pack = parse_pack(item, custom)
            except (ValueError, TypeError, AttributeError) as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            packs[pack.id] = pack
    _packs = packs
    _errors[:] = errors
    return packs


def pack_errors() -> List[str]:
    load_packs()
    return list(_errors)


def all_categories() -> List[str]:
    extra = sorted({p.category for p in load_packs().values() if p.category not in CATEGORIES})
    return list(CATEGORIES) + extra


def get_pack(key: str) -> Optional[Pack]:
    packs = load_packs()
    wanted = str(key or "").strip()
    if wanted.lower() in packs:
        return packs[wanted.lower()]
    normalized = normalize(wanted)
    return next((p for p in packs.values() if normalize(p.name) == normalized), None) if normalized else None


def resolve_packs(keys: Iterable[str]) -> List[Pack]:
    result: List[Pack] = []
    for key in keys or []:
        pack = get_pack(key)
        if pack is not None and pack not in result:
            result.append(pack)
    return result


# Everyday words that appear in many packs: never used as evidence that a topic is being discussed.
_COMMON = {normalize(w) for w in (
    "question", "réponse", "avis", "sujet", "thème", "épisode", "saison", "live", "stream", "test", "prix", "note",
    "cours", "exercice", "méthode", "conseil", "advice", "opinion", "talk", "show", "public", "style", "trend",
    "tendance", "routine", "guide", "budget", "musique", "music", "video", "chat", "match", "round", "score", "team",
    "équipe", "player", "game", "partie", "level", "niveau", "boss", "carte", "loot", "build", "skin", "ranked",
    "clutch", "noob", "kill", "power", "force", "speed", "temps", "time", "drop", "rush", "support", "carry", "farm",
    "base", "raid", "camp", "hook", "push", "save", "stream", "clip", "collab", "challenge", "film", "série",
    "album", "single", "live", "scène", "public", "joueur", "champion", "final", "finale", "record", "coach",
    "session", "chapitre", "personnage", "histoire", "monde", "world", "ville", "city", "map", "zone", "combo",
)}


def _evidence(pack: Pack) -> Tuple[List[str], List[str]]:
    aliases = [normalize(a) for a in pack.aliases if normalize(a)]
    terms = [key for key in (normalize(t) for t in pack.vocabulary())
             if len(key) >= 4 and key not in _COMMON]
    return aliases, terms


def detect_packs(texts: Iterable[str], min_score: int = 3) -> List[Tuple[Pack, int]]:
    """Packs mentioned in the texts, best first.

    A name/alias counts 5 points (up to 3 times), each distinct term 1 point. Topics that are not games
    need 8 distinct terms before their terms count, so ordinary conversation does not trigger them.
    """
    blob = f" {' '.join(normalize(t) for t in texts if t)} "
    if not blob.strip():
        return []
    results = []
    for pack in load_packs().values():
        aliases, terms = _evidence(pack)
        score = sum(5 * min(3, blob.count(f" {alias} ")) for alias in aliases)
        distinct = sum(1 for term in set(terms) if f" {term} " in blob)
        if not pack.is_game and distinct < 8:
            distinct = 0
        score += distinct
        if score >= min_score:
            results.append((pack, score))
    results.sort(key=lambda item: (-item[1], item[0].name))
    return results


def auto_packs(title_texts: Sequence[str], sample_text: str = "") -> List[Pack]:
    """At most one game and one topic, recognised from the title/metadata first, then from speech."""
    title = detect_packs(title_texts, min_score=5)
    speech = detect_packs([sample_text], min_score=3) if sample_text and sample_text.strip() else []
    game = next((p for p, _ in title if p.is_game), None) or next((p for p, _ in speech if p.is_game), None)
    topic = (next((p for p, _ in title if not p.is_game), None)
             or next((p for p, score in speech if not p.is_game and score >= 8), None))
    return [p for p in (game, topic) if p is not None]


def detect_game(texts: Iterable[str], min_score: int = 3) -> Tuple[Optional[str], int]:
    """Name of the most likely game mentioned in the texts, with its score."""
    best = next(((p, score) for p, score in detect_packs(texts, min_score) if p.is_game), None)
    return (best[0].name, best[1]) if best else (None, 0)


def pack_vocabulary(keys: Iterable[str]) -> List[str]:
    return list(_unique(term for pack in resolve_packs(keys) for term in pack.vocabulary()))


def game_vocabulary(game: Optional[str]) -> List[str]:
    pack = get_pack(game) if game else None
    return pack.vocabulary() if pack else []


def prompt_terms(user_terms: Sequence[str], pack_terms: Sequence[str], sample_text: str = "",
                 budget: int = PROMPT_BUDGET) -> List[str]:
    """Terms given to the speech model: the user's own first, then pack terms heard in the speech sample,
    then the other pack terms, within the prompt budget."""
    heard = f" {normalize(sample_text)} "
    pack_terms = list(_unique(pack_terms))
    ordered = [t for t in pack_terms if f" {normalize(t)} " in heard] + \
              [t for t in pack_terms if f" {normalize(t)} " not in heard]
    result = list(_unique(user_terms))
    used = sum(len(t) + 2 for t in result)
    known = {normalize(t) for t in result}
    for term in ordered:
        key = normalize(term)
        if key in known:
            continue
        if used + len(term) + 2 > budget:
            continue
        result.append(term)
        known.add(key)
        used += len(term) + 2
    return result


def write_pack_template() -> Path:
    """Create the user's pack folder, with an example pack the first time. Returns the folder."""
    folder = user_packs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    if not any(folder.glob("*.json")):
        example = {
            "id": "my-community",
            "name": "My community",
            "category": "Custom",
            "genre": "Server, friends, running jokes",
            "aliases": ["name of my show", "name of my server"],
            "terms": ["nickname of a friend", "name of the server", "inside joke", "your catchphrase"],
            "tags": ["mycommunity"],
        }
        (folder / "example-my-community.json").write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n",
                                                          encoding="utf-8")
    return folder


# -- hype phrases & hashtags ---------------------------------------------------------------------------

# Reactions worth keeping, per language. English ones are also used for every language
# (streamers everywhere say "let's go" and "no way").
HYPE_PHRASES: Dict[str, List[str]] = {
    # Everyday words ("allez", "vas-y", "bro", "run") are left out on purpose: they are said all the time
    # and would outweigh the real reactions.
    "en": ["no way", "let's go", "oh my god", "oh my gosh", "what the hell", "holy shit", "holy moly", "insane",
           "unbelievable", "clip that", "clip it", "no no no", "yes yes yes", "are you kidding", "that's crazy",
           "that was crazy", "oh no no", "help me", "i'm dead", "we won", "we did it", "victory"],
    "fr": ["c'est pas possible", "incroyable", "trop fort", "oh non non", "c'est chaud", "j'y crois pas",
           "oh la la", "t'es sérieux", "non non non", "oui oui oui", "au secours", "je suis mort", "c'est ouf",
           "de ouf", "énorme", "magnifique", "n'importe quoi", "clippez", "clip ça", "on a gagné", "on l'a fait",
           "c'est la fin", "t'es fou", "sa mère", "la dinguerie", "pas possible"],
    "es": ["no puede ser", "vamos", "increíble", "qué locura", "madre mía", "dios mío", "no no no", "qué pasó",
           "clip"],
    "de": ["krass", "unglaublich", "los geht's", "oh mein gott", "was zum", "alter", "digga", "nein nein nein",
           "clip"],
    "it": ["non ci credo", "andiamo", "incredibile", "oddio", "pazzesco", "dai dai", "no no no", "clip"],
    "pt": ["não acredito", "vamos", "incrível", "meu deus", "que isso", "caramba", "não não não", "clip"],
}

LAUGH_RE = re.compile(
    r"^(?:(?:ha|ah|he|hi|hu|xa|ja|je|ka){2,}h?|k{3,}|w{3,}|mdr+|ptdr+|lol+|lmf?ao+|xd+|rires?|laughs?|laughter|"
    r"laughing|risas?|lacht)$"
)

_LANGUAGE_TAGS = {
    "fr": ["twitchfr", "streamerfr"], "en": ["twitch", "streamer"], "es": ["twitchespañol"],
    "de": ["twitchde"], "it": ["twitchitalia"], "pt": ["twitchbr"],
}


def hype_phrases(language: str) -> List[str]:
    language = (language or "en").lower()[:2]
    phrases = list(HYPE_PHRASES.get(language, []))
    for phrase in HYPE_PHRASES["en"]:
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def is_laughter(token: str) -> bool:
    return bool(LAUGH_RE.match(normalize(token).replace(" ", "")))


def hashtags(packs: Union[str, Sequence[str], None], content_type: str, language: str, limit: int = 8) -> List[str]:
    """Hashtags for a short: its packs (ids or names), the content style and the language."""
    keys = [packs] if isinstance(packs, str) else list(packs or [])
    tags: List[str] = ["shorts"]
    for pack in resolve_packs(keys):
        tags += list(pack.tags)
    tags.append({"gaming": "gaming", "talk": "podcast", "vlog": "vlog"}.get(content_type, "clips"))
    if content_type == "gaming":
        tags += _LANGUAGE_TAGS.get((language or "en")[:2], _LANGUAGE_TAGS["en"])
    tags += ["highlights", "fyp"]
    unique: List[str] = []
    for tag in tags:
        if tag and tag not in unique:
            unique.append(tag)
    return ["#" + tag for tag in unique[:limit]]
