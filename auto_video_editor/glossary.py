"""Built-in knowledge used by the automatic mode.

- Popular games: recognised from the file name, the video metadata or a short speech sample, they
  auto-complete the transcription vocabulary ("creeper", "nether"...) and the hashtags.
- Hype phrases per language, and laughter detection.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

GAMES: Dict[str, dict] = {
    "Minecraft": dict(
        aliases=["minecraft"],
        terms=["creeper", "enderman", "nether", "netherite", "redstone", "villageois", "villager", "wither",
               "ender dragon", "elytra", "biome", "spawn", "mob", "farm", "craft", "pioche", "diamant", "diamond",
               "bastion", "totem", "TNT", "hardcore", "speedrun", "seed", "stuff", "cave", "zombie", "squelette"],
        tags=["minecraft"]),
    "Fortnite": dict(
        aliases=["fortnite"],
        terms=["victory royale", "top 1", "build", "edit", "storm", "tempête", "loot", "pompe", "shotgun", "sniper",
               "battle pass", "reboot", "coffre", "bouclier", "shield", "zone", "skin", "box fight"],
        tags=["fortnite"]),
    "Valorant": dict(
        aliases=["valorant"],
        terms=["spike", "defuse", "clutch", "ace", "jett", "reyna", "sage", "phoenix", "sova", "killjoy", "cypher",
               "omen", "viper", "vandal", "phantom", "operator", "eco", "retake", "radiant", "immortal"],
        tags=["valorant"]),
    "League of Legends": dict(
        aliases=["league of legends", "leagueoflegends"],
        terms=["baron", "nashor", "drake", "jungle", "jungler", "midlane", "toplane", "botlane", "gank",
               "pentakill", "flash", "ward", "nexus", "inhib", "farm", "ult", "teamfight"],
        tags=["leagueoflegends", "lol"]),
    "Counter-Strike": dict(
        aliases=["counter strike", "cs2", "csgo", "cs go"],
        terms=["awp", "ak47", "bomb", "defuse", "clutch", "ace", "eco", "force buy", "smoke", "molotov",
               "headshot", "dust2", "mirage", "inferno", "ancient", "nuke", "deagle"],
        tags=["cs2", "counterstrike"]),
    "Apex Legends": dict(
        aliases=["apex legends"],
        terms=["wraith", "pathfinder", "bloodhound", "lifeline", "octane", "horizon", "knock", "thirst", "kraber",
               "peacekeeper", "jumpmaster", "champion squad"],
        tags=["apexlegends"]),
    "GTA": dict(
        aliases=["gta", "gta rp", "gtarp", "grand theft auto"],
        terms=["los santos", "braquage", "heist", "wanted", "roleplay", "rp", "flics", "cops", "poursuite",
               "gang", "fourrière", "serveur"],
        tags=["gta", "gtarp"]),
    "Call of Duty": dict(
        aliases=["warzone", "call of duty", "black ops", "modern warfare"],
        terms=["gulag", "loadout", "killstreak", "uav", "prestige", "camo", "redeploy", "plates", "nuke"],
        tags=["callofduty", "warzone"]),
    "Rocket League": dict(
        aliases=["rocket league"],
        terms=["aerial", "flip reset", "ceiling shot", "kickoff", "demo", "boost", "overtime", "grand champ",
               "freestyle", "double touch"],
        tags=["rocketleague"]),
    "Among Us": dict(
        aliases=["among us"],
        terms=["impostor", "imposteur", "sus", "vent", "emergency meeting", "crewmate", "tasks", "eject"],
        tags=["amongus"]),
    "Elden Ring": dict(
        aliases=["elden ring", "dark souls", "sekiro", "bloodborne", "nightreign"],
        terms=["parry", "estus", "runes", "malenia", "margit", "radahn", "grace", "bonfire", "souls", "roll",
               "boss fight", "dodge"],
        tags=["eldenring", "soulslike"]),
    "Overwatch": dict(
        aliases=["overwatch"],
        terms=["payload", "genji", "mercy", "reinhardt", "widowmaker", "ana", "kiriko", "tank", "dps",
               "team fight", "ult"],
        tags=["overwatch2"]),
    "Rainbow Six Siege": dict(
        aliases=["rainbow six", "r6 siege", "rainbow 6"],
        terms=["drone", "breach", "reinforce", "thermite", "jager", "vigil", "ash", "clutch", "defuser"],
        tags=["r6", "rainbowsixsiege"]),
    "Lethal Company": dict(
        aliases=["lethal company"],
        terms=["quota", "scrap", "bracken", "coil head", "jester", "thumper", "the company", "moon"],
        tags=["lethalcompany"]),
    "Dead by Daylight": dict(
        aliases=["dead by daylight", "dbd"],
        terms=["killer", "survivor", "generator", "gen", "hook", "crochet", "looping", "pallet", "totem", "mori"],
        tags=["deadbydaylight"]),
    "Pokémon": dict(
        aliases=["pokemon", "pokémon"],
        terms=["shiny", "pokéball", "pokeball", "dresseur", "trainer", "légendaire", "legendary", "évolution",
               "nuzlocke", "arène", "gym"],
        tags=["pokemon"]),
    "Roblox": dict(
        aliases=["roblox"],
        terms=["obby", "robux", "tycoon", "noob", "brookhaven"],
        tags=["roblox"]),
    "Marvel Rivals": dict(
        aliases=["marvel rivals"],
        terms=["team up", "ult", "spider-man", "iron man", "venom", "loki", "hela", "strategist", "vanguard"],
        tags=["marvelrivals"]),
    "Genshin Impact": dict(
        aliases=["genshin", "honkai", "star rail"],
        terms=["primogems", "wish", "pity", "banner", "resin", "artifact", "constellation", "gacha"],
        tags=["genshinimpact"]),
    "Chess": dict(
        aliases=["chess", "échecs", "echecs"],
        terms=["checkmate", "échec et mat", "gambit", "blunder", "elo", "roque", "castle", "en passant",
               "sacrifice", "bullet", "blitz"],
        tags=["chess", "echecs"]),
}

# Reactions worth keeping, per language. English ones are also used for every language
# (streamers everywhere say "let's go" and "no way").
HYPE_PHRASES: Dict[str, List[str]] = {
    "en": ["no way", "let's go", "oh my god", "oh my gosh", "what the", "holy", "insane", "crazy", "unbelievable",
           "clip that", "clip it", "wow", "no no no", "yes yes yes", "are you kidding", "that's crazy", "oh no",
           "dude", "bro", "gg", "run", "help me"],
    "fr": ["c'est pas possible", "incroyable", "trop fort", "mais non", "oh non", "allez", "vas-y", "c'est chaud",
           "j'y crois pas", "oh la la", "t'es sérieux", "non non non", "oui oui oui", "au secours", "je suis mort",
           "c'est ouf", "de ouf", "énorme", "magnifique", "ah non", "mais quoi", "qu'est-ce que", "clip"],
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


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def detect_game(texts: Iterable[str], min_score: int = 3) -> Tuple[Optional[str], int]:
    """Most likely game mentioned in the texts, with its score.

    A game name counts 5 points, each distinct term of its vocabulary 1 point.
    """
    blob = f" {' '.join(normalize(t) for t in texts if t)} "
    if not blob.strip():
        return None, 0
    best, best_score = None, 0
    for name, game in GAMES.items():
        score = 0
        for alias in game["aliases"]:
            key = f" {normalize(alias)} "
            if key.strip() and key in blob:
                score += 5 * min(3, blob.count(key))
        score += sum(1 for term in game["terms"] if len(normalize(term)) >= 4 and f" {normalize(term)} " in blob)
        if score > best_score:
            best, best_score = name, score
    return (best, best_score) if best_score >= min_score else (None, best_score)


def game_vocabulary(game: Optional[str]) -> List[str]:
    if not game or game not in GAMES:
        return []
    return [game] + list(GAMES[game]["terms"])


def hype_phrases(language: str) -> List[str]:
    language = (language or "en").lower()[:2]
    phrases = list(HYPE_PHRASES.get(language, []))
    for phrase in HYPE_PHRASES["en"]:
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases


def is_laughter(token: str) -> bool:
    return bool(LAUGH_RE.match(normalize(token).replace(" ", "")))


def hashtags(game: Optional[str], content_type: str, language: str, limit: int = 8) -> List[str]:
    tags: List[str] = ["shorts"]
    if game and game in GAMES:
        tags += GAMES[game]["tags"]
    tags.append({"gaming": "gaming", "talk": "podcast", "vlog": "vlog"}.get(content_type, "clips"))
    if content_type == "gaming":
        tags += _LANGUAGE_TAGS.get((language or "en")[:2], _LANGUAGE_TAGS["en"])
    tags += ["highlights", "fyp"]
    unique = []
    for tag in tags:
        if tag not in unique:
            unique.append(tag)
    return ["#" + tag for tag in unique[:limit]]
