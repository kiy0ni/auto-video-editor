"""Vocabulary packs: catalog integrity, recognition, prompt selection and custom packs."""

import json
from collections import Counter

import pytest

from auto_video_editor import glossary
from auto_video_editor.cli import build_parser, settings_from_args
from auto_video_editor.glossary import (CATEGORIES, DATA_DIR, auto_packs, detect_packs, get_pack, hashtags,
                                        load_packs, pack_errors, parse_pack, prompt_terms, resolve_packs)


def test_catalog_is_large_and_complete():
    packs = load_packs(refresh=True)
    builtin = [p for p in packs.values() if not p.custom]
    assert pack_errors() == [] or all("example" in e for e in pack_errors())
    assert len(builtin) >= 100
    assert sum(p.is_game for p in builtin) >= 60
    assert {p.category for p in builtin} == set(CATEGORIES)
    for pack in builtin:
        assert len(pack.terms) >= 20, pack.id
        assert pack.aliases and pack.tags and pack.genre, pack.id
    raw_ids = [item["id"] for path in DATA_DIR.glob("*.json") for item in json.loads(path.read_text())["packs"]]
    duplicates = [pack_id for pack_id, count in Counter(raw_ids).items() if count > 1]
    assert not duplicates, duplicates


@pytest.mark.parametrize("title, expected", [
    ("SoImKiyo VOD - Minecraft (Mineral Contest_FK)", "minecraft"),
    ("Ranked Valorant road to Radiant", "valorant"),
    ("LE PODCAST #12 avec un invité", "podcast"),
    ("Recette facile : cuisine du dimanche", "cooking"),
    ("Séance musculation jambes", "fitness"),
    ("Wembanyama en NBA ce soir", "basketball"),
    ("GTA RP sur FiveM", "gta"),
    ("Lethal Company avec les potes", "lethal-company"),
    ("Réaction au nouvel album de Ninho | concert", "music-concerts"),
    ("Tuto FL Studio : faire une prod trap", "music-production"),
])
def test_recognition_from_titles(title, expected):
    assert expected in [p.id for p in auto_packs([title])]


def test_topics_need_real_evidence_from_speech():
    assert auto_packs([], "on a regardé le match hier, quel but franchement") == []
    commentary = ("penalty sifflé, hors-jeu signalé, carton rouge pour le défenseur, le gardien est battu, "
                  "une frappe dans la lucarne, un tacle, l'arbitre consulte la VAR, Mbappé marque")
    assert [p.id for p in auto_packs([], commentary)] == ["football"]
    gaming = "attention au creeper, on descend dans le nether chercher de la netherite"
    assert [p.id for p in auto_packs([], gaming)] == ["minecraft"]


def test_prompt_terms_priorities_and_budget():
    pool = get_pack("minecraft").vocabulary()
    terms = prompt_terms(["KiyOni"], pool, "on cherche de la netherite et une elytra")
    assert terms[0] == "KiyOni"
    assert {"netherite", "elytra"} <= set(terms[1:3])
    assert sum(len(t) + 2 for t in terms) <= glossary.PROMPT_BUDGET
    assert len(terms) < len(pool), "the prompt keeps only what fits"


def test_lookup_and_hashtags():
    assert get_pack("Minecraft").id == "minecraft" and get_pack("twitch-fr").category == "Streaming & culture"
    assert get_pack("nope") is None
    assert [p.id for p in resolve_packs(["minecraft", "Minecraft", "nope", "twitch-fr"])] == ["minecraft", "twitch-fr"]
    tags = hashtags(["minecraft", "twitch-fr"], "gaming", "fr")
    assert "#minecraft" in tags and "#twitchfr" in tags and len(tags) == len(set(tags))
    assert "#minecraft" in hashtags("Minecraft", "gaming", "fr"), "a single name still works"


def test_detect_packs_scores_games_before_topics():
    results = detect_packs(["Minecraft hardcore et just chatting"], min_score=5)
    assert [p.id for p, _ in results][:2] == ["minecraft", "twitch-fr"]


def test_custom_packs(tmp_path, monkeypatch):
    monkeypatch.setattr(glossary, "user_packs_dir", lambda: tmp_path)
    (tmp_path / "server.json").write_text(json.dumps({
        "id": "kiyo-server", "name": "Kiyo server", "category": "Custom", "aliases": ["kiyo smp"],
        "terms": ["Ronflex", "S-M Darki", "KiyOni"], "tags": ["#kiyosmp"]}))
    (tmp_path / "broken.json").write_text("{not json")
    try:
        packs = load_packs(refresh=True)
        assert packs["kiyo-server"].custom and packs["kiyo-server"].tags == ("kiyosmp",)
        assert any("broken.json" in e for e in pack_errors())
        assert [p.id for p in auto_packs(["Kiyo SMP episode 4"])] == ["kiyo-server"] or \
            "kiyo-server" in [p.id for p, _ in detect_packs(["Kiyo SMP episode 4"], min_score=5)]
        folder = glossary.write_pack_template()
        assert folder == tmp_path and not (tmp_path / "example-my-community.json").exists(), "only for empty folders"
    finally:
        monkeypatch.undo()
        load_packs(refresh=True)


def test_invalid_packs_are_rejected():
    with pytest.raises(ValueError):
        parse_pack({"id": "Bad Id", "name": "x", "category": "Games", "terms": ["a"] * 30})
    with pytest.raises(ValueError):
        parse_pack({"id": "too-small", "name": "x", "category": "Games", "terms": ["a", "b"]})
    with pytest.raises(ValueError):
        parse_pack({"id": "wrong-cat", "name": "x", "category": "Nope", "terms": [str(i) for i in range(30)]})


def test_cli_packs():
    s = settings_from_args(build_parser().parse_args(["a.mp4", "--packs", "minecraft,Twitch-FR"]))
    assert s.vocabulary_packs == ["minecraft", "twitch-fr"]
    with pytest.raises(ValueError):
        settings_from_args(build_parser().parse_args(["a.mp4", "--packs", "minecraft,unknown-game"]))
