import json

from crag.refine import decompose
from crag.store import Doc


def test_correct_uses_internal_only(make):
    crag, _, _, web = make({"reset": 9})
    rec = crag.run("How do I factory reset my Nimbus Pod X?")
    assert rec["action"] == "correct"
    assert web.queries == []
    assert all(s["kind"] == "internal" for s in rec["sources"]) and rec["sources"]


def test_incorrect_discards_internal_and_uses_web(make):
    crag, _, _, web = make({})
    rec = crag.run("What is the capital of Australia?")
    assert rec["action"] == "incorrect"
    assert web.queries == ["search terms"]
    assert [s["kind"] for s in rec["sources"]] == ["web"]


def test_ambiguous_mixes_both(make):
    crag, _, _, _ = make({"Thread": 5})
    rec = crag.run("Is the Hub a Thread border router and what is Thread?")
    assert rec["action"] == "ambiguous"
    assert {s["kind"] for s in rec["sources"]} == {"internal", "web"}


def test_every_run_is_logged(make, tmp_path):
    crag, plain, _, _ = make({"reset": 9})
    a = crag.run("factory reset Pod X")
    b = plain.run("factory reset Pod X")
    lines = [json.loads(l) for l in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert [l["run_id"] for l in lines] == [a["run_id"], b["run_id"]]
    assert lines[0]["retrieved"][0]["grade"] == 9


def test_decompose_splits_sentences():
    assert decompose(Doc(id="x", text="One fact. Two facts! Three?", source="s")) == ["One fact.", "Two facts!", "Three?"]
