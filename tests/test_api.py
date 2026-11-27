"""API contract tests for the demo server."""

import base64

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from samadhvani.app.server import app  # noqa: E402

client = TestClient(app)

VERSE = "हाउ अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म्"


def test_index_serves_the_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "Samadhvani" in r.text
    assert "/static/app.js" in r.text


def test_static_assets_exist():
    for path in ("/static/app.js", "/static/style.css"):
        r = client.get(path)
        assert r.status_code == 200, path


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert "formant" in d["engines"]
    assert "auto" in d["modes"]


def test_samples_include_metadata():
    r = client.get("/api/samples")
    assert r.status_code == 200
    d = r.json()
    assert len(d["samples"]) >= 4
    for s in d["samples"]:
        assert s["sanskrit"] and s["note"] and s["source"]
    # the seven sāma svaras must be described for the UI
    assert len(d["svaras"]) == 7
    assert {v["key"] for v in d["schools"]} >= {"kauthuma", "ranayaniya", "jaiminiya"}


def test_chant_returns_audio_and_structure():
    r = client.post("/api/chant", json={"text": VERSE, "laya": 0.34})
    assert r.status_code == 200
    d = r.json()
    wav = base64.b64decode(d["audio_b64"])
    assert wav[:4] == b"RIFF"
    assert d["mode"] in ("gana", "samhita", "style")
    assert len(d["notes"]) == len(d["syllables"]) == len(d["svaras"])
    assert d["caveats"]
    assert "evaluation" in d


def test_chant_evaluation_can_be_skipped():
    r = client.post("/api/chant", json={"text": VERSE, "evaluate": False, "laya": 0.34})
    assert r.status_code == 200
    assert "evaluation" not in r.json()


def test_chant_wav_endpoint_streams_audio():
    r = client.post("/api/chant.wav", json={"text": VERSE, "laya": 0.34})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"


def test_chant_rejects_bad_school():
    r = client.post("/api/chant", json={"text": VERSE, "school": "nope"})
    assert r.status_code == 400


def test_chant_rejects_bad_model():
    r = client.post("/api/chant", json={"text": VERSE, "model": "nope"})
    assert r.status_code == 400


def test_chant_rejects_empty_text():
    r = client.post("/api/chant", json={"text": ""})
    assert r.status_code == 422


def test_chant_rejects_non_sanskrit():
    r = client.post("/api/chant", json={"text": "hello world this is english"})
    assert r.status_code == 400
    assert "sanskrit" in r.json()["detail"].lower()


def test_neural_without_checkpoint_is_a_clear_400():
    r = client.post("/api/chant", json={"text": VERSE, "engine": "neural"})
    # if a checkpoint does happen to exist, it must instead render
    assert r.status_code in (200, 400)
    if r.status_code == 400:
        assert "checkpoint" in r.json()["detail"].lower()


def test_laya_is_bounded():
    r = client.post("/api/chant", json={"text": VERSE, "laya": 5.0})
    assert r.status_code == 422


def test_analyze_returns_text_structure_only():
    r = client.post("/api/analyze", json={"text": VERSE})
    assert r.status_code == 200
    d = r.json()
    assert d["syllables"] and d["svaras"]
    assert "notes" not in d        # analysis only: no audio, no per-note schedule
    assert "vikaras" in d


def test_data_endpoint_is_honest():
    r = client.get("/api/data")
    assert r.status_code == 200
    d = r.json()
    keys = {s["key"]: s for s in d["sources"]}
    assert keys["youtube_samagana"]["train"] is False
    assert keys["youtube_samagana"]["licence"] == "unclear"
    assert "blocked" in d["report"]
