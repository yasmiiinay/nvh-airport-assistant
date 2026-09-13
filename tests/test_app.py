"""Interface boundaries: the callbacks return the shapes the layout expects,
never raise, and the page builds. Model-backed cases skip without weights."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
gr = pytest.importorskip("gradio")
import app as ui
from src import event_log


def test_page_builds():
    assert isinstance(ui.build_ui(), gr.Blocks)


def test_empty_input_is_a_message_not_a_traceback():
    response, band, source, transcript, evidence, session = ui.answer("", None, None, {})
    assert "type a question" in response and band == "Not scored" and session["turn"] == 1


def test_assistance_needs_a_previous_turn():
    assert "Ask a question first" in ui.request_assistance("help", {})


@pytest.fixture(scope="module")
def models_ready():
    try:
        ui.get_context()
    except Exception as exc:
        pytest.skip(f"models not available here: {type(exc).__name__}")


def test_text_turn_and_ticket(models_ready, tmp_path, monkeypatch):
    monkeypatch.setattr(event_log, "EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(event_log, "TICKETS_PATH", tmp_path / "tickets.jsonl")
    response, band, source, transcript, evidence, session = ui.answer("Where is gate B12?", None, None, {})
    assert response.startswith("Pier B Gates") and band.startswith("Strong match") and source == "your words"
    assert "Matched record" in evidence and transcript == ""
    assert (tmp_path / "events.jsonl").exists()
    note = ui.request_assistance("", session)
    assert "NVH-" in note and "Departures information desk" in note


def test_unreadable_image_is_handled(models_ready, tmp_path, monkeypatch):
    monkeypatch.setattr(event_log, "EVENTS_PATH", tmp_path / "events.jsonl")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    response, band, source, *_ = ui.answer("", str(bad), None, {})
    assert "could not read" in response and band == "No reliable match"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
