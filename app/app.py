"""Nordhaven International passenger assistant: the Gradio interface.

One callback, `answer`, takes whatever the passenger supplied (typed text, a
photo, a voice clip), routes it through src.router and renders the
outcome. Models are loaded on the first question, not at import, so the
Space starts quickly and a modality that is never used is never loaded.

Run locally:  python app/app.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.settings import SETTINGS
from src.entities import load_gazetteers
from src.event_log import contact_route, event_from_outcome, log_event, new_session_id, open_ticket
from src.responses import render_outcome
from src.retrieval import RetrievalResult, build_text_index
from src.router import Outcome, build_context, route

try:
    import spaces   # present on ZeroGPU Spaces only; the probe below satisfies its startup check
    _gpu_decorator = spaces.GPU
except ImportError:
    def _gpu_decorator(fn):
        return fn


@_gpu_decorator
def zerogpu_probe() -> str:
    return "ok"


AIRPORT = "Nordhaven International (NVH)"
BAND_LABELS = {"strong match": "Strong match", "uncertain": "Uncertain, please confirm",
               "no reliable match": "No reliable match"}
ROUTE_LABELS = {"text_only": "your words", "voice_only": "your voice", "image_only": "your photo",
                "text_leads": "your words, photo checked", "voice_leads": "your voice, photo checked",
                "image_leads": "your photo, words used to narrow down", "none": "nothing usable"}
DECISION_LABELS = {"answer": "Answer", "clarify": "Question back to you", "abstain": "No answer",
                   "redirect": "Referred to the official source", "conflict": "Your inputs disagree"}

_context = None


def get_context():
    global _context
    if _context is None:
        gaz = load_gazetteers(SETTINGS.kb_path, SETTINGS.vocabulary_path)
        _context = build_context(gaz, build_text_index(gaz, SETTINGS.intent_exemplars_path))
    return _context


# ---------------------------------------------------------------------------
# what the evidence panel shows
# ---------------------------------------------------------------------------

def evidence_markdown(outcome: Outcome, gaz) -> str:
    lines = [f"**Decision:** {DECISION_LABELS.get(outcome.decision, outcome.decision)}",
             f"**Evidence used:** {ROUTE_LABELS.get(outcome.route, outcome.route)}",
             f"**Why:** {outcome.reason or 'see the answer'}"]
    text = outcome.text
    if text is not None:
        lines.append(f"**Understood as:** `{text.normalized}`")
        if text.entities:
            lines.append("**Entities:** " + ", ".join(f"{k} = {v}" for k, v in text.entities.items()))
        if text.intent:
            lines.append(f"**Intent:** {text.intent} (nearest example score {text.intent_score:.2f}: "
                         f"\"{text.intent_exemplar}\")")
        lines.append(f"**Retrieval stage:** {text.stage or 'none'}")
        if text.ranked:
            lines.append("**Top records by similarity:** " +
                         ", ".join(f"{gaz.records[rid]['name']} {score:.2f}" for rid, score in text.ranked))
    vision = outcome.vision
    if vision is not None:
        lines.append("**Photo, top categories:** " +
                     ", ".join(f"{c.replace('_', ' ')} {s:.2f}" for c, s in vision.category_ranking) +
                     f" (margin {vision.category_margin:.3f})")
        lines.append(f"**Photo, closest non-sign anchor:** {vision.best_anchor[0]} {vision.best_anchor[1]:.2f}")
        if vision.check.flags:
            lines.append("**Photo quality flags:** " + ", ".join(vision.check.flags))
    speech = outcome.speech
    if speech is not None:
        lines.append(f"**Audio:** {speech.check.seconds:.1f} s, {speech.check.rms_dbfs:.0f} dBFS, "
                     f"{'accepted' if speech.check.ok else 'rejected: ' + str(speech.check.problem)}")
    if outcome.score is not None:
        lines.append(f"**Match score:** {outcome.score:.2f}. This is a cosine similarity between your input and "
                     "the record or category text. Values for this system sit between about 0.25 and 0.60, "
                     "so 0.37 can be a strong match; the band comes from the score together with the gap "
                     "to the runner-up, not from the number alone.")
    if outcome.matched_record_id:
        lines.append(f"**Matched record:** `{outcome.matched_record_id}`")
    if outcome.candidates:
        lines.append("**Candidates:** " + ", ".join(gaz.records[r]["name"] for r in outcome.candidates if r in gaz.records))
    if outcome.conflict:
        d = outcome.conflict_detail
        lines.append(f"**Conflict:** words point to {d.get('text_record') or ', '.join(d.get('text_categories', []))}; "
                     f"photo points to {d.get('image_category')}; resolution: {d.get('resolution')}")
    if outcome.flags:
        lines.append("**Flags:** " + ", ".join(outcome.flags))
    lines.append("Scores are cosine similarities: a measure of closeness, not a probability that the answer is right.")
    return "\n\n".join(lines)


def band_text(outcome: Outcome) -> str:
    if outcome.decision == "redirect":
        return "Referred to the official source"
    if outcome.decision == "conflict":
        return "Inputs disagree, please choose"
    return BAND_LABELS.get(outcome.band or "", "Not scored")


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

def answer(text, image_path, audio_path, session):
    """The one callback. Returns: response, band, evidence source, transcript,
    evidence markdown, updated session state."""
    session = dict(session or {})
    session.setdefault("session_id", new_session_id())
    session["turn"] = session.get("turn", 0) + 1
    if not (text and text.strip()) and not image_path and not audio_path:
        return ("Please type a question, add a photo of a sign, or record your question.",
                "Not scored", "nothing yet", "", "", session)
    start = time.perf_counter()
    try:
        ctx = get_context()
        outcome = route(text, image_path, audio_path, ctx)
        response = render_outcome(outcome, ctx.gaz)
    except Exception as exc:                       # the interface must never show a traceback
        traceback.print_exc()
        log_event({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "session_id": session["session_id"],
                   "turn_index": session["turn"], "error": f"{type(exc).__name__}: {exc}"[:200]})
        return ("Something went wrong while reading your input. Please try again, or type your question.",
                "Not scored", "error", "", "", session)
    latency = time.perf_counter() - start
    log_event(event_from_outcome(outcome, session["session_id"], session["turn"], latency))
    session["last"] = {"decision": outcome.decision, "record_id": outcome.matched_record_id,
                       "candidates": list(outcome.candidates), "conflict": outcome.conflict,
                       "terminal": outcome.text.entities.get("terminal") if outcome.text is not None else None}
    transcript = ""
    if outcome.speech is not None:
        transcript = outcome.speech.transcript_raw or f"(clip rejected: {outcome.speech.check.problem})"
    return (response.replace("\n", "\n\n"), band_text(outcome), ROUTE_LABELS.get(outcome.route, outcome.route),
            transcript, evidence_markdown(outcome, ctx.gaz), session)


def request_assistance(note, session):
    """Write an escalation record and show the reference and the contact
    route the knowledge base names. This is a record, not a live chat."""
    session = dict(session or {})
    last = session.get("last")
    if not last:
        return "Ask a question first, then request assistance if the answer did not help."
    ctx = get_context()
    stub = Outcome(route="none", decision=last["decision"], matched_record_id=last["record_id"],
                   candidates=last["candidates"], conflict=last["conflict"])
    if last.get("terminal"):
        stub.text = RetrievalResult(query="", normalized="", entities={"terminal": last["terminal"]}, resolved=True)
    ticket = open_ticket(stub, ctx.gaz, session.get("session_id", "unknown"), note or "")
    return (f"Your request has been recorded with reference **{ticket['reference']}**. "
            f"Please quote it at: {ticket['contact_route']}. "
            "This assistant cannot connect you to a person; the airport's own staff and displays "
            "are the official source for anything time-critical.")


def clear():
    return "", None, None, "", "Not scored", "", "", ""


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

CSS = """
#header { padding: 0.6rem 0 0.2rem 0; }
#header h1 { margin: 0; font-size: 1.5rem; letter-spacing: 0.02em; }
#header p { margin: 0.2rem 0 0 0; opacity: 0.8; }
#notice { border-left: 4px solid #7a7a7a; padding: 0.4rem 0.8rem; margin: 0.4rem 0 0.8rem 0; }
#response { min-height: 6rem; line-height: 1.5; }
.status-box textarea { font-weight: 600; }
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    theme = gr.themes.Base(primary_hue="slate", neutral_hue="gray", font=[gr.themes.GoogleFont("Inter"), "sans-serif"])
    with gr.Blocks(theme=theme, css=CSS, title=f"{AIRPORT} passenger assistant") as demo:
        session = gr.State({})
        with gr.Column(elem_id="header"):
            gr.Markdown(f"# {AIRPORT} passenger assistant\n"
                        "Ask about gates, check-in, baggage, security, toilets, transport, lounges, "
                        "first aid and assistance. Type, add a photo of a sign, or speak.")
        gr.Markdown("**Demonstration system for a fictional airport.** All information is synthetic. "
                    "Live flight, gate and delay information is never given here; the airport's displays "
                    "and staff are the official source.", elem_id="notice")
        with gr.Row():
            with gr.Column(scale=5):
                text_in = gr.Textbox(label="Your question", placeholder="For example: Where is gate B12?",
                                     lines=2, elem_id="question")
                # image_mode=None keeps the file as uploaded: the default RGB conversion
                # turns a transparent pictogram into a black square before it reaches us
                image_in = gr.Image(label="Photo of a sign (optional)", type="filepath", sources=["upload"],
                                    image_mode=None, height=220, elem_id="photo")
                audio_in = gr.Audio(label="Or ask by voice (optional)", type="filepath",
                                    sources=["microphone", "upload"], elem_id="voice")
                with gr.Row():
                    ask = gr.Button("Ask", variant="primary", elem_id="ask")
                    reset = gr.Button("Clear", elem_id="clear")
            with gr.Column(scale=6):
                response = gr.Markdown(label="Answer", value="", elem_id="response")
                with gr.Row():
                    band = gr.Textbox(label="Match band", value="Not scored", interactive=False, lines=2,
                                      info="How well your input matched: strong, uncertain, or no reliable match. "
                                           "It is not a probability. The number behind it is under Evidence and details.",
                                      elem_classes=["status-box"], elem_id="band")
                    source = gr.Textbox(label="Evidence used", value="", interactive=False, elem_id="source")
                transcript = gr.Textbox(label="What I heard (voice input)", value="", interactive=False,
                                        elem_id="transcript")
                with gr.Accordion("Evidence and details", open=False, elem_id="evidence"):
                    evidence = gr.Markdown(value="")
                with gr.Accordion("Request assistance", open=False, elem_id="assistance"):
                    gr.Markdown("If the answer did not help, leave a short note. A reference number is "
                                "recorded for the airport's contact route named in the answer.")
                    note = gr.Textbox(label="Your note (optional)", lines=2)
                    request = gr.Button("Record assistance request")
                    ticket_out = gr.Markdown(value="")
        gr.Markdown("The match band says how well your input matched the airport information: strong match, "
                    "uncertain (please confirm), or no reliable match. It is based on similarity, not on a "
                    "probability that the answer is correct; the underlying numbers are shown under Evidence and details.")

        outputs = [response, band, source, transcript, evidence, session]
        ask.click(answer, inputs=[text_in, image_in, audio_in, session], outputs=outputs)
        text_in.submit(answer, inputs=[text_in, image_in, audio_in, session], outputs=outputs)
        reset.click(clear, inputs=None, outputs=[text_in, image_in, audio_in, response, band, source, transcript, evidence])
        request.click(request_assistance, inputs=[note, session], outputs=[ticket_out])
    return demo


if __name__ == "__main__":
    build_ui().launch()
