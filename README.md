# Smart Airport Passenger Assistance Multimodal Chatbot (AI7016)

A multimodal (image / voice / text) assistant for a **fictional** airport,
Nordhaven International (NVH), answering passenger questions from a structured,
fully synthetic knowledge base. MSc set exercise; the design is fixed in
`Architecture Freeze v1.1` and evidenced in `Evidence Pack 02B2` (project docs).

Core stack (frozen): CLIP ViT-B/32 (zero-shot image labelling with
out-of-scope anchors) · Whisper-base (ASR) + airport text normaliser ·
MiniLM sentence embeddings for intent-by-similarity and semantic KB retrieval ·
deterministic retrieval cascade (exact identifier → alias → category-filtered
semantic → full semantic → answer/clarify/abstain/redirect) · template
responses grounded in KB records · Gradio Blocks UI. No generative model
anywhere; EasyOCR is an optional, flag-gated enhancement.

## Repository layout

| Path | Responsibility (one sentence) |
|---|---|
| `app/` | Gradio entrypoint; currently a hello-world used only as the Space deployment smoke test. |
| `configs/` | The single `Settings` dataclass every script reads; thresholds live here and nowhere else. |
| `data/kb/` | The synthetic knowledge base (`airport_kb.json`, 32 records). |
| `data/images/`, `data/audio/`, `data/multimodal/` | Dataset manifests (header CSVs now; rows added in Chats 03–04 per `docs/dataset_schemas.md`). |
| `data/text/` | Seed query set (`queries_seed.csv`, 43 rows) — the text half of every later evaluation. |
| `data/vocabulary.json` | Frozen controlled vocabulary shared by image labels, intents, entities, KB categories and routing. |
| `docs/` | Airport specification, dataset schemas, and project documents. |
| `evaluation/` | Metric modules (pure functions now; model outputs plug in from Chat 03). |
| `outputs/` | Everything generated (benchmarks, evaluation artefacts); git-ignored except `.gitkeep`. |
| `scripts/` | Runnable utilities: `smoke_test.py`, `benchmark_env.py`, `audit_foundation.py`. |
| `src/` | Pipeline code (Chat 03+); today only `kb.py` and `foundation_audit.py`. |
| `tests/` | Consistency gate (`python tests/test_kb_consistency.py` or pytest). |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + evaluation/dev extras (sklearn, matplotlib, pytest, faiss dev-check)
# EasyOCR only if the OCR enhancement is enabled:
# pip install -r requirements-ocr.txt
python scripts/smoke_test.py             # must print SMOKE TEST: PASS
python scripts/audit_foundation.py       # must print ALL CHECKS PASS
```

## Environment notes — what is verified and what is not

- **Verified (11 Sep 2026):** all three requirement sets resolve together with
  `uv pip compile` on **Linux x86_64 / Python 3.12** with the pins held
  (`torch==2.9.1` remains pinned even with `easyocr` added). Resolution is not
  installation; no model was executed in that check.
- **Not yet verified (label kept until measured):** installation and imports on
  **macOS arm64** (run `scripts/smoke_test.py` locally); actual CPU latency and
  memory for any model (run `scripts/benchmark_env.py` — loaders land in
  Chat 03); behaviour on the Hugging Face Space.
- Pin policy: latest **mature** lines, not latest majors (e.g. `transformers`
  4.57.x and `gradio` 5.50.0 rather than the 5.x/6.x majors released later),
  because the course reference notebooks and the model cards used here are
  from those lines. Revisit only if a concrete blocker appears.
- On Linux, the default PyPI `torch` wheel pulls CUDA support packages that a
  CPU-only Space never uses; if Space build time or disk becomes a problem,
  a Space-specific requirements variant using the CPU wheel index is the
  documented fallback (to be tested on the Space, not assumed).
- Models download at first use into the HF cache; on Spaces the disk is
  **non-persistent**, so set `HF_HOME` (and `EASYOCR_MODULE_PATH` if OCR is
  ever enabled) to project-local paths and expect re-downloads after rebuilds.

## Hugging Face Space status

Account checked 11 Sep 2026: Gradio Spaces are creatable on the project
account (Docker is Paid — consistent with the no-Docker decision). The Space
itself is **not created yet**; the first deployment will push `app/app.py`
(hello-world) to verify the build path and record cold-start behaviour before
any model code exists. Open sub-check: free hardware option CPU Basic vs
ZeroGPU-only.

## Provenance and privacy

Everything in `data/` is synthetic and authored for this project; no real
airport's data was copied, and `source: synthetic` is machine-checked by the
audit. Planned audio recordings use pseudonymous speaker codes and written
consent; the running system will not retain raw audio or images (see the
Evidence Pack's GDPR section).
