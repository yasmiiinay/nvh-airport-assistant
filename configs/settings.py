"""Single configuration surface for the whole project (Architecture Freeze v1.1).

Everything tunable lives here so the report can print one table of settings.
Threshold values are None until set on the dev split; never hard-code them
elsewhere, and never tune them on the held-out split.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    # --- reproducibility ---
    seed: int = 42

    # --- data paths ---
    kb_path: Path = REPO_ROOT / "data" / "kb" / "airport_kb.json"
    vocabulary_path: Path = REPO_ROOT / "data" / "vocabulary.json"
    queries_seed_path: Path = REPO_ROOT / "data" / "text" / "queries_seed.csv"
    queries_heldout_path: Path = REPO_ROOT / "data" / "text" / "queries_heldout.csv"
    intent_exemplars_path: Path = REPO_ROOT / "data" / "text" / "intent_exemplars.csv"
    outputs_dir: Path = REPO_ROOT / "outputs"
    models_dir: Path = REPO_ROOT / "models"   # local model copies, git-ignored; hub id used if absent

    # --- frozen model identifiers (loaded lazily in the pipeline phase; never at import) ---
    clip_model_id: str = "openai/clip-vit-base-patch32"
    whisper_model_id: str = "openai/whisper-base"      # size choice empirical (v1.1 4.2)
    sentence_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- feature flags ---
    enable_ocr: bool = False            # EasyOCR enhancement, gated (v1.1 4.5)
    similarity_backend: str = "numpy"   # "faiss" allowed only in the dev-only equivalence check

    # --- decision thresholds: EMPIRICAL, set on the dev split during pipeline development ---
    # Chosen on the 43-query dev split by the coarse grid in
    # scripts/run_text_pipeline_seed.py; see docs/checkpoints/03_2_text_pipeline.md.
    tau_high: float | None = 0.50       # answer threshold on cosine similarity
    tau_low: float | None = 0.25        # abstain threshold on cosine similarity
    margin_delta: float | None = 0.10   # clarify when top1 - top2 is below this
    tau_intent: float | None = 0.30     # below this nearest-exemplar score the intent is "none"
    # vision bands: empirical, unset until a labelled development image split exists
    vision_tau_high: float | None = None
    vision_tau_low: float | None = None
    vision_margin_delta: float | None = None

    # --- UI wording (frozen, v1.1 4.9) ---
    score_label: str = "match score"
    score_bands: tuple = ("strong match", "uncertain - please confirm", "no reliable match")

    # --- audio quality gate bounds (initial values; validated in the pipeline phase) ---
    audio_min_seconds: float = 1.0
    audio_max_seconds: float = 60.0
    audio_min_rms_dbfs: float = -45.0   # quieter than this is treated as silence
    audio_sample_rate: int = 16000      # what Whisper expects
    vision_prompts_path: Path = REPO_ROOT / "data" / "vision_prompts.json"

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items()}

    def vision_thresholds(self) -> dict | None:
        """None until the vision bands have been set on development images."""
        values = {"vision_tau_high": self.vision_tau_high, "vision_tau_low": self.vision_tau_low,
                  "vision_margin_delta": self.vision_margin_delta}
        return None if any(v is None for v in values.values()) else values

    def thresholds(self) -> dict:
        values = {"tau_intent": self.tau_intent, "tau_high": self.tau_high,
                  "tau_low": self.tau_low, "margin_delta": self.margin_delta}
        missing = [k for k, v in values.items() if v is None]
        if missing:
            raise ValueError(f"thresholds not set on the dev split yet: {missing}")
        return values


SETTINGS = Settings()

if __name__ == "__main__":
    for key, value in SETTINGS.as_dict().items():
        print(f"{key:24s} {value}")
