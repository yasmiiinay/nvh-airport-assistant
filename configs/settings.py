"""Single configuration surface for the whole project (Architecture Freeze v1.1).

Everything tunable lives here so the report can print one table of settings.
Threshold values are None until set on the dev split -- never hard-code them
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
    outputs_dir: Path = REPO_ROOT / "outputs"

    # --- frozen model identifiers (loaded lazily in Chat 03; never at import) ---
    clip_model_id: str = "openai/clip-vit-base-patch32"
    whisper_model_id: str = "openai/whisper-base"      # size choice empirical (v1.1 4.2)
    sentence_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- feature flags ---
    enable_ocr: bool = False            # EasyOCR enhancement, gated (v1.1 4.5)
    similarity_backend: str = "numpy"   # "faiss" allowed only in the dev-only equivalence check

    # --- decision thresholds: EMPIRICAL, set on the dev split in Chat 03+ ---
    tau_high: float | None = None       # answer threshold
    tau_low: float | None = None        # abstain threshold
    margin_delta: float | None = None   # clarify margin

    # --- UI wording (frozen, v1.1 4.9) ---
    score_label: str = "match score"
    score_bands: tuple = ("strong match", "uncertain - please confirm", "no reliable match")

    # --- audio quality gate bounds (initial values; validated in Chat 03) ---
    audio_min_seconds: float = 1.0
    audio_max_seconds: float = 60.0

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items()}


SETTINGS = Settings()

if __name__ == "__main__":
    for key, value in SETTINGS.as_dict().items():
        print(f"{key:24s} {value}")
