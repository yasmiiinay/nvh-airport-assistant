"""Speech metrics (RQ2). Definitions per Evidence Pack section A4.

WER is computed at two normalisation levels and BOTH are always reported:
  L1 -- generic normalisation only (raw ASR quality; the headline number)
  L2 -- L1 plus the airport-specific normaliser
L1 - L2 is the measured benefit of the domain normaliser. Number words are
normalised at BOTH levels (following the Whisper paper's own normaliser);
only airport-specific repairs (letter words, spacing, terminal forms) are L2.
"""
from __future__ import annotations


def wer_l1(references: list[str], hypotheses: list[str]) -> dict:
    """Generic-normalisation WER via jiwer. Requires jiwer (runtime dep)."""
    import jiwer  # imported here so the harness file parses without the env
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        # TODO(Chat 03): add generic number-word -> digit transform here (L1, both sides)
        jiwer.ReduceToListOfListOfWords(),
    ])
    out = jiwer.process_words(references, hypotheses,
                              reference_transform=transform, hypothesis_transform=transform)
    return {"wer": out.wer, "substitutions": out.substitutions,
            "deletions": out.deletions, "insertions": out.insertions,
            "n_utterances": len(references)}


def wer_l2(references: list[str], hypotheses: list[str]) -> dict:
    """L1 pipeline + airport normaliser applied to both sides.
    TODO(Chat 03): implement once src/normalizer.py exists."""
    raise NotImplementedError("Blocked on src/normalizer.py (Chat 03).")


def identifier_token_accuracy(reference_identifiers: list[list[str]],
                              hypothesis_texts: list[str]) -> dict:
    """Share of utterances whose EVERY reference identifier appears, in
    canonical form, in the processed hypothesis. Also returns the observed
    substitution list for the error catalogue.
    TODO(Chat 03): canonicalisation comes from src/normalizer.py."""
    raise NotImplementedError("Blocked on src/normalizer.py (Chat 03).")


def propagation_gap(retrieval_acc_from_reference: float,
                    retrieval_acc_from_asr: float, n: int) -> dict:
    """Audio-to-retrieval propagation gap (same query set on both sides)."""
    return {"gap": retrieval_acc_from_reference - retrieval_acc_from_asr,
            "retrieval_acc_reference": retrieval_acc_from_reference,
            "retrieval_acc_asr": retrieval_acc_from_asr, "n": n}


def write_artifacts(*args, **kwargs):
    raise NotImplementedError("TODO(Chat 03+): persist to outputs/evaluation/speech/")
