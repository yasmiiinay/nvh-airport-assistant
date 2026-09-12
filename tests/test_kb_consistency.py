"""Foundation consistency gate. Runs with pytest or as a plain script.

Fails the build if the KB, vocabulary or seed query set drift apart.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.settings import SETTINGS
from src.foundation_audit import run_all


def test_foundation_consistency():
    results = run_all(SETTINGS.kb_path, SETTINGS.vocabulary_path, SETTINGS.queries_seed_path)
    problems = [f"{check}: {p}" for check, plist in results.items() for p in plist]
    assert not problems, "\n".join(problems)


def test_pure_metrics_run_without_models():
    """The metric functions that need no model must work today."""
    from evaluation.vision_metrics import top_k_accuracy
    from evaluation.retrieval_metrics import stage_firing_counts, decision_rates
    from evaluation.uncertainty_metrics import risk_coverage_curve, band_table
    assert top_k_accuracy(["a", "b"], [["a", "c"], ["c", "d"]], k=1)["accuracy"] == 0.5
    assert stage_firing_counts(["exact_identifier", "exact_identifier"]) == {"exact_identifier": 2}
    assert decision_rates(["answer", "abstain"])["n"] == 2
    curve = risk_coverage_curve([0.9, 0.2], [True, False], [0.5])
    assert curve[0]["coverage"] == 0.5 and curve[0]["risk"] == 0.0
    bands = band_table([0.9, 0.5, 0.1], [True, False, False], tau_low=0.3, tau_high=0.7)
    assert bands["strong"]["n"] == 1 and bands["abstain"]["n"] == 1


if __name__ == "__main__":
    test_foundation_consistency()
    test_pure_metrics_run_without_models()
    print("tests: PASS")
