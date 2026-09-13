"""KB retrieval metrics, stratified by query type; stage-firing evidence table."""
from __future__ import annotations
from collections import Counter, defaultdict

STAGES = ["exact_identifier", "alias_lookup", "category_filter_semantic",
          "semantic_full_kb", "no_retrieval"]


def retrieval_accuracy_by_type(query_types: list[str], gold_ids: list[str],
                               predicted_ids: list[str]) -> dict:
    per_type = defaultdict(lambda: {"n": 0, "correct": 0})
    for qt, g, p in zip(query_types, gold_ids, predicted_ids):
        per_type[qt]["n"] += 1
        per_type[qt]["correct"] += int(bool(g) and g == p)
    return {qt: {"n": v["n"], "accuracy": v["correct"] / v["n"]} for qt, v in per_type.items()}


def stage_firing_counts(stages_fired: list[str]) -> dict:
    """How often each cascade stage produced the final candidate set (v1.1 4.4:
    the counter table showing the deterministic stages doing the work)."""
    unknown = [s for s in stages_fired if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage names: {sorted(set(unknown))}")
    return dict(Counter(stages_fired))


def decision_rates(decisions: list[str]) -> dict:
    """answer/clarify/abstain/redirect shares over a query set."""
    counts = Counter(decisions)
    n = len(decisions)
    return {d: {"count": c, "rate": c / n} for d, c in counts.items()} | {"n": n}


def write_artifacts(*args, **kwargs):
    raise NotImplementedError("TODO(evaluation phase): persist to outputs/evaluation/retrieval/")
