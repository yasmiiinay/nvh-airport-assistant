"""Multimodal fusion metrics (RQ3): condition x query-type grid, conflicts,
cascade vs weighted-fusion comparison. Aggregation only; conditions come from
the multimodal manifest, outcomes from the Chat 03 pipeline."""
from __future__ import annotations
from collections import defaultdict

CONDITIONS = ["image_only", "text_only", "voice_only",
              "image_text_consistent", "image_text_conflict", "voice_image"]


def accuracy_by_condition(conditions: list[str], query_types: list[str],
                          gold_ids: list[str], predicted_ids: list[str]) -> dict:
    grid = defaultdict(lambda: {"n": 0, "correct": 0})
    for c, qt, g, p in zip(conditions, query_types, gold_ids, predicted_ids):
        if c not in CONDITIONS:
            raise ValueError(f"unknown condition '{c}'")
        cell = grid[(c, qt)]
        cell["n"] += 1
        cell["correct"] += int(bool(g) and g == p)
    return {f"{c}|{qt}": {"n": v["n"], "accuracy": v["correct"] / v["n"]}
            for (c, qt), v in grid.items()}


def conflict_detection(conflict_present: list[bool], conflict_flagged: list[bool]) -> dict:
    tp = sum(1 for p, f in zip(conflict_present, conflict_flagged) if p and f)
    fp = sum(1 for p, f in zip(conflict_present, conflict_flagged) if not p and f)
    fn = sum(1 for p, f in zip(conflict_present, conflict_flagged) if p and not f)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn,
            "n": len(conflict_present)}


def cascade_vs_fusion(*args, **kwargs):
    """TODO(post-MVP, gated): compare deterministic cascade against the
    weighted-fusion variant on the same scenario set, plus a weight-sensitivity
    curve. Runs only after cascade results exist (v1.1 section 5)."""
    raise NotImplementedError("Gated enhancement; requires cascade results first.")


def write_artifacts(*args, **kwargs):
    raise NotImplementedError("TODO(Chat 03+): persist to outputs/evaluation/multimodal/")
