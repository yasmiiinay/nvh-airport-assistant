"""Deterministic stages of the retrieval cascade (Architecture Freeze v1.1 4.4).

Frozen order:  normalise -> exact identifier -> alias/gazetteer -> [intent ->
category filter -> semantic similarity -> decision].  This module implements
everything before the first bracket, plus two scripted branches that must run
before any retrieval at all:

  * volatile redirect -- a flight reference or a volatile phrase means any
    KB answer could be stale, so the query is redirected to the official
    source *before* an identifier is looked up ("what gate is flight XY456
    leaving from" must not become a gate answer).
  * grounded negatives -- a well-formed identifier outside every KB range, a
    terminal outside the closed set, or a service category the KB holds only
    in another terminal. These are answered from KB structure, not from
    similarity, and are the cases a semantic stage would get wrong by
    returning the nearest plausible record.

Why the two grounded-negative shapes get different decisions: a non-existent
identifier ("B21") is most likely a misread or misheard token, so the system
abstains and asks the passenger to check the boarding pass. A service that
exists only elsewhere ("lounge in Terminal 2") is a real absence in the KB,
so the system answers "no" and points to where it is.

Anything not decided here is handed on unresolved with the evidence collected
so far (candidates, category hints, terminal). Unresolved is not a failure of
this stage: the paraphrase, vague and out-of-scope queries are the semantic
stage's job by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from src.entities import Extraction, Gazetteers, extract

STAGE_EXACT = "exact_identifier"
STAGE_ALIAS = "alias_lookup"
STAGE_NONE = "no_retrieval"   # scripted branches: redirect, deictic clarify


@dataclass
class DeterministicResult:
    query: str
    normalized: str
    entities: dict[str, str]
    resolved: bool
    stage: str | None = None          # one of evaluation.retrieval_metrics.STAGES, or None if unresolved
    decision: str | None = None       # answer / clarify / abstain / redirect, or None if unresolved
    matched_record_id: str | None = None
    candidates: list[str] = field(default_factory=list)
    reason: str = ""
    flags: list[str] = field(default_factory=list)
    handoff: dict = field(default_factory=dict)   # what the semantic stage receives

    def as_dict(self) -> dict:
        return asdict(self)


def _decide(result: DeterministicResult, stage: str, decision: str, reason: str,
            record_id: str | None = None, candidates: list[str] | None = None) -> DeterministicResult:
    result.resolved = True
    result.stage, result.decision, result.reason = stage, decision, reason
    result.matched_record_id = record_id
    result.candidates = candidates if candidates is not None else ([record_id] if record_id else [])
    return result


def _identifier_range_text(gaz: Gazetteers, prefix: str) -> str:
    owners = gaz.range_owner(prefix)
    if not owners:
        return "no such range in the knowledge base"
    return "; ".join(f"{shown}{lo} to {shown}{hi}" for _, lo, hi, shown in owners)


def resolve_deterministic(query: str, gaz: Gazetteers) -> DeterministicResult:
    """Run the deterministic stages on one text query.

    Returns a resolved result (stage + decision set) or an unresolved one whose
    `handoff` carries candidates and category hints for the semantic stage.
    """
    ex: Extraction = extract(query, gaz)
    result = DeterministicResult(query=ex.raw, normalized=ex.normalized,
                                 entities=ex.as_json_dict(), resolved=False)
    if ex.of_type("time"):
        # extracted but never reasoned about: the response layer must not
        # claim open/closed or compute waits (MVP has no clock)
        result.flags.append("time_reference_no_clock")

    terminals = ex.of_type("terminal")
    terminal_values = {t.value for t in terminals if t.exists}
    unknown_terminals = [t.value for t in terminals if not t.exists]

    # ---- scripted branch 1: volatile redirect (before any lookup) ----
    volatile_hits = [e for e in ex.of_type("service")
                     if e.record_ids and gaz.records[e.record_ids[0]].get("volatility") == "high"]
    if ex.of_type("flight_ref") or volatile_hits:
        target = (volatile_hits[0].record_ids[0] if volatile_hits
                  else next(rid for rid, r in gaz.records.items() if r.get("volatility") == "high"))
        trigger = ("flight reference " + ex.of_type("flight_ref")[0].value if ex.of_type("flight_ref")
                   else f"volatile phrase '{volatile_hits[0].surface}'")
        result.flags.append("volatile")
        return _decide(result, STAGE_NONE, "redirect",
                       f"{trigger}: live flight data is never answered from the KB", target)

    # ---- stage 1: exact identifier ----
    identifiers = [e for e in ex.entities if e.type in ("gate_id", "desk_id", "belt_id")]
    missing = [e for e in identifiers if not e.exists]
    if missing or unknown_terminals:
        parts = []
        for e in missing:
            prefix = e.value.rstrip("0123456789")
            parts.append(f"{e.value} is well-formed but outside the known range ({_identifier_range_text(gaz, prefix)})")
        for t in unknown_terminals:
            parts.append(f"{t} does not exist (terminals: {', '.join(sorted(gaz.terminals))})")
        alternatives = sorted({owner[0] for e in missing
                               for owner in gaz.range_owner(e.value.rstrip("0123456789"))})
        result.flags.append("grounded_negative")
        return _decide(result, STAGE_EXACT, "abstain", "; ".join(parts), None, alternatives)
    if identifiers:
        targets = sorted({e.record_ids[0] for e in identifiers})
        if len(targets) == 1:
            return _decide(result, STAGE_EXACT, "answer",
                           f"exact identifier {identifiers[0].value}", targets[0])
        result.flags.append("conflicting_identifiers")
        return _decide(result, STAGE_EXACT, "clarify",
                       "identifiers point to different records: " + ", ".join(e.value for e in identifiers),
                       None, targets)

    # ---- stage 2: alias / gazetteer ----
    alias_hits = ex.of_type("service")
    alias_targets = sorted({e.record_ids[0] for e in alias_hits if e.record_ids})
    if len(alias_targets) == 1:
        target = alias_targets[0]
        if terminal_values and gaz.records[target]["terminal"] not in terminal_values:
            # the service exists, but not where the passenger asked; the
            # response must say so rather than silently answer for Terminal 1
            result.flags.append("terminal_mismatch")
        return _decide(result, STAGE_ALIAS, "answer", f"alias '{alias_hits[0].surface}'", target)
    if len(alias_targets) > 1:
        narrowed = [rid for rid in alias_targets if gaz.records[rid]["terminal"] in terminal_values]
        if len(narrowed) == 1:
            return _decide(result, STAGE_ALIAS, "answer",
                           f"{len(alias_targets)} alias matches narrowed by terminal", narrowed[0])
        result.candidates = alias_targets
        result.reason = "several aliases matched; not narrowed by terminal"

    # ---- stage 2b: category cue (+ terminal) ----
    cue_categories = sorted({c for _, c in ex.category_cues})
    hint_records: list[str] = []
    if len(cue_categories) == 1 and not alias_targets:
        category = cue_categories[0]
        in_category = gaz.records_in_category(category)
        cue_text = ", ".join(sorted({t for t, _ in ex.category_cues}))
        if terminal_values:
            in_terminal = [rid for rid in in_category if gaz.records[rid]["terminal"] in terminal_values]
            if len(in_terminal) == 1:
                return _decide(result, STAGE_ALIAS, "answer",
                               f"category cue '{cue_text}' + terminal narrows to one record", in_terminal[0])
            if not in_terminal and in_category:
                result.flags.append("grounded_negative")
                return _decide(result, STAGE_ALIAS, "answer",
                               f"no '{category}' record in {', '.join(sorted(terminal_values))}; "
                               f"exists elsewhere", None, in_category)
            hint_records = in_terminal
        elif len(in_category) == 1:
            return _decide(result, STAGE_ALIAS, "answer",
                           f"category cue '{cue_text}': single record in category", in_category[0])
        else:
            hint_records = in_category

    # ---- scripted branch 2: deictic text with no other evidence ----
    if ex.of_type("deictic_ref") and not (alias_targets or cue_categories or terminal_values):
        result.flags.append("deictic")
        return _decide(result, STAGE_NONE, "clarify",
                       f"deictic reference '{ex.of_type('deictic_ref')[0].value}' without a photo", None)

    # ---- unresolved: hand on to the semantic stage ----
    result.handoff = {
        "category_hints": cue_categories,
        "terminal": sorted(terminal_values),
        "candidates": result.candidates or hint_records,
        "deictic": bool(ex.of_type("deictic_ref")),
    }
    if not result.reason:
        result.reason = "no identifier, alias or decisive category cue"
    return result
