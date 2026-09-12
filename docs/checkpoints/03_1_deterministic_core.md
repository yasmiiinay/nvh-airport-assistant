# Checkpoint 03.1 — Deterministic core

12 September 2026. Working mode: IMPLEMENT → TEST → MEASURE → RECORD.
Scope: `src/normalizer.py`, `src/entities.py`, the deterministic stages of
`src/retrieval.py`, and the evaluation functions that were blocked on the
normaliser. No pretrained model is loaded anywhere in this checkpoint.
Nothing in Architecture Freeze v1.1, the vocabulary or the KB structure was changed.

## 1. What was built and why it is shaped this way

**Normaliser.** Two levels in one module, each rule a row in a printable
table (`python -m src.normalizer`). L1 is the generic transform that the WER
computation also uses on both sides (lowercase, contraction expansion,
punctuation, whitespace, cardinal number words → digits, including the spoken
desk form "one forty five" → 145). L2 holds the airport repairs: letter words
and NATO words to pier letters, spacing collapse to `b12`, terminal short
forms, a few spelling variants and homophones. Two design choices worth
defending in the report: (i) bare `a`, `be` and `see` are mapped to a pier
letter only after `gate`/`pier`, because "a 12 o'clock flight" and "it will
be 12 minutes" would otherwise become gate identifiers — the unit tests pin
both non-firings; (ii) every homophone rule carries `provenance:
anticipated` until a real transcript shows the confusion, so the report can
state exactly which rules were evidence-driven. No stop word is ever removed.

**Entity extraction.** Regexes for the vocabulary's identifier forms, an
identifier index expanded from the KB ranges (140 tokens: 40 gates, 90 desks,
9 belts, P1), a service gazetteer built from record names and aliases
(116 phrases, normalised with the same normaliser as the queries so `t1
check-in` and `terminal 1 check in` are the same string), and a list of
volatile phrases taken from the volatility-`high` record. A well-formed
identifier is marked `exists=False` when it is outside every range — that is
what turns B21 into a grounded negative instead of a near-miss. The
`flight_ref` pattern is applied to the *raw* text in uppercase form and to
the normalised text only directly after the word `flight`, because after
lowercasing the vocabulary pattern would also match "at 10".

One derived structure goes slightly beyond "name + aliases": **category
cues**, single tokens that occur in the name/aliases of records of exactly
one category (74 tokens; function words excluded; the volatile record
excluded so `flight` cannot become an "information" cue). They never resolve
a record on their own. They are used in two narrow ways only — category +
terminal narrowing to exactly one record, and single-record categories — and
otherwise travel as hints in the hand-off. Their noise is real and is
reported in §5.

**Deterministic retrieval.** Order: volatile redirect → exact identifier →
alias/gazetteer → category cue (+ terminal) → deictic clarify → unresolved
hand-off. The redirect runs first on purpose: a flight reference means any KB
answer could be stale, so "what gate is flight XY456 leaving from" must not
become a gate answer. Two grounded-negative shapes are distinguished: a
non-existent identifier abstains (most likely misread/misheard; the passenger
is asked to check the boarding pass), while a service the KB holds only in
another terminal is answered as a "no" with a pointer. An unresolved result is
not a failure: it carries `candidates`, `category_hints`, `terminal` and a
`deictic` flag for the semantic stage.

**Speech metrics.** `wer_l1`, `wer_l2` and `identifier_token_accuracy` now
exist, defined on the same normaliser the pipeline uses, so the WER transform
and the retrieval input cannot drift apart. They are exercised on hand-written
string pairs only; there is no audio yet and no speech number is claimed.

## 2. Tests

`python -m pytest tests` — **81 passed** on the student's MacBook
(Python 3.14.3, macOS arm64, project venv with the pinned requirements;
12 Sep 2026) — the environment of record. The same suite also passed in the
Linux/Python 3.11 workspace where the code was written. `python
scripts/audit_foundation.py` ALL 11 CHECKS PASS on both; `python
scripts/run_deterministic_seed.py` on the MacBook reproduced the counts in
§3 exactly (22 decided, 21 unresolved, 0 false resolutions). Test files: `test_normalizer.py` (40 cases incl. must-not-fire
cases and rule-example self-checks), `test_entities.py` (22), `test_retrieval_deterministic.py`
(13, including a seed-set-wide "no false resolution" assertion), `test_speech_metrics.py` (4).

## 3. Seed-set run (`scripts/run_deterministic_seed.py`, 43 queries)

| Stage that produced the decision | Count |
|---|---|
| exact_identifier | 6 |
| alias_lookup (incl. category cue + terminal) | 13 |
| no_retrieval (scripted: redirect 2, deictic clarify 1) | 3 |
| **decided deterministically** | **22** |
| unresolved → semantic stage | 21 |

Of the 22 decisions, 22 match the seed's expected behaviour and target;
**0 false resolutions**. Decisions among the 22: answer 18, redirect 2,
abstain 1, clarify 1.

The 21 unresolved queries, by what the seed set expects the later stage to
do: **8 answer** (q009, q013, q015, q017, q026, q027, q031, q034),
**8 clarify** (q004, q005, q008, q010, q016, q021, q024, q029),
**1 redirect** (q037, volatile without a flight reference — needs the intent
stage), **4 abstain** (q038, q039, q040, q043). This is the semantic workload
for checkpoint 03.2.

Category-hint quality on the unresolved queries that have a target record:
7 carry hints, 6 include the target's category, 1 does not (q015 — the
designed baggage-vocabulary trap points at `baggage`, target is
`lost_property`). Hints must therefore be treated as evidence, not as a hard
filter; the frozen intent → category filter remains the filter.

Full table (from `outputs/checkpoint_03_1/seed_results.csv`):

| id | query | type | expected | stage | decision | record / candidates | flags | verdict |
|---|---|---|---|---|---|---|---|---|
| q001 | Where is gate B12? | precise | answer → gates_pier_b | exact_identifier | answer | gates_pier_b | - | correct |
| q002 | How do I get to A7 | precise | answer → gates_pier_a | exact_identifier | answer | gates_pier_a | - | correct |
| q003 | gate c3? | precise | answer → gates_pier_c | exact_identifier | answer | gates_pier_c | - | correct |
| q004 | Where do I board my flight? | vague | clarify → - | unresolved | - | hints - | - | handed to semantic |
| q005 | Where can I check in? | vague | clarify → - | unresolved | - | hints - | - | handed to semantic |
| q006 | Check-in desks Terminal 2 | precise | answer → checkin_t2 | alias_lookup | answer | checkin_t2 | - | correct |
| q007 | Where is desk 145? | precise | answer → checkin_t1 | exact_identifier | answer | checkin_t1 | - | correct |
| q008 | Where is security? | vague | clarify → - | unresolved | - | hints security; 3 cand. | - | handed to semantic |
| q009 | How long is the queue at security north? | ambiguous | answer → security_t1_north | unresolved | - | hints security; 3 cand. | - | handed to semantic |
| q010 | Where do I pick up my suitcase? | paraphrase | clarify → - | unresolved | - | hints - | - | handed to semantic |
| q011 | Baggage claim terminal 1 | precise | answer → baggage_reclaim_t1 | alias_lookup | answer | baggage_reclaim_t1 | - | correct |
| q012 | Where is belt 8? | precise | answer → baggage_reclaim_t2 | exact_identifier | answer | baggage_reclaim_t2 | - | correct |
| q013 | I left my bag on the plane | paraphrase | answer → lost_property_t1 | unresolved | - | hints baggage, lost_property | - | handed to semantic |
| q014 | lost and found | precise | answer → lost_property_t1 | alias_lookup | answer | lost_property_t1 | - | correct |
| q015 | My bag never arrived on the belt - where do I complain? | ambiguous | answer → lost_property_t1 | unresolved | - | hints baggage; 2 cand. | - | handed to semantic |
| q016 | Where can I get help? | vague | clarify → - | unresolved | - | hints information; 4 cand. | - | handed to semantic |
| q017 | information desk arrivals | precise | answer → info_desk_t1_arrivals | unresolved | - | hints information; 4 cand. | - | handed to semantic |
| q018 | Is there a lounge? | vague | answer → lounge_aurora | alias_lookup | answer | lounge_aurora | - | correct |
| q019 | Is there a lounge in Terminal 2? | ambiguous | answer → lounge_aurora | alias_lookup | answer | {lounge_aurora} | grounded_negative | correct |
| q020 | What time does the lounge open? | precise | answer → lounge_aurora | alias_lookup | answer | lounge_aurora | - | correct |
| q021 | Where can I eat something before my flight? | vague | clarify → - | unresolved | - | hints - | - | handed to semantic |
| q022 | Skyline restaurant opening hours | precise | answer → restaurant_skyline | alias_lookup | answer | restaurant_skyline | - | correct |
| q023 | Is the lounge open right now? | ambiguous | answer → lounge_aurora | alias_lookup | answer | lounge_aurora | time_reference_no_clock | correct |
| q024 | Where is the nearest toilet? | vague | clarify → - | unresolved | - | hints restroom; 3 cand. | - | handed to semantic |
| q025 | accessible toilet terminal 2 | precise | answer → restrooms_t2 | alias_lookup | answer | restrooms_t2 | - | correct |
| q026 | I need wheelchair assistance | vague | answer → prm_point_t1_entrance | unresolved | - | hints accessibility; 4 cand. | - | handed to semantic |
| q027 | wheelchair help terminal 2 | precise | answer → prm_point_t2 | unresolved | - | hints accessibility, information | - | handed to semantic |
| q028 | Where is first aid? | precise | answer → first_aid_t1 | alias_lookup | answer | first_aid_t1 | - | correct |
| q029 | How do I get to the city centre? | vague | clarify → - | unresolved | - | hints transport; 5 cand. | - | handed to semantic |
| q030 | Where is the train station? | precise | answer → rail_station | alias_lookup | answer | rail_station | - | correct |
| q031 | taxi? | vague | answer → taxi_rank_t1 | unresolved | - | hints transport; 5 cand. | - | handed to semantic |
| q032 | Where does the airport bus leave from? | paraphrase | answer → bus_terminal | alias_lookup | answer | bus_terminal | - | correct |
| q033 | I parked in P1 - how do I get back there? | precise | answer → car_park_p1 | alias_lookup | answer | car_park_p1 | - | correct |
| q034 | How do I get from Terminal 1 to Terminal 2? | paraphrase | answer → shuttle_t1_t2 | unresolved | - | hints - | - | handed to semantic |
| q035 | Is my flight NH123 delayed? | volatile | redirect → flight_information | no_retrieval | redirect | flight_information | volatile | correct |
| q036 | What gate is flight XY456 leaving from? | volatile | redirect → flight_information | no_retrieval | redirect | flight_information | volatile | correct |
| q037 | when does boarding start | volatile | redirect → flight_information | unresolved | - | hints - | - | handed to semantic |
| q038 | Where can I smoke? | out_of_scope | abstain → - | unresolved | - | hints - | - | handed to semantic |
| q039 | Can you book me a taxi for 6pm? | out_of_scope | abstain → - | unresolved | - | hints transport; 5 cand. | time_reference_no_clock | handed to semantic |
| q040 | What is the wifi password? | out_of_scope | abstain → - | unresolved | - | hints - | - | handed to semantic |
| q041 | Where is gate B21? | ambiguous | abstain → - | exact_identifier | abstain | {gates_pier_b} | grounded_negative | correct |
| q042 | What does this sign mean? | deictic | clarify → - | no_retrieval | clarify | - | deictic | correct |
| q043 | Kayip esya ofisi nerede? | non_english | abstain → lost_property_t1 | unresolved | - | hints - | - | handed to semantic |

## 4. The nine difficult cases

- **q009** "How long is the queue at security north?" — unresolved; hint `security`, 3 candidates including the target. The queue-time guard is not a retrieval matter: the record's `availability_note` already states that live queue times are unavailable, so the response template must render that field for `volatility: medium` records. Semantic stage must pick `security_t1_north`.
- **q013** "I left my bag on the plane" — unresolved with conflicting hints `baggage` and `lost_property` (from `left luggage enquiries`). Correctly *not* decided here; goes to semantic retrieval where `retrieval_text` of lost property mentions items left on the plane.
- **q015** "My bag never arrived on the belt - where do I complain?" — unresolved; the only hint is `baggage`, which is the wrong category. Evidence that lexical cues mislead on exactly the query the set was designed to trap. Recorded, not repaired.
- **q019** "Is there a lounge in Terminal 2?" — **grounded negative, answer**: no lounge record in Terminal 2, pointer to `lounge_aurora`. Decided from KB structure without similarity.
- **q023** "Is the lounge open right now?" — resolved to `lounge_aurora` via alias `the lounge`, with flag `time_reference_no_clock`; the hours template must not claim open/closed.
- **q036** "What gate is flight XY456 leaving from?" — **redirect** on the flight reference, before the gate cue is considered.
- **q041** "Where is gate B21?" — **abstain** with reason "B21 is well-formed but outside the known range (B1 to B18)", candidates `gates_pier_b`.
- **q042** "What does this sign mean?" — **clarify** (deictic without a photo). The router will override this when an image is present.
- **q043** "Kayip esya ofisi nerede?" — unresolved with no entities and no hints; will reach the semantic stage and is expected to abstain there. No language detection exists, by design.

## 5. Weaknesses found (measured or reproduced, not hypothetical)

1. **Category cue noise.** Tokens such as `first`, `room`, `main`, `point`,
   `left`, `city`, `stand`, `down`, `sit` are single-category by accident of
   the KB wording. Reproduced failure: "first floor terminal 2" →
   grounded negative "no medical record in Terminal 2". This branch is
   correct on the seed set but must be re-measured on the held-out set;
   if its precision is poor, the fix is to derive cues from aliases and
   name heads only, or to keep a small exclusion list beside the KB
   (dev-split tuning, recorded).
2. **Bare-word aliases.** `taxi` (singular) and `lounge` alone are not
   aliases; `lounge` resolves through the single-record-category rule,
   `taxi` does not (transport has six records) and goes to the semantic
   stage. Alias wording may be tuned on the dev split later; not changed now.
3. **Spoken flight references** are recognised only directly after
   `flight` ("flight ba 2490"); "my flight is ba 2490" is missed. Kept tight
   deliberately; revisit with real transcripts.
4. **All homophone rules are anticipated**, none observed. The rule table
   marks them as such.
5. **`a N` never collapses** without a gate/pier keyword; a passenger saying
   "I'm at a 7" (no "gate") is not recognised. Accepted trade-off against
   "a 7 hour layover".

Category-cue table (derived, for inspection):

| Category | Cue tokens |
|---|---|
| accessibility | entrance, main, point, prm, wheelchair |
| baggage | bag, baggage, belts, claim, reclaim |
| check_in | desks |
| gate | area, gate, gates |
| information | desk, help, information |
| lost_property | enquiries, found, items, left, lost, office, property |
| lounge | aurora, lounge |
| medical | aid, first, medical, nurse, room |
| restaurant | cafe, café, coffee, down, harbour, landside, restaurant, shop, sit, skyline |
| restroom | accessible, airside, bathroom, restrooms, toilet, toilets, wc |
| security | checkpoint, north, security, south |
| transport | bus, cab, car, change, city, coach, multi, nordhaven, park, parked, parking, railway, rank, shuttle, stand, stops, storey, taxi, taxis, terminals, trains, transfer |

## 6. Foundation defects

None structural. Three housekeeping items fixed in this checkpoint:
`scripts/smoke_test.py` now reports jiwer's version via
`importlib.metadata`; the README's Space status was stale relative to the
Foundation Report and now describes the live Space; the README carries the
Space's YAML configuration so a plain `git push` to the Space deploys the
committed code at Chat 05. A GitHub→Space auto-sync workflow was added and
then removed on review: it would have forced the pinned-requirements build on
ZeroGPU as a side effect of the first push, which is a Chat 05 question, not a
03.1 one.

Dev-set annotation correction (not an architecture change): the seed row
for q009 recorded `entities_json = {"terminal": "Terminal 1"}` although the
query text never mentions a terminal — the annotation encoded the target's
terminal rather than a surface entity and would have penalised entity
extraction unfairly. Corrected to `{}` on review, with the reason noted in the
row; audit and tests re-run unchanged (ALL CHECKS PASS, 81 passed).

## 7. Recommendation for checkpoint 03.2

Implement MiniLM (all-MiniLM-L6-v2, frozen) for intent-by-nearest-exemplar
and semantic retrieval over `retrieval_text`, consuming `DeterministicResult.handoff`:

1. Author intent exemplars (dev-only file, 15 intents × 5–8 phrasings)
   **without copying seed queries verbatim**, otherwise intent P/R/F1 is circular.
2. Retrieval order stays frozen: intent → compatible_categories filter →
   cosine over filtered `retrieval_text` (NumPy) → whole-KB fallback →
   decision from score + margin. Treat `category_hints` as an evaluated
   variant, not the filter (q015).
3. Intent volatility drives the second redirect path (q037).
4. The 21 unresolved queries are the first semantic test set with an expected
   split of 8 answer / 8 clarify / 1 redirect / 4 abstain; report per-stage
   firing counts for the whole cascade and retrieval accuracy by query type.
5. Set `tau_high`, `tau_low`, `margin_delta` on the dev split with a coarse
   grid; leave the held-out split unauthored until the semantic stage works.
6. Add the `minilm` loader to `scripts/benchmark_env.py` and record load
   time and RSS on the MacBook.
