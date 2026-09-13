# Checkpoint 03.2 — Text intelligence and semantic retrieval

13 September 2026. Working mode: IMPLEMENT → TEST → MEASURE → RECORD.
Scope: intent by nearest exemplar (all-MiniLM-L6-v2, frozen), semantic KB
retrieval over `retrieval_text`, integration into the frozen cascade,
text/retrieval evaluation, a TF-IDF + logistic-regression intent baseline,
and the MiniLM rows of the environment benchmark. Nothing from checkpoint
03.1 was redesigned; two evaluation-harness corrections are listed in §8.

## 1. What was built

**Encoder** (`src/text_encoder.py`). One shared MiniLM instance, created on
first use, never at import. A local copy under `models/` is used when
present (that is how the weights reached this workspace, which cannot reach
the hub), otherwise the hub id from settings. Vectors are L2-normalised, so
every similarity in this checkpoint is a dot product.

**Intent** (`src/intent.py`, `data/text/intent_exemplars.csv`). 95 authored
phrasings, 5 to 8 per intent, all marked `dev`; a test proves none is a
normalised copy of a seed query. A query takes the intent of its nearest
exemplar and the prediction carries that exemplar's text, the runner-up
intent and the margin between them, so every decision can be read back.
Below `tau_intent` the intent is `none`, which is how out-of-scope requests
reach the whole-KB fallback and the abstain path.

**Semantic stages** (`src/retrieval.py`). `resolve_semantic` runs only on a
result the deterministic stages left unresolved, in the frozen order:
intent → compatible-category candidates (narrowed by a terminal entity when
one was extracted) → cosine over the candidates' `retrieval_text` →
whole-KB fallback when the candidate set is empty → answer / clarify /
abstain from the top score and the top1−top2 margin. A volatile intent
(`ask_flight_status`) redirects before any similarity is computed. One
policy comes straight from a vocabulary field: intents whose
`response_type` is `assist` answer the nearest designated point instead of
asking a follow-up question. `candidate_records` takes a `filter_mode` so
the lexical category cues from 03.1 can be swapped in as the filter for the
comparison in §6; the frozen default is the intent filter.

**Baseline** (in `scripts/run_text_pipeline_seed.py`). TF-IDF word 1-2 grams
plus logistic regression, trained on the same 95 exemplars, scored on the
same queries.

## 2. Tests

`python -m pytest tests` — **94 passed** (Linux, Python 3.11; the MacBook run
is pending). New: `test_intent.py` (4: exemplar coverage and counts, no seed
copies, nearest-exemplar and margin on synthetic vectors, floor),
`test_retrieval_semantic.py` (9: decision rules, candidate filter modes,
deterministic queries never reach the encoder, thresholds must be set,
volatile intent redirects, out-of-scope abstains, vague query clarifies with
the right candidates, assistance policy, and a seed-set-wide property that
the semantic stage never answers with a record other than the target). The
model-dependent tests skip cleanly when MiniLM is unavailable.
`scripts/audit_foundation.py` ALL 11 CHECKS PASS; `tests/test_kb_consistency.py` PASS.

## 3. Intent results (43 seed queries, `tau_intent` = 0.30)

Accuracy 0.791 (34/43), macro F1 0.768. Per intent:

| Intent | P | R | F1 | n |
|---|---|---|---|---|
| ask_directions | 1.00 | 0.50 | 0.67 | 2 |
| ask_flight_status | 1.00 | 1.00 | 1.00 | 3 |
| ask_opening_hours | 0.67 | 0.67 | 0.67 | 3 |
| find_baggage_reclaim | 0.75 | 1.00 | 0.86 | 3 |
| find_check_in | 0.60 | 1.00 | 0.75 | 3 |
| find_gate | 1.00 | 0.80 | 0.89 | 5 |
| find_information | 1.00 | 0.50 | 0.67 | 2 |
| find_lounge | 1.00 | 1.00 | 1.00 | 2 |
| find_restaurant | 0.00 | 0.00 | 0.00 | 1 |
| find_restroom | 1.00 | 1.00 | 1.00 | 2 |
| find_security | 1.00 | 1.00 | 1.00 | 2 |
| find_transport | 0.57 | 1.00 | 0.73 | 4 |
| identify_sign | 1.00 | 1.00 | 1.00 | 1 |
| none | 0.50 | 0.33 | 0.40 | 3 |
| report_lost_item | 1.00 | 0.50 | 0.67 | 4 |
| request_accessibility_help | 1.00 | 1.00 | 1.00 | 3 |

Confusion pairs (gold → predicted): find_gate → find_check_in (q004, nearest
exemplar "I have to check in for my flight where do I go"); find_restaurant →
find_check_in (q021, "where can I hand in my luggage before the flight");
report_lost_item → find_baggage_reclaim (q015, "where do I get my checked
bag"); find_information → ask_opening_hours (q017, "until what time is the
information desk staffed"); ask_directions → find_transport (q034, "how do I
get to the other terminal"); ask_opening_hours → find_restaurant (q022,
resolved deterministically anyway); none → find_transport (q038 smoke, q039
taxi booking); report_lost_item → none (q043, Turkish, by the floor).

Two of the errors share a cause: both wrong `find_check_in` exemplars contain
"flight", and "my flight" in the query outweighs "board" or "eat". This is
an exemplar-authoring finding rather than a model finding. The exemplars
were not revised, because revising them against the same 43 queries would
be tuning on the evaluation set; a declared revision is proposed for the
next phase and must be checked on the held-out set.

`tau_intent` sweep (accuracy on all 43): 0.0 → 0.767, 0.30 → 0.791, 0.40 →
0.791, 0.50 → 0.744. The floor only changes the three out-of-scope rows and
q043; 0.30 was chosen as the lowest value that gains them.

**Baseline on the same split** (40 queries with one of the 15 intents;
`none` excluded because the baseline cannot predict it): TF-IDF + LR
accuracy 0.825, macro F1 0.800; MiniLM nearest exemplar accuracy 0.825,
macro F1 0.781. On this dev set the sentence encoder does not beat the
bag-of-words baseline for intent. The encoder earns its place in retrieval
(§4, paraphrase matching over `retrieval_text`), not in intent
classification, and the report should say so plainly.

## 4. Full cascade (thresholds `tau_high` 0.50, `tau_low` 0.25, `margin_delta` 0.10)

Stage that produced the decision, all 43 queries:

| Stage | Count |
|---|---|
| exact_identifier | 6 |
| alias_lookup | 13 |
| no_retrieval (scripted: 3 redirect, 1 deictic clarify) | 4 |
| category_filter_semantic | 18 |
| semantic_full_kb | 2 |

Decisions: answer 21, clarify 14, abstain 5, redirect 3. Outcome correct
(decision and record as the seed expects): **36/43**, against 22/43 for the
deterministic stages alone. On the 21-query semantic workload: 14 correct,
7 not as expected, **0 answers with a wrong record** (the property the
test suite now asserts). Semantic-stage decisions: clarify 13, abstain 4,
answer 3, redirect 1.

Outcome accuracy by query type (decision and record): precise 15/16, vague
9/10, ambiguous 3/5, paraphrase 2/4, volatile 3/3, out_of_scope 2/3, deictic
1/1, non_english 1/1. Record accuracy by type (target record returned, so
correct clarify/abstain counts as 0): precise 0.94, vague 0.30, ambiguous
0.20, paraphrase 0.25, volatile 1.00.

Semantic workload in full:

| id | query | expected | intent (score) | stage | decision | top record (score, margin) | verdict |
|---|---|---|---|---|---|---|---|
| q004 | Where do I board my flight? | clarify → - | find_check_in (0.7199) | category_filter_semantic | clarify | checkin_t2 (0.2599, 0.0092) | correct |
| q005 | Where can I check in? | clarify → - | find_check_in (0.5014) | category_filter_semantic | clarify | checkin_t1 (0.3043, 0.0281) | correct |
| q008 | Where is security? | clarify → - | find_security (0.8089) | category_filter_semantic | clarify | security_t1_south (0.5006, 0.0018) | correct |
| q009 | How long is the queue at security north? | answer → security_t1_north | find_security (0.4758) | category_filter_semantic | clarify | security_t1_north (0.4193, 0.0292) | wrong: decision clarify != expected answer |
| q010 | Where do I pick up my suitcase? | clarify → - | find_baggage_reclaim (0.745) | category_filter_semantic | clarify | baggage_reclaim_t1 (0.3563, 0.0417) | correct |
| q013 | I left my bag on the plane | answer → lost_property_t1 | report_lost_item (0.6795) | category_filter_semantic | clarify | lost_property_t1 (0.3675, 0.3675) | wrong: decision clarify != expected answer |
| q015 | My bag never arrived on the belt - where do I complain? | answer → lost_property_t1 | find_baggage_reclaim (0.5573) | category_filter_semantic | clarify | baggage_reclaim_t1 (0.4575, 0.0221) | wrong: decision clarify != expected answer |
| q016 | Where can I get help? | clarify → - | find_information (0.7649) | category_filter_semantic | clarify | info_desk_t2 (0.3135, 0.044) | correct |
| q017 | information desk arrivals | answer → info_desk_t1_arrivals | ask_opening_hours (0.69) | category_filter_semantic | clarify | info_desk_t1_arrivals (0.4927, 0.0289) | wrong: decision clarify != expected answer |
| q021 | Where can I eat something before my flight? | clarify → - | find_check_in (0.6279) | category_filter_semantic | abstain | checkin_t1 (0.1526, 0.0278) | wrong: decision abstain != expected clarify |
| q024 | Where is the nearest toilet? | clarify → - | find_restroom (0.7548) | category_filter_semantic | clarify | restrooms_t2 (0.6596, 0.057) | correct |
| q026 | I need wheelchair assistance | answer → prm_point_t1_entrance | request_accessibility_help (0.9248) | category_filter_semantic | answer | prm_point_t1_entrance (0.4202, 0.0358) | correct |
| q027 | wheelchair help terminal 2 | answer → prm_point_t2 | request_accessibility_help (0.791) | category_filter_semantic | answer | prm_point_t2 (0.542, 0.174) | correct |
| q029 | How do I get to the city centre? | clarify → - | find_transport (0.6237) | category_filter_semantic | clarify | rail_station (0.4263, 0.0817) | correct |
| q031 | taxi? | answer → taxi_rank_t1 | find_transport (0.8062) | category_filter_semantic | answer | taxi_rank_t1 (0.5873, 0.2298) | correct |
| q034 | How do I get from Terminal 1 to Terminal 2? | answer → shuttle_t1_t2 | find_transport (0.8179) | category_filter_semantic | clarify | shuttle_t1_t2 (0.4675, 0.0301) | wrong: decision clarify != expected answer |
| q037 | when does boarding start | redirect → flight_information | ask_flight_status (0.9887) | no_retrieval | redirect | flight_information | correct |
| q038 | Where can I smoke? | abstain → - | find_transport (0.4872) | category_filter_semantic | abstain | car_park_p1 (0.1711, 0.0185) | correct |
| q039 | Can you book me a taxi for 6pm? | abstain → - | find_transport (0.6953) | category_filter_semantic | clarify | taxi_rank_t1 (0.4592, 0.1992) | wrong: decision clarify != expected abstain |
| q040 | What is the wifi password? | abstain → - | none (0.2676) | semantic_full_kb | abstain | lost_property_t1 (0.2019, 0.0118) | correct |
| q043 | Kayip esya ofisi nerede? | abstain → lost_property_t1 | none (0.2432) | semantic_full_kb | abstain | flight_information (0.1395, 0.0264) | correct |

## 5. Thresholds

Coarse grid on the dev split only: `tau_high` ∈ {0.40, 0.45, 0.50, 0.55},
`tau_low` ∈ {0.20, 0.25, 0.30}, `margin_delta` ∈ {0.03, 0.05, 0.10}; 36
valid settings, scored by correct outcomes on the 21 semantic-workload
queries and by the number of answers given with a wrong record. Top rows:

| tau_high | tau_low | margin | correct of 21 | answer | clarify | abstain | wrong-record answers |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.25 | 0.1 | 14 | 3 | 13 | 4 | 0 |
| 0.55 | 0.25 | 0.1 | 14 | 3 | 13 | 4 | 0 |
| 0.4 | 0.25 | 0.1 | 14 | 4 | 12 | 4 | 1 |
| 0.45 | 0.25 | 0.1 | 14 | 4 | 12 | 4 | 1 |
| 0.45 | 0.25 | 0.03 | 14 | 6 | 10 | 4 | 2 |
| 0.5 | 0.2 | 0.1 | 13 | 3 | 14 | 3 | 0 |
| 0.5 | 0.3 | 0.1 | 13 | 3 | 12 | 5 | 0 |
| 0.55 | 0.2 | 0.1 | 13 | 3 | 14 | 3 | 0 |
| 0.55 | 0.3 | 0.1 | 13 | 3 | 12 | 5 | 0 |
| 0.4 | 0.2 | 0.1 | 13 | 4 | 13 | 3 | 1 |

Five settings tie at 14/21. `tau_high` 0.50 / `tau_low` 0.25 / `margin` 0.10
was chosen because it reaches the maximum with zero wrong-record answers
(the failure the freeze treats as worst); the settings that lower `tau_high`
or the margin reach the same count only by adding one or two wrong-record
answers. Sensitivity reads simply: the margin is the binding constraint.
Most KB records come in terminal pairs whose `retrieval_text` differ by one
token, so the top1−top2 gap between twins is 0.02 to 0.04 and a margin of
0.10 turns every "which terminal" ambiguity into a clarify. That is the
right behaviour for q005, q010 and q016, and the wrong one for q017 and
q034, where the query does carry the disambiguating word. With 21 queries
these values are indicative; they will be re-measured on the held-out set
and the dev/held-out gap reported. Displayed scores stay cosine
similarities; no probability is claimed.

## 6. Category-cue experiment

| Candidate filter | Correct of 21 | Top-1 is target (8 answer cases) | q015 top-1 |
|---|---|---|---|
| intent (frozen) | 14 | 7 | baggage_reclaim_t1 |
| lexical cues (03.1) | 14 | 7 | baggage_reclaim_t1 |
| none (whole KB) | 13 | 5 | baggage_reclaim_t1 |

Filtering by category helps: the whole-KB search loses two top-1 hits and
one outcome. The lexical cues neither help nor hurt on this set: wherever
a cue exists it names the same category the intent does, and on q015, the
query built to mislead them, the intent stage is misled identically
("bag", "belt" → baggage). The frozen intent filter stays; the cues remain
a recorded hand-off field. The interesting negative result is q015 itself:
both lexical and semantic evidence choose baggage, and the margin rule is
what stops the system answering with the wrong record.

## 7. Difficult cases

- **q037** "when does boarding start": redirect through intent volatility,
  score 0.989. The nearest exemplar ("when does boarding begin") is a close
  paraphrase, so this is an easy case by construction and should not be
  over-read.
- **q015**: intent find_baggage_reclaim (0.56), top record baggage_reclaim_t1
  (0.46, margin 0.02) → clarify. Not the expected answer, but not a wrong
  answer; the trap defeats similarity and lexical cues alike.
- **q013** "I left my bag on the plane": intent right (0.68), single
  candidate lost_property_t1 at 0.37 → clarify because 0.37 < `tau_high`.
  A single-candidate category has no margin to speak of; the threshold is
  doing the whole job and the score is low because `retrieval_text` says
  "items left on the plane" while the query says "bag".
- **q009** security north: intent right, top record right (north), margin
  0.029 → clarify. MiniLM does not separate north from south.
- **q017** "information desk arrivals": intent wrong (ask_opening_hours,
  via the staffed-desk exemplar), top record right (0.49) but margin 0.029
  to checkin_t1 → clarify.
- **q034** T1 to T2: intent find_transport (seed says ask_directions),
  top record right (shuttle, 0.47), margin 0.03 to rail → clarify.
- **q026 / q027**: assistance policy answers the designated point; q027
  narrows to prm_point_t2 by the terminal entity.
- **q031** "taxi?": taxi_rank_t1 at 0.59, margin 0.23 → answer.
- **q038 / q040 / q043**: abstain at 0.17 / 0.20 / 0.14. q040 sits just
  below `tau_low` = 0.25 and above 0.20, which is why 0.25 was chosen.
- **q039** "book me a taxi for 6pm": taxi_rank_t1 at 0.46, margin 0.20 →
  clarify. Expected abstain. Similarity cannot see that this is a booking
  request; a small action-verb check ("book", "reserve", "call") at the
  response layer is the natural fix and belongs to the next phase, not to
  retrieval.
- **q021** "eat something before my flight": intent wrong (find_check_in),
  0.15 among check-in records → abstain. Expected clarify. The exemplar
  finding in §3 explains it.

## 8. Defects and corrections

No defect in the 03.1 modules. Two corrections to the evaluation harness:
`judge_outcome` treated q043 as wrong because its seed row carries an
informational target although the expected behaviour is abstain; rows
expecting abstain or clarify are now judged on the decision alone.
`retrieval_accuracy_by_type` credits only returned records, which reads as
0 for a correct clarify, so `outcome_accuracy_by_type` was added beside it
and both are reported. One design observation for the KB, not changed:
twin records (T1/T2 versions of a service) have near-identical
`retrieval_text`; wording may be tuned on the dev split later, but every
such change must be declared.

## 9. Environment benchmark (this workspace, not the MacBook)

`scripts/benchmark_env.py` now has `minilm` (cold construction of the
encoder each run) and `minilm_encode` (three fixed queries through the
shared encoder). Measured here on Linux x86_64, Python 3.11, CPU, from the
local copy: first load 3.99 s, subsequent constructions 0.042 s median
(weights cached by the OS), RSS 793 MB after load (16 MB before; torch is
most of it); encoding three queries 10.6 ms median, about 3.5 ms per query.
These are workspace numbers. The MacBook rows must be produced by running
the same two commands there; none are claimed.

## 10. Recommendation for the next phase

1. Response assembly (template, KB-grounded, `volatility`-aware, with the
   `time_reference_no_clock`, `terminal_mismatch`, `grounded_negative` and
   `assist_policy` flags rendered as text) and the field-completeness test
   from the freeze. This closes the text-only path end to end and is where
   q039's action-request check belongs.
2. Author the held-out query split now that thresholds are fixed, without
   looking at the dev results while writing it; then report dev vs held-out
   for intent accuracy, outcome accuracy and the threshold choice.
3. One declared exemplar revision: remove "flight" from the two
   `find_check_in` exemplars and re-run; report before/after on both splits.
4. Run the two MiniLM benchmark commands on the MacBook and add the rows.
5. CLIP and Whisper only after review of this checkpoint.

---

# Closure addendum (13 September 2026)

Four steps were taken after the review of the checkpoint, in this order,
each committed separately so the sequence is auditable. No change was made
in response to the seven dev-set mismatches themselves.

## A. Held-out text set, authored blind and frozen first

`data/text/queries_heldout.csv`: 36 queries, all 15 intents plus `none`,
22 answer / 6 clarify / 5 abstain / 3 redirect; types precise 10, ambiguous
9, paraphrase 7, out_of_scope 3, volatile 3, vague 2, deictic 1,
non_english 1. Written from new passenger situations (damaged bag, doctor,
baby changing, cash machine, printing a boarding pass, a described sign
without a photo, German, a gate number without a pier letter, two gates in
one pier, a T2 question about a T1-only service). It passes the same audit
as the seed file and a test asserts that no held-out query equals a seed
query after normalisation and that token overlap with every seed query
stays below 0.6 (the maximum is 0.50, h012 against q027, different
intents). The file was committed (`396ccd7`) before the pipeline was run
on it. The runner gained `--set heldout`; in that mode neither the
threshold grid nor the cue experiment runs.

## B. Frozen pipeline on the held-out set (before any revision)

| | dev (43) | held-out (36) | gap |
|---|---|---|---|
| Outcome correct | 36 (0.837) | 24 (0.667) | −0.17 |
| Deterministic stages alone | 22, 0 wrong | 15, 0 wrong | |
| Semantic workload correct | 14 / 21 | 9 / 21 | |
| Semantic decisions (answer / clarify / abstain / redirect) | 3 / 13 / 4 / 1 | 2 / 15 / 2 / 2 | |
| Wrong-record answers | 0 | 1 (h019) | +1 |
| False grounded negative | 0 | 1 (h024) | +1 |
| Intent accuracy / macro F1 | 0.791 / 0.768 | 0.694 / 0.658 | −0.10 |

Held-out outcome by type: precise 9/10, ambiguous 8/9, volatile 3/3,
deictic 1/1, non-English 1/1, vague 1/2, paraphrase 1/7, out-of-scope 0/3.
The deterministic stages generalise (15 decisions, none wrong; the exact,
alias, redirect, grounded-negative and terminal-mismatch paths all fired as
designed). The semantic stage does not generalise well on paraphrases: six
of seven paraphrase queries ended in clarify because their top score stayed
below `tau_high` (h010 0.29, h021 0.27, h026 0.37, h014 0.46, h036 0.63 with
margin 0.08, h015 0.56 with margin 0.03). Out-of-scope handling failed on
all three: h019 "baby changing facilities" was answered with a restroom
record (0.56, margin 0.25), the first wrong-record answer of the project;
h023 "nearest cash machine" and h030 "print my boarding pass" clarified at
0.26 and 0.30, just above `tau_low`. Intent errors: 11 on 36 (list in
`outputs/checkpoint_03_2/heldout/intent_report.json`), including three
`none` queries absorbed by real intents.

**h024 is judged "correct" by the lenient rule and is in fact wrong.** "How
often does the shuttle to terminal 2 run" triggered the 03.1
category-plus-terminal grounded negative ("no transport record in Terminal
2") because the shuttle record is filed under Terminal 1 although it serves
both. The judge credits a grounded negative whose alternatives contain the
target; the passenger would read a false "no". This is a genuine 03.1
defect in the grounded-negative rule for cross-terminal services, recorded
here and left for the next phase rather than patched on the same day the
held-out set was first run.

## C. One declared exemplar revision

Change: the two `find_check_in` exemplars that contained "flight" became
"I have to check in where do I go" and "where can I hand in my luggage
before departure". Nothing else was touched and nothing was re-tuned.

| | dev before → after | held-out before → after |
|---|---|---|
| Intent accuracy | 0.791 → 0.814 | 0.694 → 0.722 |
| Intent macro F1 | 0.768 → 0.782 | 0.658 → 0.697 |
| Outcome correct | 36 → 35 | 24 → 24 |
| Wrong-record answers | 0 → 1 (q004) | 1 → 2 (h019, h011) |

The revision did what it targeted (find_gate→find_check_in and
find_information→find_check_in confusions gone) and made outcomes no
better. q004 "Where do I board my flight?" now has the right intent, and
the right intent narrows the search to three near-identical gate records
where pier B wins at 0.51 with a margin of 0.13, so the query is answered
with a gate area instead of asking for the gate number. h011 "Where can I
ask about flight connections?" gained the right intent (find_information)
and then ranked the `flight_information` record first at 0.52. Intent
accuracy and outcome quality are not the same thing; the report should
show this pair.

Baseline after the revision, same split: dev (40) TF-IDF+LR 0.825 / 0.800
against MiniLM 0.850 / 0.796; held-out (33) TF-IDF+LR 0.727 / 0.722
against MiniLM 0.788 / 0.752. The encoder is ahead on both after the
revision, by two to three queries; with these sample sizes that is
suggestive, not conclusive.

## D. Defect fix from the held-out run: the live-information record is never an answer

h011 exposed that the semantic stage could present `flight_information`
(volatility high) as an answer. The freeze says that record exists only to
carry the redirect. `resolve_semantic` now redirects whenever the top
record has volatility `high`, keeping the similarity stage name so the
stage counts stay honest. A regression test carries the h011 wording. After
the fix: dev 35/43, held-out 24/36; wrong-record answers dev 1 (q004),
held-out 1 (h019); h011 redirects (not the expected clarify, but not a
false answer). The seed-set property test now names q004 as the one known
wrong-record answer so that any new one fails the suite.

## E. Minimal response assembly (`src/responses.py`)

Template assembly from record fields only: name and location, description,
directions, opening hours, availability note (rendered for medium and high
volatility records, which is where "live queue times are not available"
lives), accessibility flags and notes, nearest assistance point, contact,
and a provenance line naming the record, its `last_verified` date and
either the deterministic stage or the match score with the frozen
disclaimer. Flags become sentences: `time_reference_no_clock` adds "I
cannot see the current time, so I cannot say whether it is open right now";
`grounded_negative` renders "There is no lounge in Terminal 2" plus the
nearest alternative; `terminal_mismatch` renders "Car Park P1 is not at
Terminal 2; it is at Terminal 1"; `assist_policy` introduces the designated
point; redirect renders the official-source note and never a KB location.
Two response-layer guards, both regex tables, both flagged in the result:
an action request ("book", "reserve", "print", "rebook", ...) prefixes "I
cannot book, reserve, print or arrange anything; I can only give
information", so q039 and h030 no longer imply the system can act; a live
status request ("queue", "waiting time", "how long", "busy") prefixes "I do
not have live queue or waiting times", so q009 and h007 never carry an
invented wait. Nine tests in `tests/test_responses.py` check field
completeness on a deterministic answer, the absence of open/closed and
wait-time claims, the redirect content, the grounded-negative and
terminal-mismatch wording, both guards, and the score disclaimer on a
semantic answer.

## F. State at closure

`python -m pytest tests`: **105 passed** (Linux, Python 3.11; MacBook run
pending). Audit ALL 11 CHECKS PASS. Held-out set frozen at `396ccd7` and
untouched since. Open items carried forward: the cross-terminal
grounded-negative defect (h024); out-of-scope queries that land just above
`tau_low` (h023, h030) and the one answered from a neighbouring category
(h019); paraphrase recall of the semantic stage; the dev/held-out gap of
0.17 in outcome accuracy, to be reported as such.
