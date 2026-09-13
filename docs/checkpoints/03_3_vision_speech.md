# Checkpoint 03.3 — Vision (CLIP) and speech (Whisper)

13 September 2026. Working mode: IMPLEMENT → TEST → MEASURE → RECORD.
Scope: the two remaining single-modality pipelines, each ending at the
point where the frozen text cascade takes over. No fusion, no user
interface, and no change to the text pipeline beyond two normaliser rules
whose evidence is given in §7 and whose effect on the text results is
shown to be nil. Status labels used throughout: IMPLEMENTED, EXECUTED,
VERIFIED (executed on the MacBook, the environment of record), NOT YET
EVALUATED, BLOCKED BY DATA.

Summary of status at the end of the checkpoint:

| Component | Status |
|---|---|
| Vision pipeline (`src/vision.py`) | IMPLEMENTED, EXECUTED on 101 labelled images (dev 37, held-out 64) in this workspace (§2b); MacBook re-run pending |
| Speech pipeline (`src/speech.py`) | IMPLEMENTED, VERIFIED: 120 synthetic clips (dev 100, held-out 20) run in this workspace and on the MacBook with identical transcripts and outcomes (§5) |
| Human recordings | VERIFIED on 5 clips from one speaker, both machines (§5b); too few for a rate, reported as cases |
| Vision thresholds | set on the dev images: 0.30 / 0.24 / 0.015 (§2b) |
| Text results after this checkpoint | unchanged, verified byte-for-byte (§7) |

## 1. Vision: what was built

`src/vision.py` is the whole image path up to the hand-off.

**Image checks.** `load_image` refuses unreadable files, images above 25
megapixels (refused before full decoding) and images whose shorter side is
under 64 px; EXIF orientation is applied so a phone photo taken sideways is
analysed the right way up. `check_image` computes a blur score (variance of
the Laplacian on the grey image, flagged below 60) and mean brightness
(flagged when very dark or very bright). Flags do not stop the analysis;
they travel with the result so a later response can say "the photo looks
blurred" and so the evaluation can split accuracy by stratum.

**Index.** One CLIP ViT-B/32 instance, loaded on first use from
`models/clip-vit-base-patch32/` when present (the hub is unreachable from
this workspace), otherwise from the hub id in settings. Three sets of
text vectors are built once: one vector per visual category (the mean of
that category's prompt vectors from `data/vision_prompts.json`, eleven
categories with two prompts each), one vector per out-of-scope anchor (six
sentences such as "a photograph of a person" or "a boarding pass or printed
document"), and one vector per KB record whose category has a visual class
(31 of the 32 records; `lost_property` has no visual class in the
vocabulary and is deliberately unreachable from an image). Record vectors
come from the same `retrieval_text` field the text cascade uses, so an
image and a typed query are matched against the same description of each
record.

**Ranking and decision.** An image vector is ranked against categories,
anchors and records by cosine similarity. The anchors compete in the same
ranking as the categories: when the best anchor beats the best category the
image is marked out of scope, which is the mechanism the Architecture
Freeze calls for instead of a bare similarity floor. The category and
record rankings then pass through the same `decide()` function as text,
with a separate vision threshold triple, and the decision is translated
into the three user-facing bands (strong match / uncertain, please confirm
/ no reliable match). The vision thresholds are `None` in settings; the
band is only produced when they are set, and they will be set on the dev
image split alone.

**Prompt file.** `data/vision_prompts.json` carries the prompt wording, the
anchors and a `_meta.revisions` list that is empty. Any rewording is a
declared revision made on the dev split and logged there, the same
discipline as the single exemplar revision in 03.2.

**Alternatives considered.** Fine-tuning a small classifier on the
pictograms was rejected: with no labelled images yet and a target of a few
dozen per category, a trained head would have nothing to generalise from,
and zero-shot CLIP with editable prompts keeps the categories tied to the
vocabulary rather than to whatever images happen to be collected. An OCR
path for text on signs remains an optional enhancement (requirements-ocr)
and is not part of this checkpoint. FAISS is not used at runtime: 11 + 6 +
31 vectors are a single matrix product.

## 2. Vision: data

**Pictograms (source `commons`).** The AIGA/DOT symbol signs, fetched from
the Wikimedia Commons category "AIGA symbol signs" with
`scripts/fetch_wikimedia_category.py` as 512 px server-side PNG renderings
(82 files, per-file licence recorded in `data/images/provenance_commons.csv`:
public domain or CC0, one CC BY 2.5 file excluded). 64 of them are in the
manifest with labels in `data/images/labels_aiga.csv`; the four non-AIGA
icons in the category and the CC BY file are left out. Symbols were mapped
to the vocabulary by their AIGA name (departing flights → `gate`, ticket
purchase and baggage check-in → `check_in`, waiting room → `lounge`, bus,
rail, taxi, car rental, parking → `transport`, and so on). Two vocabulary
categories have no AIGA symbol at all, `security` and `accessibility`; they
are BLOCKED BY DATA at category level and nothing is claimed about them.

Out-of-scope images are of two kinds, marked in the manifest notes and
reported separately: `generic:` symbols with no airport meaning (barber,
cashier, coat check, escalator, exit, fire extinguisher, litter, mail, no
smoking, no dogs, arrows, …) and `airport_adjacent:` symbols that belong in
an airport but not in this vocabulary (arriving flights, customs,
immigration, currency exchange, heliport, hotel information, water
transport, and lost and found, whose KB category has no visual class). The
second kind is the case predicted in the first draft of this document as
the weak spot for photo-description anchors.

**Photographs (source `own_photo`).** Ten phone photographs of the same
symbols displayed on a laptop screen, taken by the project author, some
deliberately angled or defocused. Single symbols were cropped from them (30
crops, `data/images/labels_own.csv`, `modified: yes`): 14 in-scope, 16
out-of-scope, strata `real_good_light` (21) and `real_degraded` (9). These
are photographs of a screen, not of signage in situ: they carry moiré,
softness, angle and defocus, but not the lighting, clutter and distance of
a real terminal. They are also not independent of the pictogram set: most
crops show symbol families that appear in the dev pictograms (bar,
aeroplane, parking, car rental), so the photo stratum tests the imaging
condition, not new symbols.

**Ordinary photographs (source `own_photo`, note `photo:`).** Seven of the
author's own photographs with no airport content (flowers, a shop window
of figurines, painted eggs in a basket, three sea and sky views, an
evening sky): the wrong-upload case the anchors were written for. An eighth
photograph was excluded because it shows a branded product, which the
collection rules (Evidence Pack A1: no third-party logos) do not allow.

**Splits.** Dev: 19 in-scope and 18 out-of-scope clean pictograms. Held-out:
7 clean in-scope pictograms, 14 in-scope photo crops, 43 out-of-scope (20
pictograms, 16 crops, 7 photographs). Variants of one symbol (inverted,
red, mirrored) are kept on the same side of the split. All photographs and
crops are held-out, so the thresholds were set on clean pictograms only.

## 2b. Vision results

**Two defects found by the first run, both fixed before any number was
kept.** The first dev run reported top-1 0.42 with dozens of different
symbols producing one identical embedding. The cause was the loader: the
Commons PNGs are black shapes on a transparent background, and a plain RGB
conversion drops the alpha channel, leaving an all-black image. Images with
an alpha channel are now composited on white (`flatten_on_white`, with a
test). The second was in the decision, not the data: the band was being
decided on the record ranking, and since the KB holds one record per
terminal for most categories, an image alone can never separate
`checkin_t1` from `checkin_t2`; every image would have landed in the
uncertain band by construction. The band is now decided on the category
ranking, and the records of the top category are returned as a ranked
candidate list for text or a follow-up question to narrow. That is a design
correction the data forced, and it fixes what an image can decide: a kind
of place, not which terminal's instance of it.

**Thresholds** (`vision_tau_high` 0.30, `vision_tau_low` 0.24,
`vision_margin_delta` 0.015), chosen on the dev split from an 18-cell grid.
CLIP category scores on these images occupy a narrow band, correct 0.28 to
0.37 and out-of-scope 0.26 to 0.36, so the score alone separates almost
nothing; the top1−top2 margin does (median 0.028 for correct in-scope,
0.005 for out-of-scope on dev). The chosen cell gives 12 strong / 6
uncertain / 1 no-match on dev in-scope with every strong match correct,
and 3 strong / 15 uncertain on dev out-of-scope. A stricter margin (0.020)
would cut the out-of-scope strong matches to 2 at the cost of 2 correct
strong matches; the choice is recorded, not claimed optimal.

| Measure | Dev (clean pictograms) | Held-out |
|---|---|---|
| In-scope images | 19 | 21 (7 clean, 12 photo good light, 2 photo degraded) |
| Top-1 category accuracy | 0.947 (18/19) | 0.857 (18/21) |
| Top-3 | 0.947 | 0.905 |
| By stratum, top-1 | clean 0.95 | clean 0.71 (5/7), photo good light 0.92 (11/12), photo degraded 1.0 (2/2) |
| In-scope bands | 12 strong (all correct), 6 uncertain (correct), 1 no-match (wrong) | 15 strong (all correct), 3 uncertain correct, 2 uncertain wrong, 1 strong wrong |
| Out-of-scope images | 18 | 43 (36 pictograms and crops, 7 photographs) |
| Out-of-scope marked by anchors | 0/18 | 9/43: photographs 7/7, pictograms 2/36 (stairs; defocused escalator, both via "a blank or blurry image") |
| Out-of-scope reaching "uncertain" or "no match" | 15/18 | 36/43 |
| Out-of-scope given a "strong match" (wrong-confident) | 3/18 | 7/43, all pictograms |
| Wrong-confident rate over all images | 3/37 = 0.08 | 8/64 = 0.125 |

Per category on held-out: gate 5/5, transport 5/5, baggage 2/2, restroom
2/2, information 1/1, restaurant 3/4, check_in 0/2.

**Reading the results.**

1. Category identification on clean pictograms works: 18/19 on dev, and the
   one miss is a white symbol on a transparent background
   (`aiga_railtransportation_25_white`), which after flattening on white is
   a blank image; the "blank or blurry" anchor won and the image was marked
   out of scope, which is the right behaviour for a blank image and a
   labelling limitation rather than a model error.
2. The held-out misses are interpretable. Both `check_in` images are the
   AIGA "baggage check-in" symbol, a suitcase, and CLIP read them as
   `baggage`; the manifest note predicted this confusion when the label
   was written. The clean coffee-shop symbol went to `transport` with
   `restaurant` third at a 0.008 margin (uncertain band, so no wrong
   answer was given). The suitcase case is a real ambiguity a passenger
   would also face, and it is the kind of case where text ("check-in" vs
   "reclaim") should override the image in fusion.
3. Photographs of the screen did not hurt category accuracy (11/12 in good
   light, 2/2 defocused) and the blur flag fired on all 9 degraded crops,
   but also on 6 of the 21 good-light crops: the Laplacian threshold of 60
   sits at the softness of a phone photo of a screen. The flag is
   informational only, so nothing was lost, but the threshold should be
   re-examined on real signage photos before it drives any behaviour.
4. **The out-of-scope anchors work for the case they were written for and
   not for pictograms.** All seven ordinary photographs were caught by an
   anchor (the sea and sky views by "a landscape or city photograph", the
   eggs by "food on a plate", the flowers and the shop window by "a blank or
   blurry image", a semantically loose but correct result), and their
   category scores (0.14 to 0.22) sat below `vision_tau_low` anyway, so
   they would have reached "no reliable match" even without anchors. A
   pictogram of a barber, by contrast, is closer to "an airport sign for …"
   than to any photograph description: 0/18 on dev and 2/36 on held-out
   pictograms were caught. What keeps most out-of-scope symbols out of a
   confident answer is the margin rule: 29/36 on held-out ended in the
   uncertain or no-match band. The residual
   7/36 wrong-confident cases are the main vision weakness: barber → 
   restroom, coat check → baggage, no entry → transport, litter →
   accessibility, arriving flights and heliport → baggage. A "strong match"
   on a symbol the KB does not cover is exactly the false certainty the
   ethics section has to discuss.
5. No prompt revision was made and `_meta.revisions` stays empty. Any
   anchor wording added now (stairs, arrows, shops) would be written with
   the labelled out-of-scope set in view, including the held-out half, and
   would measure the author's knowledge of the test set rather than the
   method. The honest options for Chat 04 are a pictogram-style anchor set
   written from the vocabulary alone before looking at images, or the
   fusion rule that an image-only "strong match" still asks the passenger
   to confirm the category when no text or voice agrees with it.
6. `security` and `accessibility` have no in-scope image at all, so nothing
   is known about them; both are predicted as runner-up categories often
   (accessibility second for escalators, litter, restrooms), which suggests
   the "person in a wheelchair" prompt attracts any human figure.

## 3. Speech: what was built

`src/speech.py` adds exactly two things in front of the text cascade and
nothing after it.

**Audio gate.** `load_audio` reads any format libsndfile handles, mixes to
mono and resamples to 16 kHz with a polyphase filter. `check_audio` then
applies duration bounds and a loudness floor (RMS below −45 dBFS is treated
as silence). A clip that fails is not transcribed; the result says why, so
the interface can ask the passenger to re-record or type.

**Transcription.** Whisper-base through the transformers pipeline on CPU,
English forced (`language=english, task=transcribe`) so that a non-English
utterance is transcribed as best-effort English rather than translated,
which is what lets the text cascade see it and abstain. Decoding is capped
at 64 new tokens. A transcript with no letters (Whisper emits runs of dots
or dashes for tones and noise) is treated as "no speech recognised" and the
clip is rejected after the gate. Both of these came from a measurement, not
a guess: see §9.

**Hand-off.** `process_transcript` runs the raw Whisper text through the
same L1 and L2 normaliser and the same entity extractor as typed text, keeps
all three forms (raw, L1, L2) and the identifiers found, and calls the
frozen `resolve()`. There is no speech-specific text handling anywhere;
the only place speech can influence text behaviour is the normaliser rule
table, and every rule there is printable with its provenance.

**Whisper-small.** Not compared in this checkpoint. Whisper-base already
takes about 1.5 s per clip on the workspace CPU (§8) and the failure
catalogue (§6) is dominated by short identifier tokens, which a larger
model may or may not fix; a comparison is cheap to run with the same
runner (`whisper_model_id` in settings) and is deferred to the evaluation
phase if the human recordings show the same pattern.

## 4. Data

**Synthetic voices (EXECUTED).** `data/audio/tts_utterances.csv` lists 24
utterances: 20 seed (dev) queries and 4 held-out queries, chosen to cover
every deterministic stage (gate, desk, belt identifiers; terminal only;
flight references), the assistance and out-of-scope intents, one
deliberately non-English seed query (q043, Turkish), and the short forms a
passenger would actually say. `scripts/make_tts_audio.py` rendered them on
the MacBook with five macOS voices (Daniel, Karen, Moira, Rishi, Samantha:
British, Australian, Irish, Indian and American English) into 16 kHz mono
WAV, 120 clips, 8 MB, committed under `data/audio/files/` with the manifest.
Speaker ids start with `tts_` so the runner can separate them from human
recordings.

These clips measure the pipeline on clean, well-articulated speech with a
single accent per voice and no background noise. They do not measure
hesitation, self-correction, room noise or a real passenger's phrasing, and
the held-out subset (4 utterances) is small. They are the first
measurement, not the evidence base for the report.

**Human recordings (NOT YET EVALUATED).** `scripts/add_recordings.py`
appends recordings named `<query_id>__<speaker>.<ext>` to the manifest,
converting to 16 kHz mono WAV on macOS. Speaker codes are pseudonymous
(`spk_01`, …). The runner takes `--speakers human` so their results are
reported separately from the synthetic voices, never pooled.

**Images.** See §2; 101 manifest rows: 64 pictograms, 30 photo crops, 7 photographs.

## 5. Speech results on the synthetic dev clips (100 clips, 5 voices × 20 utterances)

Run in this workspace (Linux x86_64, Python 3.11) and on the MacBook
(arm64, Python 3.14, CPU). Every count below (WER components, identifier
accuracy, propagation, failure types) is identical on the two machines, as
expected from greedy decoding and a deterministic cascade; only latency
differs and both values are given.

| Measure | Value |
|---|---|
| Gate rejections | 0 of 100 (6 of 100 at the original 1.0 s minimum; §7) |
| WER, L1 normalisation | 0.157 (45 substitutions, 1 deletion, 34 insertions, 510 reference words) |
| WER, L2 normalisation | 0.151 |
| WER, L1, English utterances only (95 clips) | 0.080 |
| WER, L2, English utterances only | 0.073 |
| Identifier-token accuracy (L2), 30 clips carrying a gate/desk/belt identifier | 0.90 (27/30) |
| Identifier-token accuracy (L1) | 0.87 |
| Outcome from typed reference correct | 95 / 100 |
| Outcome from transcript correct | 91 / 100 |
| Propagation gap | 0.04 |
| Transcript changed after L2 | 22 / 100, of which 18 kept the right outcome |
| Lost to transcription (typed right, transcript wrong) | 4 |
| Wrong-confident answers from a transcript | 1 (aud_044, §6) |
| Latency per clip, warmed up | workspace median 1.46 s, max 2.64 s (warm-up 11.3 s); MacBook median 0.34 s, max 0.85 s (warm-up 4.9 s) |

Per voice (20 clips each):

| Voice | WER L1 | WER L2 |
|---|---|---|
| tts_samantha (US) | 0.108 | 0.108 |
| tts_karen (AU) | 0.127 | 0.118 |
| tts_daniel (GB) | 0.157 | 0.157 |
| tts_moira (IE) | 0.167 | 0.167 |
| tts_rishi (IN) | 0.225 | 0.206 |

Reading these numbers. The overall WER is inflated by the one Turkish
utterance: five clips of four words that Whisper, forced to English,
renders as letter salad ("K.O.B.S.I.O. Officer Nairied.") contribute 20 of
the roughly 80 word errors. On the English utterances WER is 0.07 to 0.08,
which is in the range reported for Whisper-base on clean read speech, and
the accent ordering (US best, Indian English worst, a factor of two) matches
the direction reported in the literature for Whisper's English models. The
five typed-reference failures are all one query (q034, five voices), a
dev-set mismatch already recorded in 03.2; they are not speech errors and
are excluded from the "lost" count by construction.

L2 recovers two words over L1 (466 versus 464 hits), which is exactly the
two observed rules added in §7. No anticipated rule fired.

Held-out split (20 clips, 4 utterances × 5 voices), both machines: WER
0.092 (9 substitutions, 3 insertions, 130 reference words), identifier
accuracy 0.70 (7/10), typed 20 → transcript 17, gap 0.15, three losses,
all identifier errors, all ending in abstain or clarify; MacBook median
latency 0.41 s.

## 5b. Human recordings (5 clips, one speaker, MacBook microphone)

Five sentences recorded by the project author (`spk_01`) in a quiet room,
converted with `add_recordings.py`. The first attempt at scoring them gave
WER 1.0 and 0 of 5 outcomes, which turned out to be a labelling error, not
a speech result: the files were numbered by the speaker's own list
(Q001–Q005) while the script read those numbers as seed-query ids. Three
of the five sentences were not in either text set at all. They were
declared afterwards in `data/text/queries_spoken.csv` (`s001` "Where is
belt 9?", `s002` "Baggage claim terminal 2", `s003` "gate c1"), with intent,
entities and expected outcome, and the manifest rows were corrected before
re-scoring. The error and the correction are recorded because the wrong
number was briefly real output; the runner's per-clip transcripts are what
exposed it (Whisper had heard "Where is the train station?" for a clip
labelled "Where can I check in?").

| Clip | Spoken | Whisper | Outcome |
|---|---|---|---|
| aud_121 | Where is belt 9? | Various bads mine. | LOST: answer → abstain |
| aud_122 | Baggage claim terminal 2 | Beggich claim terminal 2. | kept: `baggage_reclaim_t2` (terminal entity plus the semantic stage) |
| aud_123 | when does boarding start | Wanda's boarding starts. | kept: redirect ("boarding" is a volatile phrase) |
| aud_124 | gate c1 | Gate C1. | kept: `gates_pier_c` |
| aud_125 | Where is the train station? | Where is the train station? | kept: `rail_station` |

WER 0.42 on 19 reference words (6 substitutions, 2 deletions), typed 5 →
transcript 4. Five clips are cases, not a rate; what they add is that the
one loss is the same class as the synthetic losses: "belt 9" became
"Various bads mine", the fourth "belt + digit" failure across four
different voices (three synthetic, one human). "Gate C1", the short
identifier that failed for the synthetic voices as "KHA 11" / "8-8-11" on
A11, was transcribed exactly here. Two transcripts were wrong but
harmless, and the reason is instructive: "Beggich claim terminal 2" still
carries the terminal entity and enough of the alias for the semantic stage,
and "Wanda's boarding starts" still carries the volatile phrase, so the
cascade reached the same decision from a damaged transcript.

Recording quality: peak amplitude 5–7 % of full scale, RMS −41 to −44 dBFS
(just above the −45 dBFS gate), 1.0–1.4 s of leading silence. Re-running
Whisper on the same clips with an 8× gain produced the same five
transcripts, so level did not cause the "belt 9" loss. The −45 dBFS floor
is close to what a laptop microphone at arm's length produces; a passenger
holding a phone will be louder, but the floor should be checked against a
few more human clips before it is called final.

## 6. Failure catalogue (every changed transcript on both splits)

Dev split, 22 changed transcripts:

| Clip | Voice | Reference | Whisper | Type | Outcome |
|---|---|---|---|---|---|
| aud_003 | daniel | gate c3? | Gabe C3 | substitution | kept (identifier C3 survived) |
| aud_012 | daniel | wheelchair help terminal 2 | Wheelchair held terminal 2. | substitution | kept |
| aud_016 | daniel | Is my flight NH123 delayed? | is my flight NH-123 delayed. | insertion | kept (redirect) |
| aud_017 | daniel | What gate is flight XY456 leaving from? | What Gator's Flight XY 456 leaving from? | substitution | kept (redirect) |
| aud_020 | daniel | Kayip esya ofisi nerede? | K.O.B.S.I.O. Officer Nairied. | insertion | kept (abstain) |
| aud_031 | samantha | Where is belt 8? | Where is bell date? | identifier | LOST: answer → abstain |
| aud_036 | samantha | wheelchair help terminal 2 | Wheelchair Health Terminal 2 | substitution | kept |
| aud_040 | samantha | Is my flight NH123 delayed? | is my flight and H-123 delayed. | insertion | kept (redirect) |
| aud_044 | samantha | Kayip esya ofisi nerede? | K-A-BSI Office-Enerid? | insertion | LOST: abstain → answer `lost_property_t1` |
| aud_055 | karen | Where is belt 8? | Where is Bill Hague? | identifier | LOST: answer → abstain |
| aud_060 | karen | wheelchair help terminal 2 | wheelchair held terminal 2. | substitution | kept |
| aud_064 | karen | Is my flight NH123 delayed? | is my flight in H-123 delayed. | insertion | kept (redirect) |
| aud_065 | karen | What gate is flight XY456 leaving from? | What gate is Flight XY 456 leaving from? | insertion | kept (redirect) |
| aud_068 | karen | Kayip esya ofisi nerede? | KBSI are obviously narrowed. | substitution | kept (abstain) |
| aud_079 | moira | Where is belt 8? | Where is Beltaid? | identifier | LOST: answer → clarify |
| aud_088 | moira | Is my flight NH123 delayed? | Is my flight NH-123 delayed? | insertion | kept (redirect) |
| aud_089 | moira | What gate is flight XY456 leaving from? | What gate is Flight XY 456 leaving from? | insertion | kept (redirect) |
| aud_092 | moira | Kayip esya ofisi nerede? | K-OB S-I-R-F-I-Z-N-E-D | insertion | kept (abstain) |
| aud_110 | rishi | I parked in P1 - how do I get back there? | i part in p 1 how do i get back there. | substitution | kept (now via the `p 1` rule) |
| aud_112 | rishi | Is my flight NH123 delayed? | is my Flight NH-123 delayed. | insertion | kept (redirect) |
| aud_113 | rishi | What gate is flight XY456 leaving from? | what gate is fly it x-y-456 leaving from. | insertion | kept (redirect) |
| aud_116 | rishi | Kayip esya ofisi nerede? | K.S.I.R.O.F.A.C.N.A.R.E.D. | insertion | kept (abstain) |

Held-out split, 5 changed transcripts of 20:

| Clip | Voice | Reference | Whisper | Type | Outcome |
|---|---|---|---|---|---|
| aud_069 | karen | Gate A11 please | KHA 11, please. | identifier | LOST: answer → abstain |
| aud_093 | moira | Gate A11 please | 8-8-11 please. | identifier | LOST: answer → clarify |
| aud_118 | rishi | I have desk 225 on my ticket which terminal is that? | I have this 225 on my ticket which terminal is that? | identifier | LOST: answer → clarify |
| aud_119 | rishi | Which belts serve terminal 2 arrivals? | which belts of Terminal 2 arrivals. | substitution | kept |
| aud_120 | rishi | Is flight LH 2004 boarding yet? | Is fly at LH2000 and for boarding yet? | insertion | kept (redirect) |

What the catalogue shows:

1. **Short identifier phrases are the weak point, and the failures are not
   spelling variants.** "belt 8" became "bell date", "Bill Hague" and
   "Beltaid" for three of five voices; "Gate A11" became "KHA 11" and
   "8-8-11"; "desk 225" became "this 225". None of these is a homophone a
   rule could safely map (a rule turning "Bill Hague" into "belt 8" would be
   exactly the per-sentence patching this project rules out). This is a
   language-model prior in Whisper-base preferring common words over an
   isolated noun plus digit, and it is the concrete reason a larger Whisper
   or a domain prompt might be worth one comparison later.
2. **Every identifier loss failed safely.** All six became abstain or
   clarify, never an answer for a different record, because the
   deterministic stage only answers on an identifier it actually finds and
   the semantic stage has no strong candidate for "where is Bill Hague".
3. **Insertions are mostly harmless.** Whisper writes "NH-123", "XY 456",
   "x-y-456"; L1 strips the hyphen, the flight-reference pattern still
   matches after "flight", and the redirect fires. The WER counts these as
   errors; the outcome does not change. This is why the propagation
   breakdown is reported alongside WER rather than instead of it.
4. **One wrong-confident answer, and it is the important one.** aud_044:
   the Turkish utterance transcribed as "K-A-BSI Office-Enerid?". The
   normaliser leaves "office", which is a category cue for the single
   record `lost_property_t1`; a single-candidate similarity above `tau_high`
   answers. Typed, the same Turkish text abstains. The defect is not in
   speech: it is the single-candidate threshold question already carried
   forward to Chat 04 from the text held-out analysis, now with a second
   independent trigger (garbage transcripts). It is left in place and
   recorded, not patched here.
5. **The held-out audio subset is easier than the held-out text set.** Its
   four utterances are all answered correctly when typed (20/20), whereas
   the text held-out set as a whole is 24/36. The subset was chosen to
   exercise identifiers and a flight reference, not to sample difficulty;
   the speech held-out gap (0.15) therefore measures transcription only, and
   should not be compared with the text-side dev/held-out gap.

## 7. Changes made from the evidence, and proof the text results did not move

Three changes, each tied to a clip id above:

- `audio_min_seconds` 1.0 → 0.5. At 1.0 s the first run rejected six
  clips unheard, all 0.87 to 0.99 s long: "gate c3?" from four voices,
  Samantha's "lost and found" and her "Where is belt 8?". A two- or
  three-word query is under a second; the bound was a guess and the data
  corrected it. After the change: 0 rejections; five of the six reach the
  right outcome and the sixth is the "bell date" identifier loss in §6.
- Rule 20 `car_park_letter_split` (`p N` → `pN`, single digit, provenance
  observed, evidence aud_110). Restricted to one digit because the car
  parks are P1 style; "the p 12 bus" is untouched, and the test says so.
- Rule 21 `homophone_desk` (`disk NNN` → `desk NNN`, three digits,
  provenance observed, evidence aud_053 in the first run). "a disk of 145
  mm" is untouched.

Both rules were added under the rule the project set itself: a normaliser
rule may only be added for an observed error class, must be printable in
the rule table with its provenance, and must not change the typed results.
The proof of the last point is direct: `run_text_pipeline_seed.py` was run
on dev and held-out after the change and every output file
(`cascade_results.csv`, `cascade_summary.json`, `intent_results.csv`,
`intent_report.json`, `threshold_grid.csv`, `cue_experiment.csv`,
`baseline_tfidf_lr.json`, on both sets) is byte-identical to the 03.2
closure outputs. Dev remains 35/43 and held-out 24/36.

Two other rule-table facts from this run: the six anticipated homophone
rules (16 to 19 plus the glued forms) did not fire on any of the 120
transcripts. The docstring's rule is that anticipated rules are relabelled
or removed once real transcripts exist; synthetic voices are not real
passengers, so the decision is deferred to the human recordings. If they do
not fire there either they are removed in Chat 04, with the rule table
regenerated.

The speech runner was also changed to warm Whisper up before timing
(one dummy transcription), so the per-clip latency now measures
transcription; the first run had reported a 25 s "maximum" that was the
model load.

## 8. Environment benchmark

From `outputs/env_benchmark.csv` (loaders in `scripts/benchmark_env.py`,
warm-up excluded from the runs, cold load reported separately). Workspace
rows first, MacBook rows (arm64, `device_available` mps, `device_used` cpu)
where measured:

| Row | Cold load | Warm mean | RSS |
|---|---|---|---|
| clip (load), workspace | 4.7 s | 0.35 s | 851 MB |
| clip (load), MacBook | 2.8 s | 0.13 s | 517 MB |
| clip_encode (one image), workspace | | 0.31 s | 1341 MB |
| whisper (load), workspace | 4.1 s | 0.16 s | 815 MB |
| whisper (load), MacBook | 2.8 s | 0.10 s | 493 MB |
| whisper_transcribe, 2 s tone, before the token cap | 36.3 s (cold) | 31.1 s | 1122 MB |
| whisper_transcribe, 2 s tone, after the cap | 9.1 s (cold) | 4.8 s | 1121 MB |
| speech runner, real clips, warmed up, workspace / MacBook | | median 1.46 s / 0.34 s | |

The "before / after the cap" pair is the measurement behind
`MAX_NEW_TOKENS = 64` (§9). Three models resident together (MiniLM, CLIP,
Whisper) will be around 2 GB of RSS on this basis; the MacBook figures are
the ones that go into the report.

## 9. Defects found and corrected in this checkpoint

- **Whisper on non-speech ran for 31 s.** A pure tone that passes the
  loudness gate made Whisper-base emit punctuation until its 448-token
  limit. Fix: `max_new_tokens = 64`, which bounds it to a few seconds and
  is far above any passenger query. Measured before and after (§8).
- **Whisper on white noise hallucinated speech** ("I'm going to put it in
  the oven"); the cascade abstained at 0.09 similarity, so no wrong answer
  reached the user, but it did reach the cascade. Fix in two parts: a
  letter-free transcript ("……") is rejected as "no speech recognised", and
  the loudness gate stays as the first line. Hallucinated real words on
  noise are not caught by this and are noted as a limitation.
- **Transparent pictograms loaded as all-black images** (§2b): the first
  vision run scored 0.42 with one shared embedding for dozens of files;
  fixed by compositing on white, with a test.
- **Vision band decided on twin records** (§2b): an image cannot choose a
  terminal; the band now comes from the category ranking and the records of
  that category are returned as candidates.
- **Gate minimum too strict** (§7).
- **First-clip latency reported as the model load** (§7).
- **Benchmark `device` column** said `mps` on the MacBook while every loader
  ran on CPU; split into `device_available` and `device_used` in 03.2's
  last commit, carried here.

## 10. Tests

`python -m pytest tests` — **138 passed** in this workspace with all three
models present (the model-dependent tests skip cleanly without them).
Added or changed in this checkpoint: `test_vision.py` (10: image checks and
refusals, index construction from prompts and KB, ranking and margin on
synthetic vectors, anchors marking an image out of scope, bands only when
thresholds exist, CLIP embedding shape and normalisation),
`test_vision.py` also gained the transparency test and the category-band
test (11 in total); `test_speech.py` (9: duration and loudness gates including the 0.9 s case,
stereo 44.1 kHz load and resample, unreadable file, the shared normaliser
on transcripts, hand-off reaching the exact-identifier stage with a stand-in
transcriber, rejected clip never calls the transcriber, letter-free
transcript rejected, Whisper on a tone bounded and non-speech),
`test_normalizer.py` (four new L2 cases for the observed rules and their
negatives), `test_scripts_compile.py` (every script compiles and imports).

## 11. Limitations to carry into the report

- Synthetic voices are clean read speech; the accent spread is five voices
  of one vendor, not passengers. The human evidence is five clips from one
  speaker; it agrees with the synthetic failure class but cannot give a
  rate. More speakers (the consented recordings planned for the user study)
  are needed before any WER for human speech is quoted.
- WER is computed against the written query; a spoken "P1" has no single
  correct spelling, so a fraction of the WER is orthographic rather than
  acoustic. The identifier-token accuracy and the propagation breakdown are
  the measures that speak to the system's behaviour.
- The Turkish clip is one utterance; the "non-English input" finding is a
  single case, not a rate.
- The image set is small (19 dev, 21 held-out in-scope), two categories
  have no image, and the photographs are of a screen rather than of
  terminal signage; the numbers in §2b are first measurements with wide
  uncertainty, and the photo stratum shares symbol families with dev.
- Out-of-scope handling for pictograms rests on the margin rule alone; the
  anchors as written do not fire on them.
- Whisper-small, a domain prompt, or beam search were not compared.

## 12. Recommendation for the next phase

Chat 04 (fusion and the remaining text issues) should take from here: the
single-candidate threshold now has two independent triggers (h019-style
text and garbage transcripts); the speech result object already carries the
gate outcome, raw and normalised transcripts and identifiers, so the router
can decide on transcript quality (identifier found, letters ratio, gate
flags) without re-transcribing; the vision result carries category and
record rankings, out-of-scope, band and quality flags, which is everything
the image + text disagreement analysis needs; the record ranking is now
restricted to the top category, so fusion's job on the image side is
terminal disambiguation and category confirmation, not record search.
Nothing in either module should need to change for fusion beyond being
called.

# PROJECT STATE — CHECKPOINT 03.3

- Repository: `github.com/yasmiiinay/nvh-airport-assistant`, branch `main`,
  pushed from the MacBook through `059e102` (human recordings) plus the
  correction commit that follows. Every commit under the project owner's
  identity, short messages.
- Closed: 03.1 deterministic core; 03.2 text intelligence, semantic
  retrieval, response assembly (dev 35/43, held-out 24/36, frozen).
- 03.3 IMPLEMENTED: `src/vision.py`, `src/speech.py`, runners
  `scripts/run_vision_eval.py` and `scripts/run_speech_eval.py`, dataset
  helpers `build_image_manifest.py`, `make_tts_audio.py`,
  `add_recordings.py`; 138 tests pass.
- 03.3 VERIFIED (workspace and MacBook, identical counts): speech on 120
  synthetic clips. Dev WER 0.157 / 0.151 (L1 / L2), English-only 0.080 /
  0.073, identifier accuracy 0.90, propagation 95 → 91 (gap 0.04), one
  wrong-confident answer (aud_044). Held-out WER 0.092, propagation 20 → 17
  (gap 0.15), three identifier losses, all safe. MacBook latency median
  0.34 s per clip; Whisper-base and CLIP cold load 2.8 s each.
- 03.3 EXECUTED: five human clips (spk_01), 4 of 5 outcomes kept, the loss
  is the "belt + digit" class again; three spoken-only sentences declared in
  `data/text/queries_spoken.csv`. Too few for a rate.
- 03.3 EXECUTED (workspace): vision on 101 labelled images. Dev top-1 0.947
  (18/19), held-out 0.857 (18/21), top-3 0.905; photo crops 13/14. Out of
  scope: ordinary photographs 7/7 caught by anchors; pictograms 2/54 caught
  by anchors, 42/54 held to uncertain or no-match by the margin rule, 10/54
  given a wrong "strong match"; wrong-confident rate 0.08 dev, 0.125
  held-out. Thresholds 0.30 / 0.24 / 0.015 set on dev; no prompt
  revision. `security` and `accessibility`: BLOCKED BY DATA (no image).
- 03.3 NOT YET VERIFIED on the MacBook: the two vision runs (expected to
  reproduce exactly; greedy, deterministic).
- Text pipeline: untouched in behaviour; two observed normaliser rules
  added with byte-identical text outputs; gate minimum 0.5 s.
- Carried to Chat 04: pictogram-style out-of-scope anchors written from the
  vocabulary before looking at images, or a fusion rule that an image-only
  strong match is confirmed by the passenger; the check-in/baggage suitcase
  ambiguity as a text-overrides-image scenario; blur threshold on real
  signage; single-candidate threshold (text h019 + speech aud_044); h024 cross-terminal grounded negative; out-of-scope near
  `tau_low`; paraphrase recall; twin-record margins; anticipated-rule
  removal decision after more human recordings; optional Whisper-small
  comparison; check of the −45 dBFS floor against more human clips;
  removal of the unused `write_artifacts` stubs in `evaluation/`.
- Not started, by design: fusion and routing, Gradio UI, Space deployment.
