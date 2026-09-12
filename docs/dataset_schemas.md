# Dataset Schemas v1 (frozen 02C)

Schemas only; population happens in Chats 03-04. Header-only manifest CSVs sit
in each data folder so collection starts by filling rows, not inventing columns.
Controlled values (category, intent, query_type, split, decision outcomes) come
from `data/vocabulary.json` and are validated by `scripts/audit_foundation.py`.

## 1. Image dataset — data/images/images_manifest.csv

| Field | Meaning / allowed values |
|---|---|
| image_id | `img_###`, unique |
| category | controlled vocabulary category (visual classes only) |
| source | `aiga_dot` \| `own_photo` \| `commons` |
| licence | e.g. `copyright-free (AIGA)`, `own work`, `CC BY 4.0`, `CC0` — CC BY-SA avoided (Evidence Pack A1) |
| licence_url | link to the licence/file page; blank only for own photos |
| attribution | attribution string even when not legally required |
| modified | `yes`/`no` (crop/resize counts as yes) |
| source_type | quality stratum origin: `clean_icon` \| `photo_signage` |
| quality_stratum | `clean` \| `real_good_light` \| `real_degraded` (RQ1 strata) |
| split | `dev` \| `heldout` |
| expected_label | equals `category`; kept explicit so mislabels are auditable |
| optional_identifier | identifier visible in the image (e.g. `B12`) or blank — feeds the OCR enhancement and conflict scenarios |
| notes | anything a reviewer needs |

Collection rules (from Evidence Pack A1): no identifiable people, no third-party
logos, no boarding passes/documents; per-file licence check for Commons items.

## 2. Text dataset — data/text/queries_seed.csv (already populated: 43 rows)

query_id, query, intent, entities_json (JSON object as string), target_kb_id
(blank allowed only when expected_behaviour ≠ answer), query_type, expected_behaviour
(decision outcome), split, notes.

## 3. Audio dataset — data/audio/audio_manifest.csv

| Field | Meaning |
|---|---|
| audio_id | `aud_###`, unique |
| query_id | the text query this utterance realises (WER reference = that query's reference_transcript) |
| speaker_id | pseudonymous code `spk_01`… — **never names** (GDPR minimisation; consent forms stored outside the repo) |
| reference_transcript | exact words spoken, human-verified |
| environment | `quiet` \| `cafe_noise` \| `announcement_noise` (playback-added noise, documented) |
| noise_condition | `clean` \| `moderate` \| `heavy` |
| expected_identifiers | canonical identifiers the transcript contains, `;`-separated, or blank |
| target_kb_id | copied from the query row for propagation-gap runs |
| split | `dev` \| `heldout` |

Audio files themselves are consented recordings; raw audio is never logged by
the running system (v1.1 privacy posture) — the dataset copies live only in the
local data folder, not in any interaction log.

## 4. Multimodal dataset — data/multimodal/multimodal_manifest.csv

| Field | Meaning |
|---|---|
| sample_id | `mm_###`, unique |
| image_id | from images manifest, or blank |
| query_id | text query id, or blank |
| audio_id | audio id, or blank (exactly one of query_id/audio_id set when text/voice present) |
| modality_condition | one of the six conditions in `evaluation/multimodal_metrics.CONDITIONS` |
| intended_intent | controlled intent |
| intended_entities | JSON object as string |
| target_kb_id | expected record, blank for abstain/clarify/redirect cases |
| consistency_label | `consistent` \| `conflict` \| `single_modality` |
| conflict_type | blank, or `identifier_mismatch` (photo B12 + text B21), `category_mismatch` (baggage photo + lounge question) |
| query_type | controlled query type |
| split | `dev` \| `heldout` |
