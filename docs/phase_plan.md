# Phases 1–6 — Plan

Written at the end of Phase 0. Each phase ends with passing tests and a commit.
Nothing starts until the previous phase's gate is green.

---

## Phase 1 — Synthetic data generation

**Output:** `data/synthetic/esq_synthetic_v1.jsonl` (~350 records),
`data/synthetic/ground_truth.jsonl`, `config/generation.yaml`, `src/ingest/`.

**Approach.** A seeded generator driven entirely by `schema.yaml`. Each record
is built in four passes:

1. **Engagement archetype.** Sample from four types — IT contractor, visiting
   lecturer, technical/trades contractor, research consultant — each with a
   prior over the structured answers reflecting how that kind of engagement
   actually looks. This is what stops the dataset being 350 draws from the same
   distribution with different nouns.
2. **Structured answers**, sampled under the archetype prior and constrained to
   follow the form's routing.
3. **Label**, derived by the CEST-approximating rule engine (pending your answer
   to Phase 0 question (g)).
4. **Free text**, generated per rationale box from the structured answer, in one
   of three registers: verbose/HR-literate, terse/technical, hedged/vague.
   Register is sampled per *record*, not per field — a manager writes the whole
   form in one voice, and Phase 4's style-invariance test depends on that being
   a per-record property.

**Contradiction planting.** ~15% of records carry at least one planted
contradiction, drawn only from that pair's `plantable_types`. Ground truth
records: record id, pair id, contradiction type, subtlety level (1–3), and the
clean text that was replaced. Storing the clean version is what makes the
style-invariance test possible later — the same engagement can be re-rendered in
three registers from one underlying state.

**Ambiguous and "unable to determine" cases.** A deliberate slice where the
structured answers genuinely conflict across tests (strong substitution right,
strong control) and the rule engine abstains. These are labelled `undetermined`
and excluded from the status classifier's training set but kept for the
reviewer-facing evaluation — a tool that flags confidently on a case CEST itself
cannot resolve is worse than one that says so.

**Reproducibility.** `generation.yaml` holds the seed, archetype priors, register
mix, contradiction rate and subtlety distribution. Regenerating from that config
must reproduce the dataset byte-for-byte; a test asserts it.

**Design choice.** Template-and-slot generation with a large controlled
vocabulary, not LLM-generated prose. Rejected LLM generation for the bulk of the
corpus because the ground truth has to be exact — I need to know precisely which
span carries the contradiction and how subtle it is, and a generative model
gives me plausible text with uncertain labels. I'd rather have slightly less
natural text with labels I can defend to an assessor. Where template output is
too stiff, an LLM *paraphrase* pass through the backend interface (preserving the
planted contradiction, verified by re-check) is the fallback — but that's an
enhancement, not the foundation.

**Gate:** schema-validation tests pass on every generated record; ground truth
is internally consistent; regeneration from config is deterministic.

---

## Phase 2 — Baselines

**Output:** `src/models/baseline_rules.py`, `src/models/baseline_tfidf.py`.

1. **Rule/keyword baseline.** Per-pair cue lists from `contradiction_pairs.yaml`,
   with negation handling (a naive keyword match fires on "we do *not* require
   them on site"), regex for rates and dates, and a per-pair threshold. Deliberately
   *different logic* from the label-generating rule engine, and I'll say so in the
   write-up — otherwise Phase 2 grades the generator against itself.
2. **TF-IDF + logistic regression.** Features over the concatenated
   `[structured option] [SEP] [free text]`, plus character n-grams to survive the
   terse register, plus explicit interaction features between the selected option
   and free-text terms. Class weights tuned for recall. Per-pair models where
   there's enough data, backing off to a pooled model.

These are honest attempts because the assignment's method comparison is worthless
otherwise — and because I expect TF-IDF to do genuinely well on
`direct_negation` and `category_mismatch`. The interesting result is where it
fails, not that it fails.

**Gate:** both baselines produce calibrated scores on a held-out split; results
table committed.

---

## Phase 3 — Transformer approach

**Output:** `src/models/backend.py` (the interface), `src/models/nli.py`,
`src/models/fewshot.py`.

**Backend interface.** One protocol, `InferenceBackend`, with
`local_transformers`, `local_llm` and `hosted_api` implementations selected by
`config/model.yaml`. Constraint 2 in full: switching is a config edit. The
hosted path is for development convenience; the production path is assumed
local-only, and a test asserts the local path works with no network.

**NLI framing.** `premise_template` renders the structured answer into a
sentence; the free text is the hypothesis; the model returns
entailment/contradiction/neutral. Model choice: a cross-encoder NLI model
fine-tuned on MNLI-scale data (DeBERTa-v3 family or similar), because
cross-encoders attend across the pair and this task is entirely about the
relationship between two short texts, not about either in isolation. Rejected a
bi-encoder/embedding-similarity approach — a contradiction and an entailment
are both *topically* similar, so cosine similarity is close to uninformative
here. Rejected fine-tuning from scratch: 350 synthetic records is nowhere near
enough, and zero-shot NLI transfer is the honest use of the pretrained model.

Practical issue to resolve in this phase: the 512-token limit against the
duties narrative (Phase 0 question (f)).

**Few-shot LLM comparison** through the same backend, with a structured output
contract and the same per-pair prompt derived from `contradiction`. Compared on
accuracy, recall, latency, and — relevant for the local-only production
assumption — whether it runs at all on modest hardware.

**Gate:** interface tests pass with a mock backend; both approaches evaluated on
the same split as Phase 2.

---

## Phase 4 — Evaluation and bias testing

Macro-F1, per-contradiction-type precision/recall, ROC-AUC, calibration plot
(reliability diagram + Brier score). Then:

- **Style invariance.** Same underlying engagement rendered verbose / terse /
  hedged. Report flag-set agreement (Jaccard) and confidence drift. Any material
  drift is a finding to report, not a bug to hide — it means the model is reading
  register rather than substance.
- **Role-type disparity.** Per-archetype breakdown. My prior is that visiting
  lecturers will be hardest, because the genuinely borderline cases cluster there.
- **Subtlety curve.** Recall against planted subtlety 1–3, per method. This is
  where I expect the transformer to earn its place.
- **Recall-optimised thresholds**, justified in the docs: a missed contradiction
  reaches HMRC as a determination made without reasonable care; a false flag costs
  a reviewer thirty seconds. The asymmetry is roughly two orders of magnitude, and
  the threshold should reflect that. The precision floor is set by reviewer
  fatigue, not by a metric — if reviewers start dismissing without reading, the
  tool has failed regardless of its recall.

**Gate:** evaluation notebook runs end to end; results committed; limitations
written up honestly.

---

## Phase 5 — Interface

Streamlit. Paste or upload a submission → per-field flags with confidence and a
short human-readable explanation → reviewer accept/dismiss, captured with
reviewer id, timestamp, flag id and optional free-text reason.

**WCAG 2.2 AA specifics:** flags carry an icon and a text label, never colour
alone; full keyboard operation with a visible focus indicator; every control has
a programmatic label; 4.5:1 contrast minimum on text and 3:1 on UI components;
flags announced via an ARIA live region; target size ≥24×24px (2.2 addition);
no drag-only interactions.

The determination itself is never displayed as an output of the tool. The status
signal is framed as *"indicative — for reviewer context only"*, and the
accept/dismiss control is on the *flags*, not on the status. That framing is the
Art.22 posture made visible in the UI, and it should survive contact with a
stakeholder who would quite like the tool to just tell them the answer.

**Gate:** keyboard-only walkthrough; contrast checked; decision capture tested.

---

## Phase 6 — Handover

README with local run instructions; pinned `requirements.txt` (and a lockfile);
offline-first install notes including model weight caching; a two-page
stakeholder runbook for the IR35 team written in their language, not mine; and
deployment notes covering what changes for real data — the DPIA, the
local-backend requirement, de-identifier validation against real submissions,
retention for the reviewer decision log, and a monitoring plan for drift as the
form and HMRC guidance change.

**Gate:** clean clone → install → run, offline, on a machine that has never seen
the project.
