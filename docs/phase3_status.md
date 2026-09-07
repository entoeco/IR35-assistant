# Phase 3 — status, and an environment blocker

**Built and tested: all of it. Run against real model weights: none of it.**

This document exists so that the gap is recorded rather than discovered later.

---

## What is blocked, and why

The development sandbox this project is being built in enforces an outbound
network policy. Package registries are permitted; model hosting is not.

| Host | Needed for | Result |
|---|---|---|
| `pypi.org`, `files.pythonhosted.org` | `torch`, `transformers` | reachable — both installed |
| `huggingface.co`, `cdn-lfs.huggingface.co` | model weights | **403 at the egress proxy (organisation policy)** |
| `api.anthropic.com` | hosted few-shot arm | reachable, but no API key is present in the environment |

So the cross-encoder NLI weights cannot be downloaded, and the hosted few-shot
path has no credential. Both arms of the Phase 3 comparison are blocked on the
environment, not on the code.

The proxy documentation is explicit that a 403 is an organisation policy denial
and should be reported rather than worked around, so no attempt was made to
route around it.

**Nothing has been fabricated to fill the gap.** `scripts/run_phase3.py` refuses
to run when the selected backend is unusable, and when run deliberately against
the mock backend it labels its own output as a verification run and prints that
warning at the top of the report. A plumbing check presented as a finding would
be worse than no finding.

---

## What was built and is fully tested

| Component | File | Tested by |
|---|---|---|
| Swappable backend interface (constraint 2) | `src/models/backend.py` | 11 tests |
| Config-selected backends: mock, local transformers, local LLM, hosted API | `config/model.yaml` | 11 tests |
| Cross-encoder NLI detector | `src/models/nli.py` | 11 tests |
| Few-shot LLM detector | `src/models/fewshot.py` | 12 tests |
| Model fetch script | `scripts/fetch_models.py` | run in `--check` mode |
| Phase 3 evaluation runner | `scripts/run_phase3.py` | run end to end on the mock backend |

40 tests, all passing. They cover the things that would otherwise only surface
after a long inference run: premise construction, chunking rather than
truncation, bidirectional aggregation, the three-class-to-one-score mapping,
JSON parse robustness and retry, parse failures being recorded rather than
imputed, and that no detector's explanation asserts a determination.

The verification run also confirms the structural property the phase turns on:
**both new methods score the `record` and `unit` splits identically**, because
neither fits anything. That is the same property that kept the Phase 2 rule
baseline stable while TF-IDF collapsed — and it means that when the real run
happens, any difference between these methods and the rule baseline on the
`unit` split is a genuine difference in generalisation.

One real bug was caught by these tests: the mock backend was matching trigger
phrases against the whole few-shot prompt, including the worked examples, one of
which legitimately contains the word "reimbursed". Every input was therefore
scored as a contradiction. The same class of error against a real model —
matching on the wrong span of a prompt — would have been much harder to see.

---

## How to produce the real results

On any machine that can reach `huggingface.co`:

```bash
pip install -r requirements.txt          # see the note on torch below
python scripts/fetch_models.py           # ~1.5 GB into models/weights/
python scripts/run_phase3.py             # backend from config/model.yaml
```

`local_files_only: true` is set in `config/model.yaml`, so once the weights are
cached the pipeline never touches the network again — which is the deployment
posture the project assumes, not merely a convenience.

For the hosted few-shot arm, set `ANTHROPIC_API_KEY` and run
`python scripts/run_phase3.py --backend hosted_api`. Note that this sends
justification text to a third party; on real submissions that is an Information
Governance decision, and `require_local: true` in `config/model.yaml` exists to
make the local-only posture enforceable rather than aspirational.

**Note on `torch`.** `requirements.txt` pins the version resolved in this
sandbox, which is a CUDA build. On a laptop, install the CPU wheel instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The CUDA build works on CPU but is several gigabytes larger, and the deployment
target is a laptop.

---

## What the real run is expected to show, and how it could be wrong

Stated in advance so that the write-up records a prediction rather than a
rationalisation after the fact.

**Expected.** Zero-shot NLI generalises to unseen phrasing where TF-IDF did not,
because it was trained on the entailment relation rather than on this corpus's
vocabulary. Concretely: a `unit`-split PR-AUC materially above the 0.007
no-skill floor, and better recall than the rule baseline on
`hedged_non_support` (0.20) and `named_person_dependency` (0.40).

**Ways that could fail, all of which are reportable findings rather than
problems to engineer away:**

1. **The domain is too far from MNLI.** NLI models are trained on short everyday
   sentences. "In answer to 'Do you have the right to reject a substitute?', the
   engaging manager stated: No" is a strange premise, and administrative English
   about procurement is not the training distribution. The premise template is
   in config precisely so its wording can be treated as an experiment, but if
   the gap is large enough, no wording fixes it.
2. **The neutral class may be too noisy to use.** The `non_support_weight`
   parameter assumes neutral is informative for `hedged_non_support`. If neutral
   is simply where the model puts everything it finds unfamiliar, that signal
   will add false positives rather than recall. Setting the weight to zero
   measures exactly this.
3. **Bidirectional scoring may cost more precision than it buys in recall.**
   Taking the maximum over both directions is a recall-oriented choice; the
   `aggregate` setting makes the alternative measurable.
4. **The few-shot arm may be impractical locally.** A 1.5B model on CPU across
   ~10,000 instances is a very different runtime proposition from a
   cross-encoder, and if it is impractical, that is the answer to the deployment
   question even if its accuracy is good.

---

## For the Information Governance conversation

Two points from this phase belong in the DPIA discussion:

- **The hosted path is a third-country transfer.** It is implemented for the
  method comparison only, guarded by `require_local`, and refuses prompts that
  still contain apparent identifiers as a backstop. The production assumption
  remains local-only.
- **The few-shot prompt is designed so a determination cannot be produced.** The
  model sees one question, one selected option and one justification. It is not
  shown the other answers, so it does not have the information an employment
  status determination would require — the Art.22 posture is structural rather
  than a matter of instructing the model not to.
