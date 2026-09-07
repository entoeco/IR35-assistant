# Phase 6 — Handover

The Phase 6 gate, from the original plan, is blunt on purpose: **clean
clone, install, run, offline, on a machine that has never seen this
project.** This phase is written to that gate, not to a checklist of
documents — the documents exist because the gate demands them, and the one
real bug this phase found was discovered by actually trying to clear the
gate rather than by writing about it.

## The gate found a real problem, and it's fixed

Before writing a word of README, the honest thing to do was try the gate as
committed at the end of Phase 5: `git clone`, fresh virtual environment,
`pip install -r requirements.txt`. It would have failed. `requirements.txt`
pinned `torch==2.14.0+cu130` — a CUDA-specific build that only resolves
against the build sandbox's own package index. On an ordinary laptop with no
GPU, `pip install -r requirements.txt` has no wheel to install and the
command simply errors out, before a single test can run.

This was not a hidden problem — the file already carried a comment telling a
laptop user to install the CPU wheel by hand — but a comment is not a fix,
and "the first command in the README fails unless you already know to skip
a line and read a comment first" is not a passing gate.

**Fixed by splitting the file.** `requirements.txt` now contains only what
is needed to run the tests, the corpus generator, the Phase 2/4 pipelines and
the Phase 5 app — nothing that needs a GPU or a non-default index. `torch`
and `transformers`, needed only for Phase 3's real local-transformer and
local-LLM backends, moved to `requirements-transformers.txt`, pointed at
PyTorch's own CPU-wheel index instead of a CUDA build.

**This was checked, not assumed to be safe.** `src/models/backend.py`
imports `torch` and `transformers` lazily, inside the specific methods that
need them, rather than at module load time — so nothing else in the project
should need them at all. To confirm rather than trust that, both packages
were uninstalled from the working environment and the full suite was run
again:

```
$ pip uninstall -y torch transformers
$ pytest -q
........................................................................ [ 31%]
........................................................................ [ 62%]
........................................................................ [ 93%]
................                                                         [100%]
```

All 232 tests passed with neither package present. That is the actual claim
this phase makes about the split, not an inference from reading the import
statements.

**Then the real gate was run**, not simulated: a bare virtual environment —
nothing else on the machine, no leftover packages from this build sandbox's
much larger toolchain — with only `requirements.txt` installed:

```
$ python3 -m venv /tmp/lockenv
$ /tmp/lockenv/bin/pip install -r requirements.txt
$ /tmp/lockenv/bin/python -m pytest -q
........................................................................ [ 31%]
........................................................................ [ 62%]
........................................................................ [ 93%]
................                                                         [100%]
```

232 tests, clean environment, no network access beyond the initial `pip
install`, no GPU. That is the clean-clone-install-run gate, actually cleared,
not asserted.

**One thing this phase could not verify**, and says so rather than papering
over: `requirements-transformers.txt`'s exact pin against PyTorch's CPU wheel
index could not be checked from this sandbox either —
`download.pytorch.org` returned the same 403-at-the-proxy organisational
policy denial that blocked `huggingface.co` in Phase 3
(`docs/phase3_status.md`). The file says so directly and gives the fallback
command to find a working pin. This is the same posture Phase 3 took with
the transformer weights: report the gap precisely rather than guess at a
number that looks plausible.

## The lockfile

`requirements.txt` pins direct dependencies with a rationale comment next to
each — good for a human reading the file, insufficient on its own for
byte-for-byte reproduction, because it says nothing about what those
packages themselves depend on (Streamlit alone pulls in pandas, click,
Jinja2, pyarrow and more). `requirements-lock.txt` pins all of that too, at
the exact versions resolved into the same clean virtual environment used for
the gate check above.

**Rejected: a raw `pip freeze` from the working development environment.**
This sandbox's Python environment has a great deal installed that has
nothing to do with this project — document conversion tools, image
libraries, a whole separate agent SDK. A freeze of that environment would
have produced a "lockfile" bloated with over a hundred unrelated packages,
misrepresenting what this project actually depends on and making the real
dependency list harder to see, not easier. The lockfile was generated from a
bare virtual environment instead, specifically to avoid that.

**Rejected: locking `requirements-transformers.txt` too.** Torch's build is
inherently machine-specific — CPU, or one of several CUDA versions — so a
single central lock would either be wrong for most machines or would force
one specific build on everyone. The file says so and leaves that pinning to
whoever installs it, on their own machine.

## The handover documents

Three documents, each for a different reader, because a single "handover
doc" trying to serve a developer setting up the repo, a DPO deciding whether
a pilot is safe, and a reviewer using the finished screen tends to serve none
of them well:

- **`README.md`** — a developer or marker cloning the repo. Install, run,
  what's where, and where to find the reasoning behind each phase.
- **`docs/stakeholder_runbook.md`** — the IR35 team, using the finished
  screen. No mention of PR-AUC, calibration, or splits — what a flag means,
  what to do with one, what the tool gets wrong and why that's expected, and
  who to contact for what kind of problem. Two things carried over
  deliberately from the technical work, translated rather than dropped: the
  Article 22 framing (as "it will never tell you the answer, and if it looks
  like it's trying to, that's a bug") and the calibration finding from
  Phase 4 (as "a percentage would have been actively misleading, so we don't
  show one").
- **`docs/deployment_notes.md`** — whoever signs off moving from synthetic
  data to a real pilot. This is the document written most carefully, because
  it is the one place in this handover with genuine teeth: a DPIA
  requirement, a de-identifier validation step that has not happened and
  cannot happen without real text, a config flag (`require_local`) that
  defaults to the wrong value for production and must be changed
  deliberately, and a decision-log design explicitly built as a synthetic-data
  placeholder rather than a production proposal. Each item points at the
  specific file and test that makes the underlying claim checkable, rather
  than asking for trust.

**Rejected: one combined "handover pack."** A DPO reading a document written
partly in the voice of "click Accept to record your decision" would not take
the DPIA section seriously, and a reviewer handed a document opening with
Article 35 obligations would stop reading before reaching the part that
tells them how to use the screen. Splitting by audience cost three files
instead of one; it was worth it.

## What this phase did not do, and why

**Did not build a CI pipeline.** The gate this phase targets —
clean-clone-install-run — was verified by hand, with the commands and output
recorded above rather than automated. A GitHub Actions workflow running
`pytest` on every push would be a reasonable next step for a project that
continues past this handover, but the brief's Phase 6 gate is about proving
the install and run steps work once, correctly, not about continuous
verification — and standing up CI infrastructure for a project handed over
at this point would be effort spent on a capability nobody asked for yet.

**Did not attempt the real Phase 3 run.** Still blocked by the same
organisational egress policy documented in `docs/phase3_status.md`, now
joined by `download.pytorch.org` for the same reason. Nothing changed here
that would unblock it; `docs/phase3_status.md`'s "how to produce the real
results" section remains the accurate runbook for whoever has access to run
it.

**Did not change any detection logic, threshold, or config value.** Phase 6
is a handover phase. Touching `config/ir35_weights.yaml` or
`config/rule_baseline.yaml` here, however tempting after reading the
deployment notes, would be scope creep against a phase whose job is to make
the existing, already-evaluated system installable and explainable to three
different audiences — not to re-open evaluation questions Phase 2 and Phase 4
already answered and documented.

## Where this leaves the project

232 tests pass, from a clean install, offline, verified in a bare virtual
environment rather than assumed. Every phase's reasoning, including its
rejected alternatives and the bugs its own rigour caught, is written down
where the next person to touch this — a marker, a future maintainer, the IR35
team's technical contact — can find it. The one thing standing between this
and a real pilot is not more code: it is the DPIA, the de-identifier
validation run, and the `require_local` flag flip listed in
`docs/deployment_notes.md`, none of which a developer can complete alone.
