# Phase 9 — Reviewer App Revamp

The request: *"lets revamp the streamlit app. I want it to be simpler, yet
clearly show what questions are impacting the check the most. Also, make it
more easily readable and abide by accessibility requirements/compliance."*
Two scope decisions were made up front, in discussion with the project
owner: the "what's impacting it most" ask becomes a new "Top things to look
at first" panel that links down to full detail, rather than a second
dashboard; and "simpler" means tightening the layout — collapsing sections
that usually have nothing to say — without cutting any check or any piece
of information the app already surfaces.

## What changed

**Sections 2 and 3 merged into one "Review summary."** The old app showed
the review-priority banner and the privacy/completeness checks as two
separate numbered sections before the flags themselves. They are now one
section: the priority banner, the "what goes into this, and what doesn't"
explainer, the new top-drivers panel, then the privacy/completeness checks.
Nothing in the banner's logic or wording changed — this is a layout
regrouping, not a rescoring.

**"Top things to look at first" ranks the submission's existing flags and
findings, it does not invent a second metric.** `src/review/review_priority.py`
already computes a weighted score per flag (confidence-band weight ×
materiality weight, with a gate override) and per cross-field finding
(severity weight × materiality weight, same override) on its way to the
five-point review-priority aggregate. That per-item arithmetic was factored
out into `_flag_impact`/`_cross_field_impact` and exposed through a new
`rank_by_impact()` function, so the panel orders items by exactly the
numbers the banner above it is built from — never a second, divergent
notion of "importance," and never anything that reads which way an answer
leans. `config/review.yaml`'s new `review_priority.top_drivers_shown: 5`
caps how many items the panel lists, the same way every other UI-tunable
number in that file lives in config rather than as a literal in
`src/ui/app.py`.

Each item in the panel is a link — `<a href="#item-id">` — to a `<div
id="item-id">` placed immediately before that flag or finding's full card
further down the page. The anchor id is built from the same deterministic
key (`_flag_key`/`_cf_key`) already used for that item's Streamlit widget
key and decision-log lookup, so the summary panel and the detail section
can never drift onto different identifiers for the same item. When nothing
is flagged, the panel does not render at all — there is nothing to rank —
and the summary says so in one line pointing at the full checks below.

**Privacy and completeness checks collapse by default, but only when
genuinely clean.** The expander's label is now computed from the actual
result: `"Privacy and completeness checks — clean, nothing to review"` when
both the de-identifier and the validator found nothing, collapsed by
default in that case; otherwise the label states the specific counts (e.g.
*"2 personal-data item(s) removed, 3 completeness finding(s)"*) and the
expander opens automatically. A reviewer's default scroll-through is
shorter on a quiet submission without ever hiding a real issue behind an
extra click — the content inside the expander is completely unchanged.

**Sections renumbered, nothing else moved.** What was 2/3/4/5/6 is now
2 ("Review summary," merged)/3 ("Flags for review")/4 (cross-field
checks)/5 ("Decisions recorded"). Every check that ran before still runs;
every card that showed before still shows.

## Accessibility

This phase extends, rather than replaces, the WCAG 2.2 AA work already
documented in `docs/reports/phase5_interface_report.md` — contrast ratios,
focus-visible outlines, target sizes, and the colour-is-never-the-only-signal
rule all carry over unchanged. Two things specific to the new panel needed
their own check.

**The impact badge is real information, not decoration.** Every other icon
on this page (the ●/▲/■ markers on flag cards) is `aria-hidden="true"`
because the adjacent text already says the same thing in words. The new
`.impact-badge` pill is different: it is the only place in the link's
accessible name that states the band or severity label at all, so marking
it `aria-hidden` would silently drop that information for a screen-reader
user, even though a sighted user sees it fine. It is therefore **not**
`aria-hidden`, and the link's visible text was correspondingly written so
the badge's word is not also repeated in the following span — both to
avoid a sighted user hearing/reading "Priority" twice and because a screen
reader concatenates the badge and the text into a single accessible name
("PriorityQuestion 4.14 — Based upon this engagement, does the...").
Verified directly: a Playwright pass that tabs through every focusable
element on the page confirmed the link receives the standard 3px
focus-visible outline in sequence with everything else, and `innerText`
read back from the live DOM matches the expected concatenated accessible
name with no duplication.

**Native same-page navigation, no JavaScript.** The panel's links are plain
`<a href="#...">` anchors to plain `<div id="...">` targets — both keyboard-
and screen-reader-navigable the same way any in-page anchor link is,
without a custom router or scroll-handler that would need its own
accessibility testing. Verified via Playwright: clicking a panel link
lands the viewport on the correct detail card.

**Existing contrast and sizing choices were reused, not re-derived.** The
`.impact-item` link reuses the page's already-audited foreground colour
(`#2c2c2c` on white, ≈13:1) and the badge's border/text use `currentColor`,
so no new colour was introduced that would need its own contrast check.

Tests: `tests/test_review_priority.py` gained seven tests for
`rank_by_impact` (ordering by impact score, the gate override sorting
first regardless of score, merging flags and cross-field findings into one
order, respecting `limit`, behaving on empty input, each item identifying
only its own kind, and the same "takes no raw record" structural guarantee
`review_priority_for` already carries). `tests/test_ui_app.py` gained three
tests: the panel renders and links out for the default sample, the panel is
absent when nothing is flagged, and the privacy/completeness expander's
label reflects a genuinely clean record (using `SYN-0001` from the
committed synthetic corpus, confirmed directly against the validator, the
de-identifier, every detector, and every cross-field check before being
used as a fixture — a wholly empty record does not work for this, since an
empty record still trips 31 routing-completeness warnings even though it
is flag-free). Full suite: 319 tests, all passing.

## What this did not touch

`src/review/materiality.py`, `src/review/assess.py`'s pipeline wiring, the
`review_priority_for` aggregate's scoring itself, and every existing flag
and cross-field card's content are all unchanged. The revamp is a
presentation layer over numbers the app already computed and already
showed — nothing about what counts as a flag, how severely, or why,
changed in this phase.
