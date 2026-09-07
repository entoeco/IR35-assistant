"""Phase 5 — reviewer-facing assessment and decision capture.

This package sits downstream of everything built in Phases 1-4 and adds
nothing to the detection methods themselves. It answers three questions the
detectors do not:

* How is a raw score shown to a person who is not a data scientist?
  (:mod:`src.review.bands`)
* How is one submission run through the full pipeline —
  de-identify, validate, score, explain — in one call?
  (:mod:`src.review.assess`)
* What happens to a reviewer's accept/dismiss decision?
  (:mod:`src.review.decision_log`)

Nothing here computes or stores an employment-status determination. That is
by design — see ``docs/phase_plan.md`` Phase 5 and
``docs/reports/phase5_interface_report.md``.
"""

from __future__ import annotations
