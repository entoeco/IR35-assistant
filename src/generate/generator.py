"""Synthetic ESQ generator.

Two-stage by design: ``build_state`` decides *what a record says*, and
``render`` decides *how it is written*. Nothing about the engagement changes
between the two, so the same state can be rendered verbose, terse and hedged
and the three renderings are guaranteed to be the same engagement.

That split is what makes Phase 4's style-invariance test a real test. Re-render
one state in three registers, run the detector on all three, and any difference
in the flags is attributable to surface form alone — because the propositions,
the structured answers and the planted contradictions are literally the same
objects.

Generation order for one record
-------------------------------
1. Archetype and register (both per record; a manager writes in one voice).
2. Slot fillers, fixed for the record so all 30 boxes describe one engagement.
3. Structured answers, sampled under the archetype's per-test polarity bias.
4. Ambiguity injection: for a slice of records, push two tests to opposite
   extremes so the rule engine lands in its abstention band.
5. Routing: walk the form's branches and blank the questions it skips.
6. Label, from the CEST-approximating rule engine over the reachable answers.
7. Content units: one supporting proposition per answered pair.
8. Contradiction planting, replacing a supporting unit with a contradicting one.
9. Anomalies — non-responsive and blank justifications — kept separate from
   contradictions so Phase 4 can report them apart.

Constraint 1: every identity value is drawn from the invented pools in
``config/generation.yaml``. No real person, supplier or rate appears anywhere.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import asdict, dataclass, field as dc_field
from typing import Any, Iterable, Mapping, Sequence

from src.ingest.schema_loader import FieldType, Schema
from src.models.cest_rules import CestRuleEngine, StatusResult
from src.utils.logging import StructuredLogger, get_logger

from .registers import RegisterRealiser, VocabContext

__all__ = [
    "PlantedContradiction",
    "Anomaly",
    "RecordState",
    "EsqGenerator",
    "walk_route",
]

_GENERIC_SUPPORTING = "This reflects the arrangement agreed for {deliverable}."


# =============================================================================
# Routing
# =============================================================================
def walk_route(
    schema: Schema, answers: Mapping[str, Any], section_prefix: str = "4."
) -> list[str]:
    """Walk the form's branching logic and return the questions actually asked.

    The ESQ is mostly linear with a branch cluster around substitution
    (4.3 → 4.4/4.5/4.6/4.7, with 4.7 routing *back* to 4.6) and two forward
    branches around exclusivity and IP. A blank answer off the active route is
    legitimate, so anything that checks completeness needs this distinction or
    it reports noise on every record.

    Args:
        schema: The loaded data contract.
        answers: Structured answers keyed by field id.
        section_prefix: ``form_ref`` prefix identifying the status questions.
            Passed from ``generation.yaml`` so a renumbered form is a config
            edit (constraint 5).

    Returns:
        Reachable structured field ids, in the order the form asks them.

    Note:
        Back edges are resolved by continuing from the field *after* an
        already-visited target, which is what a manager following the paper form
        would do. Without that the 4.7 → 4.6 edge is an infinite loop.
    """
    order = [
        f.id
        for f in schema.structured_answer_fields
        if str(f.form_ref or "").startswith(section_prefix)
    ]
    position = {fid: i for i, fid in enumerate(order)}
    reachable: list[str] = []
    seen: set[str] = set()
    cursor: str | None = order[0] if order else None

    while cursor is not None and len(reachable) <= len(order):
        reachable.append(cursor)
        seen.add(cursor)
        routing = schema[cursor].routing or {}
        target = routing.get(answers.get(cursor))
        nxt = target if isinstance(target, str) and target in position else None
        if nxt is None:
            index = position[cursor] + 1
            nxt = order[index] if index < len(order) else None

        # Skip forward past anything already asked. Needed because 4.7 routes
        # BACK to 4.6: on the route 4.3 -> 4.5 -> 4.7 -> 4.6 the back edge is
        # live and 4.6 must be asked, but on 4.3 -> 4.5 -> 4.6 -> 4.7 it points
        # at a question already answered. Advancing one step is not enough —
        # the field after 4.6 is 4.7 itself, which is also already seen — so
        # this walks forward until it finds an unasked question or runs out.
        while nxt is not None and nxt in seen:
            index = position[nxt] + 1
            nxt = order[index] if index < len(order) else None
        cursor = nxt
    return reachable


# =============================================================================
# State
# =============================================================================
@dataclass(frozen=True)
class PlantedContradiction:
    """Ground truth for one injected contradiction.

    Attributes:
        pair_id: Which (structured, free-text) pair carries it.
        structured_field: The dropdown field id.
        free_text_field: The rationale field id.
        ir35_test: The shared test.
        structured_value: The option the manager selected.
        contradiction_type: From the taxonomy in ``schema.yaml``.
        subtlety: 1 blatant, 2 requires reading, 3 requires domain knowledge.
        undermines_outside_leaning: True when the contradiction attacks an
            answer that pointed away from employment. These are the audit risk
            under HMRC's reasonable-care duty and are reported separately.
        clean_unit: The supporting proposition that was replaced. Kept so the
            record can be re-rendered without the contradiction, which is how
            the style-invariance and subtlety analyses get a control condition.
        planted_unit: The contradicting proposition that replaced it.
    """

    pair_id: str
    structured_field: str
    free_text_field: str
    ir35_test: str
    structured_value: str
    contradiction_type: str
    subtlety: int
    undermines_outside_leaning: bool
    clean_unit: str
    planted_unit: str


@dataclass(frozen=True)
class Anomaly:
    """A non-contradiction data-quality problem.

    Kept separate from ``PlantedContradiction`` on purpose. A blank rationale is
    a completeness failure the form itself cares about — "Your ESQ will be
    rejected if not all fields have been completed" — and a reviewer needs to
    see it, but scoring it as a contradiction would flatter the detector.
    """

    field_id: str
    kind: str


@dataclass
class RecordState:
    """Everything needed to render a record, in any register."""

    record_id: str
    seed: int
    archetype: str
    register: str
    vocab: dict[str, str]
    answers: dict[str, str]
    reachable: list[str]
    units: dict[str, str] = dc_field(default_factory=dict)
    contradictions: list[PlantedContradiction] = dc_field(default_factory=list)
    anomalies: list[Anomaly] = dc_field(default_factory=list)
    identity: dict[str, str] = dc_field(default_factory=dict)
    duties_parts: list[str] = dc_field(default_factory=list)
    label: StatusResult | None = None
    is_ambiguous_by_design: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialise the state, excluding the rendered text."""
        return {
            "record_id": self.record_id,
            "seed": self.seed,
            "archetype": self.archetype,
            "register": self.register,
            "vocab": self.vocab,
            "answers": self.answers,
            "reachable": self.reachable,
            "units": self.units,
            "contradictions": [asdict(c) for c in self.contradictions],
            "anomalies": [asdict(a) for a in self.anomalies],
            "duties_parts": self.duties_parts,
            "is_ambiguous_by_design": self.is_ambiguous_by_design,
        }


# =============================================================================
# Generator
# =============================================================================
class EsqGenerator:
    """Builds synthetic ESQ submissions with ground-truth labels."""

    def __init__(
        self,
        schema: Schema,
        generation_config: Mapping[str, Any],
        text_bank: Mapping[str, Any],
        engine: CestRuleEngine,
        logger: StructuredLogger | None = None,
    ) -> None:
        """Create a generator.

        Args:
            schema: The loaded data contract.
            generation_config: Parsed ``config/generation.yaml``.
            text_bank: Parsed ``config/text_bank.yaml``.
            engine: The labelling rule engine.
            logger: Structured logger.
        """
        self.schema = schema
        self.cfg = generation_config
        self.bank = text_bank
        self.engine = engine
        self.realiser = RegisterRealiser(text_bank["registers"])
        self.log = logger or get_logger("generate")
        self.dataset_version: str = str(generation_config.get("dataset_version", "0"))

        self._archetypes: Mapping[str, Any] = generation_config["archetypes"]
        self._archetype_names = list(generation_config["archetype_mix"])
        self._archetype_weights = [
            generation_config["archetype_mix"][n] for n in self._archetype_names
        ]
        self._pair_by_structured = {
            p.structured: p for p in schema.pairs.values()
        }
        self._field_priors: Mapping[str, Mapping[str, float]] = generation_config.get(
            "field_priors", {}
        )
        self._plan_cache: dict[int, list[str]] | None = None
        self._plan_size: int | None = None
        # Constraint 5: the only field ids the generator knows come from config.
        self._id: Mapping[str, str] = generation_config["identity_field_map"]
        self._fid: Mapping[str, str] = generation_config["engagement_fields"]
        self._section4_prefix: str = str(
            generation_config.get("section_four_prefix", "4.")
        )

    # -- sampling helpers -------------------------------------------------
    @staticmethod
    def _weighted(rng: random.Random, mapping: Mapping[Any, float]) -> Any:
        """Draw one key from a mapping of key -> weight."""
        keys = list(mapping)
        return rng.choices(keys, weights=[float(mapping[k]) for k in keys], k=1)[0]

    def _polarity_pools(self, field_id: str) -> tuple[list[str], list[str]]:
        """Split a field's value domain into outside-leaning and inside-leaning.

        Uses the same value scores as the labelling engine, so an archetype's
        "60% outside on control" bias means the same thing the label means.
        Neutral values (score exactly zero) join both pools — they are genuine
        answers and excluding them would silently shrink the value domain.
        """
        table = self.engine.value_scores.get(field_id, {})
        outside = [v for v, s in table.items() if s > 0]
        inside = [v for v, s in table.items() if s < 0]
        neutral = [v for v, s in table.items() if s == 0]
        return outside + neutral, inside + neutral

    def _sample_answer(
        self, rng: random.Random, field_id: str, bias: float
    ) -> str | None:
        """Choose an option value.

        A field with an explicit prior in ``field_priors`` is drawn from that
        prior; everything else is drawn by polarity bias.

        The distinction matters. Polarity sampling asks "should this answer lean
        outside?", which is the right model for most of the form. For a few
        questions the answer is governed by a base rate in the world instead:
        office holders are rare, and a substitute having actually been sent and
        paid is rare. Both are determinative gates in the labelling engine, so
        sampling them by polarity put roughly half the corpus behind a gate and
        made the label distribution meaningless.

        Args:
            rng: Seeded generator.
            field_id: Structured field to answer.
            bias: Probability of drawing from the outside-leaning pool.
        """
        field = self.schema.fields.get(field_id)
        if field is None or not field.value_domain:
            return None
        prior = self._field_priors.get(field_id)
        if prior:
            return str(self._weighted(rng, prior))
        outside, inside = self._polarity_pools(field_id)
        pool = outside if rng.random() < bias else inside
        if not pool:
            pool = list(field.value_domain)
        return rng.choice(pool)

    def _apply_consistency(
        self, rng: random.Random, answers: dict[str, str]
    ) -> list[Anomaly]:
        """Enforce cross-field consistency, with deliberate violations.

        Work that has not started cannot have had a substitute sent. The form's
        own cross-field check ``x_started_but_not_applicable`` exists because
        managers get this wrong, so the corpus needs both the consistent
        majority and a measured share of violations — injected on purpose at
        ``anomalies.routing_violation_rate``, so the ground truth knows which
        records are inconsistent rather than discovering it later.

        Args:
            rng: Seeded generator.
            answers: Structured answers, modified in place.

        Returns:
            Any anomalies deliberately injected.
        """
        injected: list[Anomaly] = []
        violation_rate = float(self.cfg["anomalies"]["routing_violation_rate"])
        for rule in self.cfg.get("consistency_rules", []):
            conditions: Mapping[str, str] = rule.get("when", {})
            if not all(answers.get(k) == v for k, v in conditions.items()):
                continue
            if rng.random() < violation_rate:
                for field_id in {**rule.get("force", {}), **rule.get("forbid", {})}:
                    injected.append(Anomaly(field_id=field_id, kind="routing_violation"))
                continue
            for field_id, value in rule.get("force", {}).items():
                answers[field_id] = value
            for field_id, forbidden in rule.get("forbid", {}).items():
                if answers.get(field_id) == forbidden:
                    answers[field_id] = rule["fallback"][field_id]
        return injected

    # -- state ------------------------------------------------------------
    def build_state(self, index: int) -> RecordState:
        """Build one record's state.

        Args:
            index: Record index. Combined with the config seed so each record is
                independently reproducible — regenerating record 137 does not
                require replaying the previous 136.

        Returns:
            A fully populated ``RecordState`` with a label attached.
        """
        seed = int(self.cfg["seed"]) + index
        rng = random.Random(seed)
        record_id = f"SYN-{index:04d}"

        archetype = rng.choices(
            self._archetype_names, weights=self._archetype_weights, k=1
        )[0]
        spec = self._archetypes[archetype]

        register_source = (
            spec.get("register_bias")
            if self.cfg["registers"].get("use_archetype_bias", True)
            else None
        ) or self.cfg["registers"]["mix"]
        register = self._weighted(rng, register_source)

        vocab = VocabContext.sample(
            rng,
            self.bank["vocab"]["common"],
            self.bank["vocab"].get(archetype, {}),
        )

        # Per-test polarity bias, with optional ambiguity injection.
        bias = dict(spec["polarity_bias"])
        ambiguous = rng.random() < float(self.cfg["ambiguity"]["rate"])
        if ambiguous and len(bias) >= 2:
            pushed = rng.sample(sorted(bias), 2)
            bias[pushed[0]] = float(self.cfg["ambiguity"]["push_outside_bias"])
            bias[pushed[1]] = float(self.cfg["ambiguity"]["push_inside_bias"])

        answers: dict[str, str] = {}
        for field in self.schema.structured_answer_fields:
            test_bias = bias.get(field.ir35_test or "", 0.5)
            value = self._sample_answer(rng, field.id, test_bias)
            if value is not None:
                answers[field.id] = value

        # Section 2/3 route depends on how the worker is engaged, not on a test.
        via_third_party = rng.random() < float(spec.get("via_third_party_rate", 0.5))
        answers[self._fid["via_third_party"]] = "Yes" if via_third_party else "No"
        if not via_third_party:
            answers[self._fid["engaged_directly"]] = "Yes"

        consistency_anomalies = self._apply_consistency(rng, answers)
        reachable = walk_route(self.schema, answers, self._section4_prefix)
        for field in self.schema.structured_answer_fields:
            if (
                str(field.form_ref or "").startswith(self._section4_prefix)
                and field.id not in reachable
            ):
                answers.pop(field.id, None)

        label = self.engine.evaluate(answers)

        state = RecordState(
            record_id=record_id,
            seed=seed,
            archetype=archetype,
            register=register,
            vocab=dict(vocab.values),
            answers=answers,
            reachable=reachable,
            label=label,
            is_ambiguous_by_design=ambiguous,
            anomalies=list(consistency_anomalies),
        )

        self._choose_supporting_units(rng, state)
        self._plant_contradictions(rng, state, index)
        self._inject_anomalies(rng, state)
        self._build_identity(rng, state)
        self._build_duties(rng, state)
        return state

    def _choose_supporting_units(self, rng: random.Random, state: RecordState) -> None:
        """Pick one supporting proposition per answered pair."""
        for pair in self.schema.pairs.values():
            value = state.answers.get(pair.structured)
            if value is None:
                continue
            entry = self.bank["pairs"].get(pair.id, {})
            options = (entry.get("supporting") or {}).get(value)
            state.units[pair.id] = (
                rng.choice(list(options)) if options else _GENERIC_SUPPORTING
            )

    def _contradiction_candidates(self, state: RecordState) -> list[tuple[str, list[Mapping[str, Any]]]]:
        """Pairs that have an authored contradiction for the selected answer."""
        candidates: list[tuple[str, list[Mapping[str, Any]]]] = []
        for pair in self.schema.pairs.values():
            value = state.answers.get(pair.structured)
            if value is None:
                continue
            entry = self.bank["pairs"].get(pair.id, {})
            units = (entry.get("contradicting") or {}).get(value)
            if units:
                candidates.append((pair.id, list(units)))
        return candidates

    def contradiction_plan(self, total: int | None = None) -> dict[int, list[str]]:
        """Decide, corpus-wide, which records carry which contradiction types.

        WHY A PLAN AND NOT PURE PER-RECORD SAMPLING
        Sampling contradictions independently per record produced a type
        distribution far too skewed to evaluate: ``named_person_dependency``
        appeared once in 350 records. That is not a bug in the sampler — it
        falls out of how many pairs can express each type and how often those
        pairs are reachable — but a type with n=1 makes per-type recall
        meaningless, and Phase 4's headline deliverable is precision and recall
        *per contradiction type*.

        So the corpus is stratified by type: a balanced pool of target types is
        dealt out across the records selected to carry contradictions, and each
        record then picks a pair that can express its assigned type.

        THE TRADE-OFF, STATED PLAINLY
        The type mix is now a property of the experiment design, not an estimate
        of how contradictions are distributed in real submissions. Per-type
        recall is estimable; the type *prior* is not evidence about the world.
        Phase 4's write-up must say so, and must not report an aggregate
        contradiction-detection rate as if it were a field estimate.

        Determinism is preserved: the plan is a pure function of the master seed
        and the record count, so record *i* is still reproducible on its own.

        Args:
            total: Corpus size. Defaults to the configured ``n_records``.

        Returns:
            Record index -> list of target contradiction types.
        """
        cfg = self.cfg["contradictions"]
        size = total if total is not None else int(self.cfg["n_records"])
        if self._plan_cache is not None and self._plan_size == size:
            return self._plan_cache

        rng = random.Random(int(self.cfg["seed"]) ^ 0x9E3779B9)
        n_with = round(size * float(cfg["record_rate"]))
        indices = sorted(rng.sample(range(size), min(n_with, size)))

        counts = {
            idx: int(self._weighted(rng, {int(k): v for k, v in cfg["count_distribution"].items()}))
            for idx in indices
        }
        needed = sum(counts.values())

        # Exclude non_responsive: it is an anomaly, not a contradiction, and is
        # injected separately so Phase 4 can report the two apart.
        types = [t for t in self.schema.contradiction_types if t != "non_responsive"]
        pool: list[str] = []
        while len(pool) < needed:
            block = list(types)
            rng.shuffle(block)
            pool.extend(block)
        pool = pool[:needed]
        rng.shuffle(pool)

        plan: dict[int, list[str]] = {}
        cursor = 0
        for idx in indices:
            take = counts[idx]
            plan[idx] = pool[cursor : cursor + take]
            cursor += take

        self._plan_cache = plan
        self._plan_size = size
        return plan

    def _plant_contradictions(
        self, rng: random.Random, state: RecordState, index: int
    ) -> None:
        """Replace supporting propositions with contradicting ones.

        The record's assigned types come from ``contradiction_plan``. Within a
        type, pair choice is weighted towards the high-priority pairs (where a
        contradiction most directly flips a determination) and towards answers
        that lean outside (where undermining the justification is the audit risk
        under HMRC's reasonable-care duty). Neither weighting is exclusive: the
        low-priority pairs and inside-leaning answers still get cases, or the
        evaluation would say nothing about them.
        """
        cfg = self.cfg["contradictions"]
        targets = self.contradiction_plan().get(index)
        if not targets:
            return

        candidates = self._contradiction_candidates(state)
        if not candidates:
            return

        used: set[str] = set()
        for target_type in targets:
            eligible = [
                (pair_id, [u for u in units if str(u.get("type")) == target_type])
                for pair_id, units in candidates
                if pair_id not in used
            ]
            eligible = [(pid, units) for pid, units in eligible if units]
            if not eligible:
                # No reachable pair can express this type on this record. Fall
                # back to any available contradiction rather than dropping the
                # planned instance, and let the ground truth record what was
                # actually planted.
                eligible = [(pid, units) for pid, units in candidates if pid not in used]
            if not eligible:
                break

            weights = []
            for pair_id, _ in eligible:
                pair = self.schema.pairs[pair_id]
                weight = (
                    float(cfg["high_priority_pair_weight"]) if pair.is_high_priority else 1.0
                )
                if state.answers.get(pair.structured) == pair.outside_leaning:
                    rate = float(cfg["target_outside_leaning_rate"])
                    weight *= rate / (1.0 - rate)
                weights.append(weight)

            pick = rng.choices(range(len(eligible)), weights=weights, k=1)[0]
            pair_id, units = eligible[pick]
            used.add(pair_id)

            # Subtlety is drawn per contradiction, not per record. Drawing once
            # per record would correlate subtlety with every other record-level
            # property (archetype, register), and Phase 4's subtlety curve would
            # be confounded by the archetype breakdown.
            target_subtlety = int(
                self._weighted(
                    rng, {int(k): v for k, v in cfg["subtlety_distribution"].items()}
                )
            )
            matching = [u for u in units if int(u.get("subtlety", 2)) == target_subtlety]
            unit = rng.choice(matching or units)

            pair = self.schema.pairs[pair_id]
            value = state.answers[pair.structured]
            state.contradictions.append(
                PlantedContradiction(
                    pair_id=pair_id,
                    structured_field=pair.structured,
                    free_text_field=pair.free_text,
                    ir35_test=pair.ir35_test,
                    structured_value=value,
                    contradiction_type=str(unit["type"]),
                    subtlety=int(unit.get("subtlety", 2)),
                    undermines_outside_leaning=(value == pair.outside_leaning),
                    clean_unit=state.units.get(pair_id, ""),
                    planted_unit=str(unit["text"]),
                )
            )
            state.units[pair_id] = str(unit["text"])

    def _inject_anomalies(self, rng: random.Random, state: RecordState) -> None:
        """Add non-responsive or blank justifications.

        Never applied to a pair carrying a planted contradiction: blanking the
        text would destroy the contradiction while the ground truth still
        claimed it was there, which would silently corrupt recall measurement.
        """
        cfg = self.cfg["anomalies"]
        occupied = {c.pair_id for c in state.contradictions}
        available = [pid for pid in state.units if pid not in occupied]
        if not available:
            return
        for kind, rate in (
            ("non_responsive", cfg["non_responsive_rate"]),
            ("blank", cfg["blank_justification_rate"]),
        ):
            if rng.random() < float(rate) and available:
                pair_id = rng.choice(available)
                available.remove(pair_id)
                state.anomalies.append(
                    Anomaly(field_id=self.schema.pairs[pair_id].free_text, kind=kind)
                )
                state.units[pair_id] = "" if kind == "blank" else "__NON_RESPONSIVE__"

    def _build_identity(self, rng: random.Random, state: RecordState) -> None:
        """Populate the personal-detail fields from the invented pools.

        These values exist so that the de-identifier is exercised on every
        record rather than assumed to work. They are recombined fragments, not
        a list of real people (constraint 1).
        """
        pools = self.cfg["identity_pools"]
        forename = rng.choice(pools["forename_fragments"]) + rng.choice(pools["forename_endings"])
        surname = rng.choice(pools["surname_fragments"]) + rng.choice(pools["surname_endings"])
        officer = (
            rng.choice(pools["forename_fragments"]) + rng.choice(pools["forename_endings"]),
            rng.choice(pools["surname_fragments"]) + rng.choice(pools["surname_endings"]),
        )
        postcode = (
            f"{rng.choice(pools['postcode_areas'])}{rng.randint(1, 40)} "
            f"{rng.randint(1, 9)}{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}"
            f"{rng.choice('ABDEFGHJLNPQRSTUWXYZ')}"
        )
        ident = self._id
        identity = {
            ident["title"]: rng.choice(pools["titles"]),
            ident["first_name"]: forename,
            ident["surname"]: surname,
            ident["address_line1"]: (
                f"{rng.randint(1, 180)} {rng.choice(pools['street_names'])} "
                f"{rng.choice(pools['street_types'])}"
            ),
            ident["address_line2"]: rng.choice(pools["towns"]),
            ident["postcode"]: postcode,
            ident["telephone"]: f"07{rng.randint(100, 999)} {rng.randint(100000, 999999)}",
            ident["internet_presence"]: (
                f"www.{surname.lower()}-"
                f"{rng.choice(['consulting', 'associates', 'services'])}.example.com"
            ),
            ident["school_division"]: rng.choice(pools["schools"]),
            ident["engaging_officer"]: f"{officer[0]} {officer[1]}",
            ident["officer_phone"]: f"01273 {rng.randint(600000, 899999)}",
            ident["officer_email"]: (
                f"{officer[0].lower()}.{officer[1].lower()}"
                f"@{rng.choice(pools['email_domains'])}"
            ),
        }
        if state.answers.get(self._fid["via_third_party"]) == "Yes":
            identity[ident["company_name"]] = (
                f"{surname} {rng.choice(['', 'Technical ', 'Advisory '])}"
                f"{rng.choice(pools['company_suffixes'])}".replace("  ", " ")
            )
            identity[ident["company_number"]] = f"{rng.randint(10000000, 99999999)}"
        state.identity = identity

    def _build_duties(self, rng: random.Random, state: RecordState) -> None:
        """Choose the sentences for the document-level duties narrative."""
        bank = self.bank["duties_narrative"]
        state.duties_parts = [
            rng.choice(bank["openings"]),
            rng.choice(bank["bodies"]),
            rng.choice(bank["closings"]),
        ]

    # -- rendering --------------------------------------------------------
    def render(self, state: RecordState, register: str | None = None) -> dict[str, Any]:
        """Render a state as a submission.

        Args:
            state: A built record state.
            register: Override the state's register. This is how Phase 4's
                style-invariance test produces three surface forms of one
                engagement without changing anything it says.

        Returns:
            A record dict keyed by schema field id, plus bookkeeping keys
            (``record_id``, ``archetype``, ``register``).
        """
        chosen = register or state.register
        # zlib.crc32, NOT the builtin hash(). Python salts string hashing per
        # process (PYTHONHASHSEED), so hash("verbose") differs between runs and
        # the "regenerating from config reproduces the corpus byte-for-byte"
        # claim held only within a single process. The in-process determinism
        # test passed throughout and could never have caught it. crc32 is stable
        # across processes, versions and platforms.
        rng = random.Random(state.seed + zlib.crc32(chosen.encode("utf-8")) % 10_000)
        vocab = VocabContext(state.vocab)

        record: dict[str, Any] = {
            "record_id": state.record_id,
            "archetype": state.archetype,
            "register": chosen,
            "dataset_version": self.dataset_version,
        }
        record.update(state.identity)
        record.update(state.answers)

        blanks = {a.field_id for a in state.anomalies if a.kind == "blank"}
        for pair_id, unit in state.units.items():
            free_text_field = self.schema.pairs[pair_id].free_text
            if free_text_field in blanks or unit == "":
                record[free_text_field] = ""
            elif unit == "__NON_RESPONSIVE__":
                record[free_text_field] = vocab.fill(
                    rng.choice(self.bank["non_responsive"])
                )
            else:
                record[free_text_field] = self.realiser.realise(unit, chosen, rng, vocab)

        self._add_identity_mentions(rng, state, record)

        record[self._fid["role_title"]] = vocab.fill("{deliverable}").title()
        record[self._fid["duties_narrative"]] = self.realiser.realise_narrative(
            state.duties_parts, chosen, rng, vocab
        )
        if state.answers.get(self._fid["via_third_party"]) == "Yes":
            record[self._fid["entity_relationship"]] = self.realiser.realise(
                "the worker is the sole director and shareholder of the company",
                chosen,
                rng,
                vocab,
            )
            record[self._fid["years_in_business"]] = f"{rng.randint(1, 14)} years"
        return record

    def _add_identity_mentions(
        self, rng: random.Random, state: RecordState, record: dict[str, Any]
    ) -> None:
        """Have the manager name the worker inside a justification box.

        This is the realistic leak: the schema classifies a rationale box as
        carrying no personal data, and a manager arguing that nobody else can do
        the work names the person they cannot replace. Putting it in the corpus
        means the de-identifier's gazetteer — its highest-precision and
        highest-stakes path — is exercised on generated records, not only in
        unit tests.

        Args:
            rng: Seeded generator.
            state: The record state, for the identity values.
            record: The rendered record, modified in place.
        """
        cfg = self.cfg.get("identity_mentions", {})
        if rng.random() >= float(cfg.get("record_rate", 0.0)):
            return
        templates = self.bank.get("identity_mentions") or []
        if not templates:
            return

        filled = {
            "title": state.identity.get(self._id["title"], ""),
            "first": state.identity.get(self._id["first_name"], ""),
            "surname": state.identity.get(self._id["surname"], ""),
        }
        eligible = [
            self.schema.pairs[pid].free_text
            for pid in state.units
            if record.get(self.schema.pairs[pid].free_text)
        ]
        if not eligible:
            return
        for field_id in rng.sample(
            eligible, min(int(cfg.get("max_per_record", 1)), len(eligible))
        ):
            if rng.random() < 0.5 and field_id != eligible[0]:
                continue
            mention = VocabContext(filled).fill(rng.choice(templates))
            record[field_id] = f"{record[field_id]} {mention}"

    def ground_truth(self, state: RecordState) -> dict[str, Any]:
        """Build the ground-truth record for one state.

        Returns:
            Labels and provenance: the status label with its full score
            breakdown, every planted contradiction with type and subtlety, every
            anomaly, and the archetype/register cohort variables Phase 4 needs
            for its disparity and style analyses.
        """
        assert state.label is not None
        return {
            "record_id": state.record_id,
            "seed": state.seed,
            "archetype": state.archetype,
            "register": state.register,
            "dataset_version": self.dataset_version,
            "ir35_label": state.label.status.value,
            "label_detail": state.label.as_dict(),
            "is_ambiguous_by_design": state.is_ambiguous_by_design,
            "n_contradictions": len(state.contradictions),
            "has_contradiction": bool(state.contradictions),
            "contradictions": [asdict(c) for c in state.contradictions],
            "anomalies": [asdict(a) for a in state.anomalies],
            "contradiction_pairs": sorted({c.pair_id for c in state.contradictions}),
        }

    # -- corpus -----------------------------------------------------------
    def generate(
        self, n_records: int | None = None
    ) -> tuple[list[RecordState], list[dict[str, Any]], list[dict[str, Any]]]:
        """Generate the corpus.

        Args:
            n_records: Override the configured record count.

        Returns:
            States, rendered records and ground-truth records, index-aligned.
        """
        total = n_records if n_records is not None else int(self.cfg["n_records"])
        states = [self.build_state(i) for i in range(total)]
        records = [self.render(s) for s in states]
        truth = [self.ground_truth(s) for s in states]
        self.log.event(
            "generate.completed",
            meta={
                "n_records": total,
                "dataset_version": self.dataset_version,
                "seed": self.cfg["seed"],
                "with_contradictions": sum(1 for t in truth if t["has_contradiction"]),
                "labels": {
                    label: sum(1 for t in truth if t["ir35_label"] == label)
                    for label in ("inside", "outside", "undetermined")
                },
            },
        )
        return states, records, truth
