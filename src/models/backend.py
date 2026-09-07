"""The swappable inference backend — constraint 2.

All inference goes through ``InferenceBackend``. Which implementation runs is
decided by ``config/model.yaml`` and nothing else: no detector imports
``transformers``, no detector knows an API exists, and switching from a local
model to a hosted endpoint is a one-line config edit.

Why the interface has exactly two methods
-----------------------------------------
``classify_nli`` and ``generate``. Those are the two shapes of inference this
project needs, and keeping the surface that small is what makes the
implementations genuinely interchangeable. A wider interface would leak the
capabilities of one backend into the callers and quietly make the swap a lie.

A backend that cannot do one of the two raises ``UnsupportedOperation`` rather
than degrading silently — a cross-encoder NLI model cannot generate text, and
pretending otherwise would produce plausible nonsense.

The local-only posture
----------------------
ESQ submissions contain named individuals, day rates and tax-sensitive
information. Sending them to a hosted endpoint is a third-country transfer and
an Information Governance decision, not an engineering default. Two guards:
``require_local`` refuses to construct a hosted backend at all, and the hosted
backend refuses payloads that have not been through de-identification. Both fail
at construction or call time rather than warning in a log nobody reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.logging import StructuredLogger, get_logger

__all__ = [
    "NliLabel",
    "NliResult",
    "GenerationResult",
    "UnsupportedOperation",
    "BackendUnavailable",
    "InferenceBackend",
    "MockBackend",
    "LocalTransformersBackend",
    "LocalLlmBackend",
    "HostedApiBackend",
    "get_backend",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


class UnsupportedOperation(NotImplementedError):
    """The selected backend cannot perform this kind of inference."""


class BackendUnavailable(RuntimeError):
    """The backend is selected but cannot run — missing weights, key or package.

    Raised with a message that says what to do about it. A backend that fails
    with an ImportError deep in a batch is a much worse experience than one that
    refuses to start and names the missing piece.
    """


class NliLabel(str):
    """The three NLI classes, as plain strings for serialisability."""

    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class NliResult:
    """A probability distribution over the three NLI classes.

    Attributes:
        entailment: P(the hypothesis follows from the premise).
        neutral: P(neither follows nor conflicts). For this project the neutral
            mass is a signal in its own right — a definite answer justified by
            text that neither supports nor contradicts it is the shape of a
            hedged non-supporting justification.
        contradiction: P(the hypothesis conflicts with the premise).
    """

    entailment: float
    neutral: float
    contradiction: float

    def as_dict(self) -> dict[str, float]:
        """Serialise."""
        return {
            "entailment": round(self.entailment, 6),
            "neutral": round(self.neutral, 6),
            "contradiction": round(self.contradiction, 6),
        }

    @property
    def top_label(self) -> str:
        """The most probable class."""
        return max(
            (
                (self.entailment, NliLabel.ENTAILMENT),
                (self.neutral, NliLabel.NEUTRAL),
                (self.contradiction, NliLabel.CONTRADICTION),
            )
        )[1]


@dataclass
class GenerationResult:
    """One generation, plus what it cost.

    Attributes:
        text: The generated text.
        input_tokens: Prompt tokens, where the backend reports them.
        output_tokens: Completion tokens, where the backend reports them.
        latency_seconds: Wall-clock time for the call.
        error: Set when the call failed and was not retried successfully.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    error: str | None = None


class InferenceBackend(ABC):
    """One interface, several implementations, selected by config."""

    #: Short name used in reports and logs.
    name: str = "backend"

    #: True when inference runs entirely on this machine.
    is_local: bool = True

    def __init__(self, config: Mapping[str, Any], logger: StructuredLogger | None = None) -> None:
        """Create a backend.

        Args:
            config: The selected entry from the ``backends`` block of
                ``config/model.yaml``.
            logger: Structured logger.
        """
        self.config = dict(config)
        self.log = logger or get_logger(f"backend.{self.name}")

    def classify_nli(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[NliResult]:
        """Classify (premise, hypothesis) pairs.

        Args:
            pairs: Premise/hypothesis pairs.

        Returns:
            One distribution per pair, in order.

        Raises:
            UnsupportedOperation: If this backend does not do NLI.
        """
        raise UnsupportedOperation(f"{self.name} does not support NLI classification")

    def generate(
        self, prompts: Sequence[str], system: str | None = None
    ) -> list[GenerationResult]:
        """Generate text for each prompt.

        Args:
            prompts: User prompts.
            system: Optional system prompt applied to all of them.

        Returns:
            One result per prompt, in order.

        Raises:
            UnsupportedOperation: If this backend does not generate.
        """
        raise UnsupportedOperation(f"{self.name} does not support generation")

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Report whether this backend can actually run.

        Returns:
            ``(ok, message)``. Called before a long run so that a missing model
            or key fails immediately with a message that says what to do, rather
            than part-way through a batch.
        """

    def describe(self) -> dict[str, Any]:
        """Metadata for the run record. No content, so it is safe to log."""
        return {
            "backend": self.name,
            "is_local": self.is_local,
            "model": self.config.get("model_id") or self.config.get("model", ""),
        }


# =============================================================================
# Mock
# =============================================================================
class MockBackend(InferenceBackend):
    """Deterministic stub. No weights, no network.

    Exists so that every component downstream of the backend can be tested
    properly — premise construction, score mapping, aggregation, parsing,
    threshold selection — without model weights being available. That matters
    beyond convenience: the plumbing bugs this catches are the ones that would
    otherwise be discovered only after a long inference run.

    Outputs are a deterministic function of the input text, so tests are
    reproducible. It also honours a small set of trigger phrases, which lets a
    test assert "given a contradiction, the detector scores it high" without
    depending on a real model's judgement.
    """

    name = "mock"
    is_local = True

    #: Text containing any of these is returned as a confident contradiction.
    CONTRADICTION_TRIGGERS = (
        "nobody else",
        "specifically need",
        "recharged",
        "reimbursed",
        "full time",
        "monday to friday",
        "supervised",
    )

    #: Text containing any of these is returned as hedged / neutral.
    NEUTRAL_TRIGGERS = ("i believe", "as far as i am aware", "not been tested", "hypothetical")

    def _hash_unit(self, text: str) -> float:
        """A stable pseudo-random number in [0, 1) from the text."""
        seed = str(self.config.get("seed", 0))
        digest = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def classify_nli(self, pairs: Sequence[tuple[str, str]]) -> list[NliResult]:
        """Return a deterministic distribution per pair."""
        out: list[NliResult] = []
        for premise, hypothesis in pairs:
            text = hypothesis.lower()
            if any(trigger in text for trigger in self.CONTRADICTION_TRIGGERS):
                out.append(NliResult(entailment=0.05, neutral=0.10, contradiction=0.85))
            elif any(trigger in text for trigger in self.NEUTRAL_TRIGGERS):
                out.append(NliResult(entailment=0.15, neutral=0.75, contradiction=0.10))
            else:
                jitter = self._hash_unit(premise + hypothesis) * 0.2
                out.append(
                    NliResult(
                        entailment=0.70 - jitter,
                        neutral=0.22 + jitter,
                        contradiction=0.08,
                    )
                )
        return out

    @staticmethod
    def _case_under_test(prompt: str) -> str:
        """Extract just the case being asked about, ignoring the few-shot shots.

        A few-shot prompt contains worked examples as well as the case under
        test, and some of those examples legitimately contain the same words the
        triggers look for — the materials shot contains "reimbursed". Matching
        against the whole prompt therefore returned "contradicts" for every
        input, which a test caught. Everything after the last ``JUSTIFICATION:``
        marker is the case actually being asked about.
        """
        marker = "JUSTIFICATION:"
        if marker in prompt:
            return prompt.rsplit(marker, 1)[1].lower()
        return prompt.lower()

    def generate(
        self, prompts: Sequence[str], system: str | None = None
    ) -> list[GenerationResult]:
        """Return a deterministic JSON verdict per prompt."""
        out: list[GenerationResult] = []
        for prompt in prompts:
            text = self._case_under_test(prompt)
            if any(trigger in text for trigger in self.CONTRADICTION_TRIGGERS):
                verdict, confidence = "contradicts", 0.88
                reason = "The justification states the opposite of the selected option."
            elif any(trigger in text for trigger in self.NEUTRAL_TRIGGERS):
                verdict, confidence = "unclear", 0.5
                reason = "The justification is hedged and does not support the option chosen."
            else:
                verdict, confidence = "consistent", 0.8
                reason = "The justification supports the selected option."
            out.append(
                GenerationResult(
                    text=json.dumps(
                        {"verdict": verdict, "confidence": confidence, "reason": reason}
                    ),
                    input_tokens=len(prompt) // 4,
                    output_tokens=30,
                    latency_seconds=0.0,
                )
            )
        return out

    def health_check(self) -> tuple[bool, str]:
        """Always available."""
        return True, "mock backend: deterministic, no weights required"


# =============================================================================
# Local transformers (cross-encoder NLI)
# =============================================================================
class LocalTransformersBackend(InferenceBackend):
    """Cross-encoder NLI run locally with ``transformers``.

    Imports ``transformers`` lazily, inside ``_load``, so that the package is a
    requirement only for the run that actually selects this backend. A project
    whose test suite imports torch to check a config file is a project nobody
    can run on a laptop.
    """

    name = "local_transformers"
    is_local = True

    def __init__(self, config: Mapping[str, Any], logger: StructuredLogger | None = None) -> None:
        """Create the backend. Weights are loaded on first use, not here."""
        super().__init__(config, logger)
        self._model = None
        self._tokenizer = None
        self._label_index: dict[str, int] | None = None

    @property
    def cache_dir(self) -> Path:
        """Where weights are expected to be cached."""
        cache = self.config.get("cache_dir", "models/weights")
        path = Path(cache)
        return path if path.is_absolute() else REPO_ROOT / path

    def _load(self) -> None:
        """Load tokenizer and model, once.

        Raises:
            BackendUnavailable: If transformers is missing or the weights are
                not cached locally. The message names the fetch script rather
                than leaving the caller to work it out.
        """
        if self._model is not None:
            return
        try:
            from transformers import (  # noqa: PLC0415
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise BackendUnavailable(
                "transformers is not installed. `pip install -r requirements.txt`, "
                "or select backend: mock in config/model.yaml."
            ) from exc

        model_id = self.config["model_id"]
        kwargs = {
            "cache_dir": str(self.cache_dir),
            "local_files_only": bool(self.config.get("local_files_only", True)),
            "revision": self.config.get("revision", "main"),
        }
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise BackendUnavailable(
                f"could not load {model_id} from {self.cache_dir}. "
                "Run `python scripts/fetch_models.py` on a machine that can reach "
                "huggingface.co, or select backend: mock in config/model.yaml. "
                f"Underlying error: {exc}"
            ) from exc

        self._model.eval()
        # Prefer the checkpoint's own label mapping. Label ORDER differs between
        # NLI checkpoints, and silently assuming one is how a model ends up
        # reporting entailment as contradiction with no error anywhere.
        declared = getattr(self._model.config, "id2label", None) or {}
        mapping = {str(v).lower(): int(k) for k, v in declared.items()}
        if {"entailment", "neutral", "contradiction"} <= set(mapping):
            self._label_index = mapping
        else:
            order = self.config.get(
                "label_order", ["entailment", "neutral", "contradiction"]
            )
            self._label_index = {str(name).lower(): i for i, name in enumerate(order)}
            self.log.warning(
                "backend.label_order_assumed",
                meta={"model": model_id, "order": list(order)},
            )

    def classify_nli(self, pairs: Sequence[tuple[str, str]]) -> list[NliResult]:
        """Classify pairs in batches.

        Args:
            pairs: (premise, hypothesis) pairs.

        Returns:
            One distribution per pair.
        """
        import torch  # noqa: PLC0415

        self._load()
        assert self._label_index is not None
        batch_size = int(self.config.get("batch_size", 16))
        max_length = int(self.config.get("max_length", 512))
        results: list[NliResult] = []

        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            encoded = self._tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probabilities:
                results.append(
                    NliResult(
                        entailment=float(row[self._label_index["entailment"]]),
                        neutral=float(row[self._label_index["neutral"]]),
                        contradiction=float(row[self._label_index["contradiction"]]),
                    )
                )
        return results

    def health_check(self) -> tuple[bool, str]:
        """Try to load the model and report what happened."""
        try:
            self._load()
        except BackendUnavailable as exc:
            return False, str(exc)
        return True, f"{self.config['model_id']} loaded from {self.cache_dir}"


# =============================================================================
# Local instruction-tuned LLM
# =============================================================================
class LocalLlmBackend(InferenceBackend):
    """Instruction-tuned causal model run locally.

    The local-only counterpart of the hosted few-shot path. Its purpose in the
    method comparison is to answer a deployment question rather than an accuracy
    one: whether the few-shot approach is viable at all when the data cannot
    leave the University.
    """

    name = "local_llm"
    is_local = True

    def __init__(self, config: Mapping[str, Any], logger: StructuredLogger | None = None) -> None:
        """Create the backend. Weights load on first use."""
        super().__init__(config, logger)
        self._model = None
        self._tokenizer = None

    @property
    def cache_dir(self) -> Path:
        """Where weights are expected to be cached."""
        cache = Path(self.config.get("cache_dir", "models/weights"))
        return cache if cache.is_absolute() else REPO_ROOT / cache

    def _load(self) -> None:
        """Load tokenizer and model, once.

        Raises:
            BackendUnavailable: If transformers is missing or weights absent.
        """
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("transformers is not installed") from exc

        model_id = self.config["model_id"]
        kwargs = {
            "cache_dir": str(self.cache_dir),
            "local_files_only": bool(self.config.get("local_files_only", True)),
            "revision": self.config.get("revision", "main"),
        }
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailable(
                f"could not load {model_id} from {self.cache_dir}. "
                "Run `python scripts/fetch_models.py` where huggingface.co is reachable. "
                f"Underlying error: {exc}"
            ) from exc
        self._model.eval()

    def generate(
        self, prompts: Sequence[str], system: str | None = None
    ) -> list[GenerationResult]:
        """Generate a completion per prompt."""
        import torch  # noqa: PLC0415

        self._load()
        results: list[GenerationResult] = []
        for prompt in prompts:
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoded = self._tokenizer(text, return_tensors="pt")
            started = time.perf_counter()
            with torch.no_grad():
                generated = self._model.generate(
                    **encoded,
                    max_new_tokens=int(self.config.get("max_new_tokens", 200)),
                    do_sample=float(self.config.get("temperature", 0.0)) > 0,
                    temperature=max(float(self.config.get("temperature", 0.0)), 1e-5),
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            completion = self._tokenizer.decode(
                generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            results.append(
                GenerationResult(
                    text=completion,
                    input_tokens=int(encoded["input_ids"].shape[1]),
                    output_tokens=int(generated.shape[1] - encoded["input_ids"].shape[1]),
                    latency_seconds=time.perf_counter() - started,
                )
            )
        return results

    def health_check(self) -> tuple[bool, str]:
        """Try to load the model and report what happened."""
        try:
            self._load()
        except BackendUnavailable as exc:
            return False, str(exc)
        return True, f"{self.config['model_id']} loaded from {self.cache_dir}"


# =============================================================================
# Hosted API
# =============================================================================
class HostedApiBackend(InferenceBackend):
    """Hosted Messages API.

    Development and comparison only. Two guards, both failing loudly:

    * ``require_deidentified`` rejects any prompt still carrying an obvious
      identifier. It is a backstop, not a substitute for the de-identification
      stage — but a backstop is worth having on the one code path that sends
      data outside the University.
    * ``require_local`` in the top-level config prevents this backend being
      constructed at all.
    """

    name = "hosted_api"
    is_local = False

    #: Patterns that must not appear in a prompt sent off-site. These are the
    #: shapes the de-identifier replaces, so their presence means it did not run.
    _IDENTIFIER_PATTERNS = (
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b"),
        re.compile(r"(?:\+44\s?|\b0)(?:\d\s?){9,10}\b"),
    )

    def _credentials(self) -> tuple[str, str]:
        """Resolve base URL and API key from the environment.

        Raises:
            BackendUnavailable: If the API key is not set.
        """
        base = os.environ.get(
            str(self.config.get("base_url_env", "ANTHROPIC_BASE_URL")),
            "https://api.anthropic.com",
        ).rstrip("/")
        key = os.environ.get(str(self.config.get("api_key_env", "ANTHROPIC_API_KEY")), "")
        if not key:
            raise BackendUnavailable(
                f"{self.config.get('api_key_env', 'ANTHROPIC_API_KEY')} is not set. "
                "Set it, or select a local backend in config/model.yaml."
            )
        return base, key

    def _check_deidentified(self, prompt: str) -> None:
        """Refuse to send a prompt that still looks like raw submission text.

        Raises:
            ValueError: If an identifier pattern is present.
        """
        if not self.config.get("require_deidentified", True):
            return
        for pattern in self._IDENTIFIER_PATTERNS:
            if pattern.search(prompt):
                raise ValueError(
                    "refusing to send a prompt containing an apparent identifier to a "
                    "hosted endpoint; run de-identification first"
                )

    def generate(
        self, prompts: Sequence[str], system: str | None = None
    ) -> list[GenerationResult]:
        """Send each prompt to the hosted API, with retries."""
        import urllib.error  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        base, key = self._credentials()
        results: list[GenerationResult] = []
        max_retries = int(self.config.get("max_retries", 3))
        timeout = int(self.config.get("timeout_seconds", 60))

        for prompt in prompts:
            self._check_deidentified(prompt)
            payload: dict[str, Any] = {
                "model": self.config.get("model", "claude-sonnet-4-5"),
                "max_tokens": int(self.config.get("max_tokens", 400)),
                "temperature": float(self.config.get("temperature", 0.0)),
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                payload["system"] = system

            started = time.perf_counter()
            result: GenerationResult | None = None
            for attempt in range(max_retries):
                request = urllib.request.Request(
                    f"{base}/v1/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01",
                        "x-api-key": key,
                    },
                )
                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        body = json.loads(response.read().decode("utf-8"))
                    usage = body.get("usage", {})
                    result = GenerationResult(
                        text="".join(
                            block.get("text", "")
                            for block in body.get("content", [])
                            if block.get("type") == "text"
                        ),
                        input_tokens=int(usage.get("input_tokens", 0)),
                        output_tokens=int(usage.get("output_tokens", 0)),
                        latency_seconds=time.perf_counter() - started,
                    )
                    break
                except Exception as exc:  # pragma: no cover - network dependent
                    if attempt == max_retries - 1:
                        result = GenerationResult(
                            text="",
                            latency_seconds=time.perf_counter() - started,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    else:
                        time.sleep(2**attempt)
            results.append(result or GenerationResult(text="", error="no result"))
        return results

    def health_check(self) -> tuple[bool, str]:
        """Check the key is present. Does not spend a call to verify it."""
        try:
            base, _ = self._credentials()
        except BackendUnavailable as exc:
            return False, str(exc)
        return True, f"hosted API configured at {base}"


# =============================================================================
# Factory
# =============================================================================
_REGISTRY: dict[str, type[InferenceBackend]] = {
    "mock": MockBackend,
    "local_transformers": LocalTransformersBackend,
    "local_llm": LocalLlmBackend,
    "hosted_api": HostedApiBackend,
}


def get_backend(
    config: Mapping[str, Any],
    override: str | None = None,
    logger: StructuredLogger | None = None,
) -> InferenceBackend:
    """Build the backend named in config.

    This function is the whole of constraint 2. Callers pass the parsed
    ``config/model.yaml``; nothing else in the codebase names a backend class.

    Args:
        config: Parsed ``config/model.yaml``.
        override: Backend name to use instead of the configured one. For tests
            and for a ``--backend`` flag; not a way to bypass ``require_local``.
        logger: Structured logger.

    Returns:
        A ready backend. Weights are loaded lazily on first inference.

    Raises:
        KeyError: If the named backend has no entry in the ``backends`` block.
        BackendUnavailable: If ``require_local`` is set and the selected backend
            sends data off this machine.
    """
    name = override or str(config.get("backend", "mock"))
    backends = config.get("backends", {})
    if name not in backends:
        raise KeyError(
            f"backend {name!r} is not defined in config/model.yaml "
            f"(available: {sorted(backends)})"
        )
    entry = backends[name]
    kind = str(entry.get("kind", name))
    if kind not in _REGISTRY:
        raise KeyError(f"unknown backend kind {kind!r} (available: {sorted(_REGISTRY)})")

    backend = _REGISTRY[kind](entry, logger)
    backend.name = name

    if config.get("require_local", False) and not backend.is_local:
        raise BackendUnavailable(
            f"require_local is set in config/model.yaml but backend {name!r} sends data "
            "off this machine. ESQ submissions contain personal and tax-sensitive data; "
            "clear the hosted path with Information Governance before disabling this."
        )
    return backend
