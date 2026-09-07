#!/usr/bin/env python3
"""Download and cache the model weights Phase 3 needs.

Run this once, on a machine that can reach huggingface.co. Afterwards
``models/weights/`` holds everything, ``local_files_only: true`` keeps inference
off the network entirely, and the whole pipeline runs offline — which is the
deployment posture the project assumes, not just a convenience.

This is a separate script rather than an automatic download inside the backend
on purpose. A pipeline that silently reaches out to the internet mid-run is one
that behaves differently in the University's environment than in development,
and the difference surfaces at the worst possible moment. Fetching is an
explicit, auditable step.

Usage:
    python scripts/fetch_models.py                # everything in config/model.yaml
    python scripts/fetch_models.py --backend local_transformers
    python scripts/fetch_models.py --check        # report cache state, download nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingest.schema_loader import CONFIG_DIR, load_yaml  # noqa: E402

FETCHABLE_KINDS = {"local_transformers", "local_llm"}


def cache_dir_for(entry: Mapping[str, Any]) -> Path:
    """Resolve a backend's cache directory to an absolute path."""
    cache = Path(entry.get("cache_dir", "models/weights"))
    return cache if cache.is_absolute() else REPO_ROOT / cache


def check(entry: Mapping[str, Any]) -> tuple[bool, str]:
    """Report whether a model is already cached, without touching the network.

    Args:
        entry: One entry from the ``backends`` block of ``config/model.yaml``.

    Returns:
        ``(cached, message)``.
    """
    model_id = entry["model_id"]
    cache = cache_dir_for(entry)
    try:
        from transformers import AutoConfig  # noqa: PLC0415
    except ImportError:
        return False, "transformers is not installed"
    try:
        AutoConfig.from_pretrained(
            model_id,
            cache_dir=str(cache),
            local_files_only=True,
            revision=entry.get("revision", "main"),
        )
    except Exception:
        return False, f"not cached in {cache}"
    return True, f"cached in {cache}"


def fetch(entry: Mapping[str, Any], kind: str) -> tuple[bool, str]:
    """Download a model's weights and tokenizer into the cache.

    Args:
        entry: One entry from the ``backends`` block.
        kind: ``local_transformers`` or ``local_llm``.

    Returns:
        ``(ok, message)``. Never raises: a blocked network is a normal outcome
        for this script and the caller reports it rather than crashing.
    """
    model_id = entry["model_id"]
    cache = cache_dir_for(entry)
    cache.mkdir(parents=True, exist_ok=True)
    revision = entry.get("revision", "main")

    try:
        if kind == "local_transformers":
            from transformers import (  # noqa: PLC0415
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            loader = AutoModelForSequenceClassification
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            loader = AutoModelForCausalLM
    except ImportError:
        return False, "transformers is not installed — `pip install -r requirements.txt`"

    try:
        AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache), revision=revision)
        loader.from_pretrained(model_id, cache_dir=str(cache), revision=revision)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"downloaded to {cache}"


def main() -> int:
    """Fetch or check every fetchable backend named in config/model.yaml."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=None, help="fetch only this backend")
    parser.add_argument("--check", action="store_true", help="report cache state only")
    args = parser.parse_args()

    config = load_yaml(CONFIG_DIR / "model.yaml")
    backends = config.get("backends", {})
    names = [args.backend] if args.backend else list(backends)

    failures = 0
    for name in names:
        entry = backends.get(name)
        if entry is None:
            print(f"{name}: not defined in config/model.yaml")
            failures += 1
            continue
        kind = str(entry.get("kind", name))
        if kind not in FETCHABLE_KINDS:
            print(f"{name}: nothing to fetch ({kind})")
            continue

        cached, message = check(entry)
        if cached:
            print(f"{name}: OK — {message}")
            continue
        if args.check:
            print(f"{name}: MISSING — {message}")
            failures += 1
            continue

        print(f"{name}: fetching {entry['model_id']} …")
        ok, message = fetch(entry, kind)
        print(f"{name}: {'OK' if ok else 'FAILED'} — {message}")
        if not ok:
            failures += 1

    if failures:
        print(
            "\nSome models are unavailable. Phase 3 can still run with "
            "`backend: mock` in config/model.yaml, which exercises every code path "
            "but produces no empirical result.",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
