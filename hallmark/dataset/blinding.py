"""Single source of truth for the fields a verifier must never see.

Dispatch-time blinding used to live in exactly one method
(:meth:`hallmark.dataset.schema.BenchmarkEntry.to_blind`) that every runner
was *expected* to call. A runner that read the corpus JSONL directly and
serialized the record verbatim skipped that method and shipped the ``url``
field into the prompt — which is what happened for a handful of checkpoints
produced by ``scripts/parallel_resume_test_public.py``.

The bug class is that blinding was conventional rather than structural. This
module fixes that:

* :data:`BLIND_EXCLUDED_FIELDS` declares the blind-list **once**, instead of a
  literal ``fields.pop("url")`` repeated at each call site.
* :class:`hallmark.dataset.schema.BlindEntry` strips the blind-list on
  construction, so a ``BlindEntry`` carrying a blinded field cannot exist and
  ``to_blind()`` is no longer something a caller can forget.
* Dict-level runners (the ad-hoc parallel/JSONL scripts that never build a
  ``BlindEntry``) get :func:`blind_record` to blind and :func:`assert_blinded`
  to fail loudly at the serialization boundary.

``raw_bibtex`` is covered too: it is forwarded verbatim by ``to_blind()`` and
preferred over the field dict by ``to_bibtex()``, so an unscrubbed
``raw_bibtex`` would smuggle a blinded field straight into the prompt even
when ``fields`` is clean. :func:`scrub_bibtex` removes blinded field
assignments from a raw BibTeX string, and returns ``None`` when it cannot
guarantee the result is clean (callers then fall back to rendering from the
already-blinded field dict).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# ---------------------------------------------------------------------------
# The blind-list
# ---------------------------------------------------------------------------

#: BibTeX fields withheld from every verifier at dispatch time.
#:
#: ``url`` is withheld because resolving it hands a tool the source record's
#: metadata for free: a hallucinated entry that inherited its source paper's
#: URL could be detected by fetching the URL and diffing the returned
#: metadata, which measures URL resolution rather than citation verification.
BLIND_EXCLUDED_FIELDS: frozenset[str] = frozenset({"url"})


class BlindingViolationError(RuntimeError):
    """Raised when an entry about to reach a verifier still carries a blinded field."""


# ---------------------------------------------------------------------------
# Field-dict and raw-BibTeX blinding
# ---------------------------------------------------------------------------


def blind_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``fields`` with every blinded field removed."""
    return {k: v for k, v in fields.items() if k not in BLIND_EXCLUDED_FIELDS}


def _end_of_value(text: str, i: int) -> int | None:
    """Index just past the BibTeX value starting at ``text[i]``.

    Handles brace-delimited, quote-delimited, and bare values. Returns
    ``None`` when the value is unterminated, so callers can treat the string
    as unparsable rather than guess.
    """
    if i >= len(text):
        return None
    ch = text[i]
    if ch == "{":
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return j + 1
        return None
    if ch == '"':
        for j in range(i + 1, len(text)):
            if text[j] == '"' and text[j - 1] != "\\":
                return j + 1
        return None
    for j in range(i, len(text)):
        if text[j] in ",\n}":
            return j
    return len(text)


def _assignment_re(key: str) -> re.Pattern[str]:
    """Match a BibTeX field assignment for ``key`` at the start of a line."""
    return re.compile(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*")


def _remove_assignment(text: str, key: str) -> str:
    """Remove every ``key = <value>`` assignment from a raw BibTeX string."""
    pattern = _assignment_re(key)
    while True:
        match = pattern.search(text)
        if match is None:
            return text
        end = _end_of_value(text, match.end())
        if end is None:
            return text  # unparsable; scrub_bibtex nulls the string instead
        j = end
        while j < len(text) and text[j] in " \t":
            j += 1
        if j < len(text) and text[j] == ",":
            j += 1
        while j < len(text) and text[j] in " \t":
            j += 1
        if j < len(text) and text[j] == "\n":
            j += 1
        text = text[: match.start()] + text[j:]


_FIELD_KEY_RE = re.compile(r"(?im)^[ \t]*([A-Za-z][A-Za-z0-9_+-]*)[ \t]*=")


def _field_keys(text: str) -> set[str]:
    return {m.group(1).lower() for m in _FIELD_KEY_RE.finditer(text)}


def scrub_bibtex(raw: str | None) -> str | None:
    """Strip blinded field assignments from a raw BibTeX string.

    Returns ``None`` rather than a string the caller cannot trust — when a
    blinded assignment survives, or when scrubbing an unterminated value ate a
    neighbouring field. The caller then renders from the blinded field dict,
    which is complete and already clean.
    """
    if not raw:
        return raw
    out = raw
    for key in sorted(BLIND_EXCLUDED_FIELDS):
        out = _remove_assignment(out, key)
    if any(_assignment_re(key).search(out) for key in BLIND_EXCLUDED_FIELDS):
        return None
    expected = _field_keys(raw) - {k.lower() for k in BLIND_EXCLUDED_FIELDS}
    if expected - _field_keys(out):
        return None
    return out


# ---------------------------------------------------------------------------
# Dict-level (JSONL) runners
# ---------------------------------------------------------------------------


def blind_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Blind a corpus JSONL record in place of ``BenchmarkEntry.to_blind()``.

    For runners that read the corpus JSONL directly and never build a
    :class:`~hallmark.dataset.schema.BlindEntry`. Blinds ``fields``, scrubs
    ``raw_bibtex``, and leaves everything else untouched — dropping the
    ground-truth keys stays the caller's business.
    """
    out = dict(record)
    fields = out.get("fields")
    if isinstance(fields, Mapping):
        out["fields"] = blind_fields(fields)
    if "raw_bibtex" in out:
        out["raw_bibtex"] = scrub_bibtex(out["raw_bibtex"])
    return out


def find_blind_violations(obj: Any) -> list[str]:
    """Blinded fields still reachable from ``obj``, sorted.

    Accepts a corpus record / prediction payload (mapping with ``fields``), a
    bare field mapping, or any object exposing ``fields`` and ``raw_bibtex``
    (``BenchmarkEntry``, ``BlindEntry``). A violation in ``raw_bibtex`` is
    reported as ``"<field> (raw_bibtex)"``.
    """
    fields: Mapping[str, Any]
    raw: Any
    if isinstance(obj, Mapping):
        nested = obj.get("fields")
        fields = nested if isinstance(nested, Mapping) else obj
        raw = obj.get("raw_bibtex")
    else:
        fields = getattr(obj, "fields", None) or {}
        raw = getattr(obj, "raw_bibtex", None)

    found = [k for k in sorted(BLIND_EXCLUDED_FIELDS) if k in fields]
    if isinstance(raw, str) and raw:
        found += [
            f"{k} (raw_bibtex)"
            for k in sorted(BLIND_EXCLUDED_FIELDS)
            if _assignment_re(k).search(raw)
        ]
    return found


def assert_blinded(obj: Any, *, context: str = "") -> None:
    """Raise :class:`BlindingViolationError` if ``obj`` still carries a blinded field.

    Call this at the boundary where an entry becomes verifier input — the
    prompt renderer — so a runner that skipped blinding fails loudly instead
    of silently leaking.
    """
    violations = find_blind_violations(obj)
    if not violations:
        return
    key = ""
    if isinstance(obj, Mapping):
        key = str(obj.get("bibtex_key", ""))
    else:
        key = str(getattr(obj, "bibtex_key", ""))
    where = f" in {context}" if context else ""
    subject = f" (entry {key})" if key else ""
    raise BlindingViolationError(
        f"Entry reached a verifier{where} still carrying blinded "
        f"field(s) {violations}{subject}. Blind it via BenchmarkEntry.to_blind(), "
        f"BlindEntry(...), or hallmark.dataset.blinding.blind_record() first."
    )
