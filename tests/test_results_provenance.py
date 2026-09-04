"""Guard the provenance of baseline result files.

Two directories once held two different runs under one set of names:
``results/`` and ``data/v1.2/baseline_results/`` shared fifteen filenames and
**every pair disagreed** on detection rate, false-positive rate and F1, three of
them on the number of entries scored as well. Nothing compared them, nothing
recorded which was current, and ``results/`` is the default ``--results-dir``
for the CLI leaderboard and the figure scripts — so the uncanonical set was the
one being read.

A reproducer hits that before anything else, so these tests make the collision
impossible rather than documenting it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _REPO_ROOT / "results"
_SUPERSEDED = _RESULTS / "superseded_pre_relabel"

#: Comparable summary fields. A disagreement on any of these means the two files
#: describe different runs, whatever their names say.
_METRIC_KEYS = ("detection_rate", "false_positive_rate", "f1_hallucination", "num_entries")

#: Results deliberately withdrawn from the released set, with the reason. An
#: archived file whose name no longer exists among the released results is
#: normally an accident -- a result retired without a replacement -- so a
#: deliberate withdrawal has to be written down here to be distinguishable from
#: one. Adding a name is a claim that the release is better without it.
WITHDRAWN: dict[str, str] = {
    "harc_dev_public.json": (
        "2026-09-04: scored 521 of 1,119 entries, matching no split, and reported "
        "FPR 0.000 with no coverage field — a truncated run, not a result. "
        "harc_with_s2key_dev_public.json is the valid HaRC evaluation "
        "(n=1,119, coverage 1.0, DR 0.209, FPR 0.045)."
    ),
}


def _baseline_result_dirs() -> list[Path]:
    return sorted(p for p in _REPO_ROOT.glob("data/*/baseline_results") if p.is_dir())


def _canonical_names() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in _baseline_result_dirs():
        for p in d.glob("*.json"):
            if p.name != "manifest.json":
                out[p.name] = p
    return out


def test_at_least_one_baseline_results_dir_exists():
    """Otherwise the collision tests below would pass vacuously."""
    assert _baseline_result_dirs(), "no data/*/baseline_results directory found"


def test_no_result_filename_collides_with_a_released_one():
    """A name may live in exactly one place.

    Superseded runs belong under ``results/superseded_pre_relabel/``, which is
    excluded here precisely because it is the archive for the old collisions.
    """
    canonical = _canonical_names()
    collisions = sorted(p.name for p in _RESULTS.glob("*.json") if p.name in canonical)
    assert not collisions, (
        f"{len(collisions)} result file(s) share a name with a released result and will "
        f"be read instead of it by anything defaulting to --results-dir results: "
        f"{collisions}. Move them to results/superseded_pre_relabel/ or delete them."
    )


def test_superseded_files_are_not_also_canonical():
    """The archive must not shadow a live name either."""
    if not _SUPERSEDED.is_dir():
        pytest.skip("no superseded archive present")
    canonical = _canonical_names()
    # Files here are *expected* to share names with canonical ones — that is why
    # they were archived. What must not happen is the reverse: an archived file
    # being the only copy of a name, which would mean a released result was
    # retired by accident.
    orphans = sorted(
        p.name
        for p in _SUPERSEDED.glob("*.json")
        if p.name not in canonical and p.name not in WITHDRAWN
    )
    assert not orphans, (
        f"archived result(s) with no canonical counterpart: {orphans}. These were "
        "retired without a replacement. Restore them, or add the name to WITHDRAWN "
        "with the reason the release is better without it."
    )


def test_withdrawn_entries_are_actually_gone():
    """A withdrawal that never happened is a false record — worse than none."""
    canonical = _canonical_names()
    still_present = sorted(name for name in WITHDRAWN if name in canonical)
    assert not still_present, (
        f"listed as withdrawn but still in the released set: {still_present}. "
        "Either remove the file or drop it from WITHDRAWN."
    )


def test_archive_documents_itself():
    if not _SUPERSEDED.is_dir():
        pytest.skip("no superseded archive present")
    readme = _SUPERSEDED / "README.md"
    assert readme.is_file(), "an archive of superseded results must say what it is"
    assert "canonical" in readme.read_text().lower()


def _is_null_run(data: dict) -> bool:
    """A run that detected nothing, flagged nothing and made no API call.

    That is the signature of ``fallback_predictions`` firing because an external
    CLI was missing: the wrapper returns all-VALID for every entry and the
    harness scores it like any other result. Four such files shipped as the
    pre-screening ablations for bibtexupdater and harc.

    **The conjunction is load-bearing, not belt-and-braces.** ``mean_api_calls``
    of 0.0 is uninformative rather than diagnostic for every HaRC row, because
    that wrapper never records API calls at all:
    ``harc_with_s2key_dev_public.json`` reports 0.0 while being a real
    evaluation at full coverage with DR 0.209. Requiring a zero detection rate
    AND a zero false-positive rate alongside it is the only thing standing
    between this guard and a genuine result.
    """
    return (
        data.get("detection_rate") == 0.0
        and data.get("false_positive_rate") == 0.0
        and (data.get("mean_api_calls") or 0.0) == 0.0
        and (data.get("num_entries") or 0) > 0
    )


#: Baselines whose null shape is the point rather than a failure. ``always_valid``
#: predicts VALID for every entry and queries nothing, so it produces DR 0.0,
#: FPR 0.0 and zero API calls BY DESIGN -- it is the degenerate floor the other
#: baselines are measured against, and excluding it would delete the reference
#: point that makes the rest interpretable.
#:
#: This is why the shape below is a quarantine gate and not a classifier: it says
#: "a null-shaped result may not sit among real ones", and the fix is to move it
#: or re-run it, never to reclassify it silently. Distinguishing a tool that
#: evaluated everything and found nothing from one that evaluated nothing needs a
#: recorded flag, not a signature.
DEGENERATE_BASELINES = frozenset({"always_valid", "doi_presence_heuristic"})


def test_no_result_outside_failed_runs_is_a_null_run():
    """A failed run must be quarantined, not left where it reads as a measurement."""
    offenders: list[str] = []
    searched = 0
    for path in _REPO_ROOT.glob("data/*/baseline_results/*.json"):
        if path.name == "manifest.json":
            continue
        searched += 1
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or not _is_null_run(data):
            continue
        tool = str(data.get("tool_name") or path.stem)
        if any(name in tool for name in DEGENERATE_BASELINES):
            continue  # null by design, not by failure
        offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert searched > 0, "no released results scanned — the guard would pass vacuously"
    assert not offenders, (
        f"null run(s) sitting among real results: {offenders}. DR=0, FPR=0 and no API "
        "calls over a full split means the wrapper fell back to all-VALID because its "
        "CLI was missing. Move them to results/failed_runs/ or re-run them."
    )


@pytest.mark.parametrize("results_dir", _baseline_result_dirs(), ids=lambda p: p.parent.name)
def test_released_results_are_internally_consistent(results_dir: Path):
    """Every released result must at least parse and report the fields it claims."""
    bad: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            bad.append(f"{path.name}: unparseable ({exc})")
            continue
        if not isinstance(data, dict):
            bad.append(f"{path.name}: not an object")
            continue
        if data.get("tool_name") is None:
            bad.append(f"{path.name}: no tool_name")
        for key in ("detection_rate", "num_entries"):
            if data.get(key) is None:
                bad.append(f"{path.name}: no {key}")
    assert not bad, "malformed released results: " + "; ".join(bad)


def test_every_registered_runner_accepts_the_kwargs_the_registry_passes():
    """A baseline that rejects an unexpected kwarg cannot be run at all.

    ``run_baseline`` forwards ``split=`` to every runner. ``run_doi_only`` had
    no ``**_kw`` catch-all, so ``hallmark evaluate --baseline doi_only`` raised
    TypeError before making a single request -- which is a plausible reason its
    released result sat stale at 1,068 of 1,119 entries: it could not be re-run.
    """
    import inspect

    from hallmark.baselines.registry import get_registry

    offenders: list[str] = []
    for name, info in sorted(get_registry().items()):
        try:
            sig = inspect.signature(info.runner)
        except (TypeError, ValueError):  # builtins / C callables
            continue
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            offenders.append(name)
    assert not offenders, (
        f"runner(s) without a **kwargs catch-all: {offenders}. run_baseline forwards "
        "split= to every runner, so these raise TypeError instead of running."
    )


class TestBibtexCheckBinaryPinning:
    """Which bibtex-check build answers must be decidable and knowable.

    The wrapper invoked ``bibtex-check`` by name, so PATH order chose the build.
    On this machine the first entry is an editable install importing from the
    bibtexupdater working tree: it reports 1.3.1.dev18 while the pipx copy on the
    same PATH is 1.10.1. An ablation ran three hours against the checkout while
    two commits landed in it, so entries before and after were scored by
    different code and nothing in the output said so.
    """

    def test_env_var_pins_the_binary(self, monkeypatch, tmp_path):
        from hallmark.baselines import bibtexupdater as btu

        fake = tmp_path / "bibtex-check"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv(btu.BIBTEX_CHECK_BIN_ENV, str(fake))
        assert btu.resolve_bibtex_check_bin() == str(fake)

    def test_missing_pinned_binary_is_reported_not_silently_ignored(self, monkeypatch, tmp_path):
        """Falling back to PATH would run a different build than the operator asked for."""
        from hallmark.baselines import bibtexupdater as btu

        monkeypatch.setenv(btu.BIBTEX_CHECK_BIN_ENV, str(tmp_path / "nope"))
        assert btu.resolve_bibtex_check_bin() is None

    def test_falls_back_to_path_when_unpinned(self, monkeypatch):
        from hallmark.baselines import bibtexupdater as btu

        monkeypatch.delenv(btu.BIBTEX_CHECK_BIN_ENV, raising=False)
        monkeypatch.setattr(btu.shutil, "which", lambda _n: "/usr/bin/bibtex-check")
        assert btu.resolve_bibtex_check_bin() == "/usr/bin/bibtex-check"

    def test_version_probe_never_raises(self, monkeypatch, tmp_path):
        """Provenance must not be able to break an evaluation."""
        from hallmark.baselines import bibtexupdater as btu

        assert btu.bibtex_check_version(str(tmp_path / "absent")) is None
