"""Selective prediction and calibration metrics.

A citation checker that says "I cannot tell" is doing something different from
one that guesses wrong, and the classification metrics in :mod:`.metrics` cannot
tell them apart: UNCERTAIN is excluded there, so a tool that abstains on
everything hard reports the metrics of its easy subset. This module scores the
abstention itself.

Three things live here.

**Risk-coverage.** Order predictions by how confident the tool is, reject the
least confident first, and plot error rate on what remains against how much
remains. A tool whose confidence carries information sheds error faster than it
sheds coverage, and the area under that curve (AURC) summarises it in one
number. AURC is only comparable across tools whose curves are defined over the
same coverage range, so :func:`risk_coverage_curve` reports the range it used
and :func:`compare_aurc` refuses to rank across disjoint domains.

**Calibration where a decision costs something.** An ECE averaged over a corpus
that is mostly VALID and mostly answered correctly hides the failure that
matters: confident wrong accusations. Measured on a 5,043-reference screening
run of real workshop submissions, the wrong HALLUCINATED verdicts sat at 0.93 to
0.99 confidence while the aggregate looked healthy. :func:`calibration_report`
therefore computes calibration on the flagged subset as well as overall.

**Brier, decomposed.** A single Brier score confounds "the tool is
miscalibrated" with "the tool cannot separate the classes". Murphy's
decomposition splits it into reliability (calibration error, lower is better)
and resolution (discrimination, higher is better) over an irreducible
uncertainty term, which says which of the two to go and fix.

Error fallbacks
---------------
Baselines write ``[Error fallback]`` into ``reason`` when an API call failed, and
those records carry label UNCERTAIN. They are not abstentions — no model decided
anything — so counting them as such flatters the tool's selective behaviour. They
are excluded from the curve by default and reported separately, never silently
dropped: a reader needs to know how much of an abstention rate was never a
decision at all. In the released v1.2 results these number 9,652, concentrated in
the crossdomain splits, with none in ``stress_test`` or ``hidden``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

from hallmark.dataset.schema import BenchmarkEntry, Prediction

ERROR_FALLBACK_MARKER = "[Error fallback]"


def is_error_fallback(pred: Prediction) -> bool:
    """True when a prediction records an API failure rather than a judgement."""
    return ERROR_FALLBACK_MARKER in (pred.reason or "")


def p_hallucinated(pred: Prediction) -> float:
    """Convert a prediction to P(entry is HALLUCINATED).

    ``confidence`` is P(the predicted label is correct), not P(HALLUCINATED) —
    a VALID prediction at 0.9 claims 90% certainty the entry is *valid*. Ranking
    by ``confidence`` directly would order a confident VALID and a confident
    HALLUCINATED together at the top, which produces a risk-coverage curve that
    looks well-behaved and means nothing. UNCERTAIN carries no signal and sits at
    0.5.
    """
    if pred.label == "HALLUCINATED":
        return float(pred.confidence)
    if pred.label == "VALID":
        return 1.0 - float(pred.confidence)
    return 0.5


def rejection_score(pred: Prediction) -> float:
    """How confident the tool is in its own call, regardless of direction.

    Selective prediction rejects the least confident predictions first. That
    ordering is by certainty, not by class, so both a confident VALID and a
    confident HALLUCINATED are retained and an UNCERTAIN is shed first.
    """
    return abs(p_hallucinated(pred) - 0.5) * 2.0


@dataclass(frozen=True)
class RiskCoveragePoint:
    coverage: float
    risk: float
    n_covered: int
    n_errors: int


@dataclass(frozen=True)
class RiskCoverageCurve:
    """A risk-coverage curve and the domain it is defined over.

    ``aurc`` is the area under the curve, integrated over ``coverage_range``.
    Comparing AURC between tools whose ranges differ compares different
    integrals; ``coverage_range`` is carried so a caller can check rather than
    assume.
    """

    points: list[RiskCoveragePoint]
    aurc: float
    coverage_range: tuple[float, float]
    n_scored: int
    n_error_fallbacks: int
    n_missing: int

    @property
    def risk_at_full_coverage(self) -> float:
        """Error rate with nothing rejected — the tool's plain accuracy complement."""
        return self.points[-1].risk if self.points else 0.0


def _paired(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
) -> tuple[list[tuple[BenchmarkEntry, Prediction]], int]:
    """Pair entries with their predictions; also count entries with none."""
    if isinstance(predictions, list):
        pairs = [(e, p) for e, p in zip(entries, predictions, strict=False)]
        return pairs, max(0, len(entries) - len(pairs))
    pairs = []
    missing = 0
    for entry in entries:
        pred = predictions.get(entry.bibtex_key)
        if pred is None:
            missing += 1
            continue
        pairs.append((entry, pred))
    return pairs, missing


def risk_coverage_curve(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
    *,
    exclude_error_fallbacks: bool = True,
) -> RiskCoverageCurve:
    """Risk as a function of coverage, rejecting least-confident predictions first.

    Risk is the error rate on the retained set, where an UNCERTAIN prediction on
    a retained entry counts as an error: at full coverage the tool has to commit,
    and declining to is not a correct answer.

    Missing predictions are excluded rather than treated as VALID. The
    conservative default elsewhere in this package is right for classification
    metrics, but here it would credit a tool for entries it never processed.

    Args:
        entries: Benchmark entries (ground truth).
        predictions: Tool predictions, keyed by bibtex_key or parallel to entries.
        exclude_error_fallbacks: Drop records marking an API failure. These are
            not abstentions and counting them as such flatters the curve.
    """
    pairs, missing = _paired(entries, predictions)

    n_fallback = sum(1 for _, p in pairs if is_error_fallback(p))
    if exclude_error_fallbacks:
        pairs = [(e, p) for e, p in pairs if not is_error_fallback(p)]

    if not pairs:
        return RiskCoverageCurve([], 0.0, (0.0, 0.0), 0, n_fallback, missing)

    scored = sorted(pairs, key=lambda ep: rejection_score(ep[1]), reverse=True)

    # Coverage is a share of the ENTRY SET, not of the predictions the tool
    # happened to return. Normalising to the scored set would let a tool that
    # answered 68 of 500 entries report coverage 1.0, which is the incomparability
    # `compare_aurc` exists to catch — and it would hide it at the source.
    denom = len(entries) if entries else len(scored)

    points: list[RiskCoveragePoint] = []
    errors = 0
    for i, (entry, pred) in enumerate(scored, start=1):
        if pred.label != entry.label:
            errors += 1
        points.append(
            RiskCoveragePoint(coverage=i / denom, risk=errors / i, n_covered=i, n_errors=errors)
        )

    # Trapezoidal integration over coverage. The first point sits at 1/n rather
    # than 0, so the integral runs over the range actually observed.
    aurc = 0.0
    for a, b in pairwise(points):
        aurc += (b.coverage - a.coverage) * (a.risk + b.risk) / 2.0
    span = points[-1].coverage - points[0].coverage
    aurc = aurc / span if span > 0 else points[0].risk

    return RiskCoverageCurve(
        points=points,
        aurc=aurc,
        coverage_range=(points[0].coverage, points[-1].coverage),
        n_scored=len(scored),
        n_error_fallbacks=n_fallback,
        n_missing=missing,
    )


def compare_aurc(curves: dict[str, RiskCoverageCurve], *, min_overlap: float = 0.5) -> dict:
    """Rank tools by AURC, refusing where the coverage domains do not overlap.

    A tool answering 68 of 500 entries has a curve over a different domain from
    one answering all 500; their AURCs are not the same integral. Where the
    shared coverage range is narrower than ``min_overlap`` this returns the
    ranking as ``None`` and says why, rather than producing a number that invites
    a comparison it cannot support.
    """
    usable = {k: c for k, c in curves.items() if c.points}
    if not usable:
        return {"ranking": None, "reason": "no curve had any scored prediction"}

    lo = max(c.coverage_range[0] for c in usable.values())
    hi = min(c.coverage_range[1] for c in usable.values())
    overlap = max(0.0, hi - lo)
    if overlap < min_overlap:
        return {
            "ranking": None,
            "reason": (
                f"coverage domains overlap on only {overlap:.1%} of the range "
                f"(need {min_overlap:.0%}); AURCs are not comparable"
            ),
            "coverage_ranges": {k: c.coverage_range for k, c in usable.items()},
        }
    return {
        "ranking": sorted(usable, key=lambda k: usable[k].aurc),
        "aurc": {k: c.aurc for k, c in usable.items()},
        "shared_coverage": (lo, hi),
    }


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return self.accuracy - self.mean_confidence


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy's decomposition: ``brier = reliability - resolution + uncertainty``.

    Reliability is calibration error and should be small. Resolution is
    discrimination and should be large. Uncertainty is a property of the split,
    not of the tool, and is the Brier score a constant base-rate predictor would
    achieve.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float

    @property
    def skill(self) -> float:
        """Brier skill score against the base-rate predictor; 1.0 is perfect."""
        return 1.0 - self.brier / self.uncertainty if self.uncertainty > 0 else 0.0


def _bins(pairs: list[tuple[float, bool]], n_bins: int) -> list[ReliabilityBin]:
    """Equal-frequency bins, matching the adaptive default of ``expected_calibration_error``."""
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda t: t[0])
    n = len(ordered)
    size = max(1, n // n_bins)
    out: list[ReliabilityBin] = []
    for start in range(0, n, size):
        chunk = ordered[start : start + size]
        if not chunk:
            continue
        confs = [c for c, _ in chunk]
        out.append(
            ReliabilityBin(
                lo=min(confs),
                hi=max(confs),
                n=len(chunk),
                mean_confidence=sum(confs) / len(chunk),
                accuracy=sum(1 for _, ok in chunk if ok) / len(chunk),
            )
        )
    return out


def brier_decomposition(pairs: list[tuple[float, bool]], n_bins: int = 10) -> BrierDecomposition:
    """Decompose the Brier score over (P(hallucinated), is_hallucinated) pairs."""
    if not pairs:
        return BrierDecomposition(0.0, 0.0, 0.0, 0.0)
    n = len(pairs)
    base = sum(1 for _, y in pairs if y) / n
    brier = sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / n
    uncertainty = base * (1.0 - base)

    reliability = 0.0
    resolution = 0.0
    for b in _bins(pairs, n_bins):
        w = b.n / n
        reliability += w * (b.mean_confidence - b.accuracy) ** 2
        resolution += w * (b.accuracy - base) ** 2
    return BrierDecomposition(brier, reliability, resolution, uncertainty)


@dataclass
class CalibrationReport:
    """Calibration overall, on flagged predictions, and per difficulty tier."""

    overall: BrierDecomposition
    overall_bins: list[ReliabilityBin]
    flagged: BrierDecomposition
    flagged_bins: list[ReliabilityBin]
    per_tier: dict[int, BrierDecomposition] = field(default_factory=dict)
    per_tier_bins: dict[int, list[ReliabilityBin]] = field(default_factory=dict)
    n_flagged: int = 0
    n_error_fallbacks: int = 0


def calibration_report(
    entries: list[BenchmarkEntry],
    predictions: dict[str, Prediction] | list[Prediction],
    *,
    n_bins: int = 10,
    exclude_error_fallbacks: bool = True,
) -> CalibrationReport:
    """Reliability and Brier decomposition, overall and where a decision costs something.

    The ``flagged`` figures cover HALLUCINATED predictions only. That is the
    population a user acts on — every one is an accusation against a real
    author — and it is where miscalibration is expensive. An aggregate over a
    mostly-VALID corpus averages it away.

    Per-tier figures use ``difficulty_tier``. VALID entries carry no tier and are
    included in every tier's calibration, since a tool's false-positive
    behaviour is part of its calibration at any difficulty.
    """
    pairs_all, _ = _paired(entries, predictions)
    n_fallback = sum(1 for _, p in pairs_all if is_error_fallback(p))
    if exclude_error_fallbacks:
        pairs_all = [(e, p) for e, p in pairs_all if not is_error_fallback(p)]

    scored = [(e, p) for e, p in pairs_all if p.label != "UNCERTAIN"]

    def to_pairs(items: list[tuple[BenchmarkEntry, Prediction]]) -> list[tuple[float, bool]]:
        return [(p_hallucinated(p), e.label == "HALLUCINATED") for e, p in items]

    flagged = [(e, p) for e, p in scored if p.label == "HALLUCINATED"]

    per_tier: dict[int, BrierDecomposition] = {}
    per_tier_bins: dict[int, list[ReliabilityBin]] = {}
    for tier in (1, 2, 3):
        subset = [(e, p) for e, p in scored if e.difficulty_tier == tier or e.label == "VALID"]
        if subset:
            per_tier[tier] = brier_decomposition(to_pairs(subset), n_bins)
            per_tier_bins[tier] = _bins(to_pairs(subset), n_bins)

    return CalibrationReport(
        overall=brier_decomposition(to_pairs(scored), n_bins),
        overall_bins=_bins(to_pairs(scored), n_bins),
        flagged=brier_decomposition(to_pairs(flagged), n_bins),
        flagged_bins=_bins(to_pairs(flagged), n_bins),
        per_tier=per_tier,
        per_tier_bins=per_tier_bins,
        n_flagged=len(flagged),
        n_error_fallbacks=n_fallback,
    )


def abstention_breakdown(
    predictions: dict[str, Prediction] | list[Prediction],
) -> dict[str, int]:
    """Split UNCERTAIN into genuine abstentions and recorded API failures.

    ``num_uncertain`` alone is not interpretable: an infrastructure failure and a
    model declining to rule are different events written to the same field.
    """
    preds = list(predictions.values()) if isinstance(predictions, dict) else list(predictions)
    uncertain = [p for p in preds if p.label == "UNCERTAIN"]
    fallbacks = sum(1 for p in uncertain if is_error_fallback(p))
    return {
        "n_predictions": len(preds),
        "n_uncertain": len(uncertain),
        "n_error_fallbacks": fallbacks,
        "n_genuine_abstentions": len(uncertain) - fallbacks,
    }


def format_reliability_diagram(bins: list[ReliabilityBin], width: int = 40) -> str:
    """Render a reliability diagram as text, for a report or a terminal.

    Each row shows a confidence bin, its mean confidence against the accuracy
    observed in it, and the gap. A well-calibrated tool has gaps near zero; a
    positive gap means underconfidence and a negative one overconfidence.
    """
    if not bins:
        return "(no scored predictions)"
    lines = [f"{'confidence':>18}  {'n':>5}  {'conf':>6}  {'acc':>6}  {'gap':>7}  diagram"]
    for b in bins:
        filled = round(b.accuracy * width)
        marker = round(b.mean_confidence * width)
        bar = list("·" * width)
        for i in range(min(filled, width)):
            bar[i] = "█"
        if 0 <= marker < width:
            bar[marker] = "|"
        sign = "+" if b.gap >= 0 else ""
        lines.append(
            f"{b.lo:>8.3f}-{b.hi:<8.3f} {b.n:>5}  {b.mean_confidence:>6.3f}  "
            f"{b.accuracy:>6.3f}  {sign}{b.gap:>6.3f}  {''.join(bar)}"
        )
    return "\n".join(lines)


def format_risk_coverage(curve: RiskCoverageCurve, n_rows: int = 10) -> str:
    """Render a risk-coverage curve as a small table, sampled evenly."""
    if not curve.points:
        return "(no scored predictions)"
    step = max(1, len(curve.points) // n_rows)
    rows = curve.points[step - 1 :: step]
    if rows[-1] is not curve.points[-1]:
        rows = [*rows, curve.points[-1]]
    lines = [
        f"AURC {curve.aurc:.4f} over coverage "
        f"{curve.coverage_range[0]:.3f}-{curve.coverage_range[1]:.3f} "
        f"({curve.n_scored} scored"
        + (
            f", {curve.n_error_fallbacks} error fallbacks excluded"
            if curve.n_error_fallbacks
            else ""
        )
        + (f", {curve.n_missing} missing" if curve.n_missing else "")
        + ")",
        f"{'coverage':>9}  {'risk':>7}  {'covered':>8}  {'errors':>7}",
    ]
    for p in rows:
        lines.append(f"{p.coverage:>9.3f}  {p.risk:>7.4f}  {p.n_covered:>8}  {p.n_errors:>7}")
    return "\n".join(lines)


__all__ = [
    "ERROR_FALLBACK_MARKER",
    "BrierDecomposition",
    "CalibrationReport",
    "ReliabilityBin",
    "RiskCoverageCurve",
    "RiskCoveragePoint",
    "abstention_breakdown",
    "brier_decomposition",
    "calibration_report",
    "compare_aurc",
    "format_reliability_diagram",
    "format_risk_coverage",
    "is_error_fallback",
    "p_hallucinated",
    "rejection_score",
    "risk_coverage_curve",
]
