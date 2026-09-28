"""Gate de compliance fail-closed sobre el artefacto final."""

from kliptych.gate.checks import DEFAULT_VALIDATORS, CheckOutcome, GateContext, Validator
from kliptych.gate.engine import Gate, GateError
from kliptych.gate.models import (
    CheckResult,
    CheckStatus,
    GateResult,
    GateStatus,
    MediaInfo,
    Piece,
)
from kliptych.gate.probe import FFprobeProbe, MediaProbe, ProbeError
from kliptych.gate.watermark import check_watermark_full_video, check_watermark_present

__all__ = [
    "DEFAULT_VALIDATORS",
    "CheckOutcome",
    "CheckResult",
    "CheckStatus",
    "FFprobeProbe",
    "Gate",
    "GateContext",
    "GateError",
    "GateResult",
    "GateStatus",
    "MediaInfo",
    "MediaProbe",
    "Piece",
    "ProbeError",
    "Validator",
    "check_watermark_full_video",
    "check_watermark_present",
]
