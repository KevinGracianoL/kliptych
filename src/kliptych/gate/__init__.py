"""Gate de compliance fail-closed sobre el artefacto final."""

from kliptych.gate.brand_safety import (
    BrandSafetyAssessment,
    BrandSafetyError,
    check_brand_safety,
    make_brand_safety_validator,
    make_model_assessor,
    parse_brand_safety_response,
)
from kliptych.gate.checks import DEFAULT_VALIDATORS, CheckOutcome, GateContext, Validator
from kliptych.gate.engine import Gate, GateError
from kliptych.gate.hook import check_hook_keyword
from kliptych.gate.models import (
    CheckResult,
    CheckStatus,
    GateResult,
    GateStatus,
    MediaInfo,
    Piece,
    SubtitleSegment,
)
from kliptych.gate.probe import FFprobeProbe, MediaProbe, ProbeError
from kliptych.gate.watermark import check_watermark_full_video, check_watermark_present

__all__ = [
    "DEFAULT_VALIDATORS",
    "BrandSafetyAssessment",
    "BrandSafetyError",
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
    "SubtitleSegment",
    "Validator",
    "check_brand_safety",
    "check_hook_keyword",
    "check_watermark_full_video",
    "check_watermark_present",
    "make_brand_safety_validator",
    "make_model_assessor",
    "parse_brand_safety_response",
]
