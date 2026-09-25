"""Modelos del gate: estados, resultado y pieza a validar."""

from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from kliptych.contract import Platform


class _GateBase(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class CheckStatus(StrEnum):
    """Estado de un check individual del gate."""

    PASS = "pass"
    FAIL = "fail"
    MANUAL_REVIEW = "manual_review"
    UNSUPPORTED = "unsupported"


class GateStatus(StrEnum):
    """Estado global del gate para una pieza."""

    PASSED = "passed"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"
    PENDING_REVIEW = "pending_review"
    UNSUPPORTED = "unsupported"


class CheckResult(_GateBase):
    """Resultado de una regla concreta, con su evidencia mecánica."""

    id: str = Field(min_length=1)
    status: CheckStatus
    evidence: dict[str, object] = Field(default_factory=dict)


class GateResult(_GateBase):
    """Resultado completo del gate sobre una pieza."""

    status: GateStatus
    checks: tuple[CheckResult, ...]
    artifact_sha256: str | None
    contract_sha256: str

    @property
    def passed(self) -> bool:
        """Indica si la pieza quedó aprobada por el gate.

        Returns:
            ``True`` solo cuando el estado global es ``passed``.
        """
        return self.status is GateStatus.PASSED


class MediaInfo(_GateBase):
    """Metadatos del artefacto obtenidos con ffprobe.

    ``duration_s`` es ``None`` cuando ffprobe no reporta una duración
    medible; el gate debe tratar ese caso como no verificable, nunca como 0.
    """

    format_name: str
    duration_s: float | None = None
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None


class Piece(_GateBase):
    """Pieza lista para validar: el artefacto final más sus textos."""

    piece_id: str = Field(min_length=1, max_length=64)
    platform: Platform
    caption: str
    hashtags: tuple[str, ...] = ()
    subtitle_text: str | None = None
    artifact_path: Path
