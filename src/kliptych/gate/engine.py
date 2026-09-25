"""Motor del gate: ejecuta validadores y deriva el estado fail-closed."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from kliptych.assets import AssetRegistry
from kliptych.contract import Contract, RuleStrength, contract_digest
from kliptych.gate.checks import DEFAULT_VALIDATORS, GateContext, Validator
from kliptych.gate.models import (
    CheckResult,
    CheckStatus,
    GateResult,
    GateStatus,
    MediaInfo,
    Piece,
)
from kliptych.gate.probe import MediaProbe, ProbeError
from kliptych.hashing import sha256_file


class GateError(Exception):
    """La pieza o el contrato no permiten ejecutar el gate."""


class Gate:
    """Ejecuta las reglas declaradas en el contrato sobre una pieza."""

    def __init__(
        self,
        probe: MediaProbe,
        *,
        validators: Mapping[str, Validator] | None = None,
    ) -> None:
        """Configura el probe y el catálogo de validadores.

        Args:
            probe: Inspección de medios del artefacto.
            validators: Catálogo rule_id -> validador; por defecto el base.
        """
        self._probe: MediaProbe = probe
        self._validators: dict[str, Validator] = dict(
            DEFAULT_VALIDATORS if validators is None else validators
        )

    def run(self, *, contract: Contract, piece: Piece, assets: AssetRegistry) -> GateResult:
        """Valida la pieza contra el contrato.

        Args:
            contract: Contrato validado que declara las reglas.
            piece: Pieza final a validar (artefacto más textos).
            assets: Registro de assets del workspace.

        Returns:
            El resultado del gate, con un check por regla declarada.

        Raises:
            GateError: Si la plataforma de la pieza no está en el contrato.
        """
        if piece.platform not in contract.platforms:
            msg = f"la plataforma '{piece.platform}' de la pieza no está declarada en el contrato"
            raise GateError(msg)
        artifact_sha256 = _hash_artifact(piece.artifact_path)
        media = self._probe_media(piece.artifact_path)
        context = GateContext(
            contract=contract,
            rules=contract.platforms[piece.platform],
            piece=piece,
            artifact_sha256=artifact_sha256,
            media=media,
            assets=assets,
        )
        outcomes: list[tuple[RuleStrength, CheckResult]] = []
        for strength, rule_ids in (
            (RuleStrength.HARD, contract.rules.hard),
            (RuleStrength.RECOMMENDED, contract.rules.recommended),
            (RuleStrength.MANUAL_REVIEW, contract.rules.manual_review),
        ):
            outcomes.extend(
                (strength, self._run_rule(rule_id, strength, context)) for rule_id in rule_ids
            )
        return GateResult(
            status=_derive_status(outcomes),
            checks=tuple(check for _, check in outcomes),
            artifact_sha256=artifact_sha256,
            contract_sha256=contract_digest(contract),
        )

    def _run_rule(self, rule_id: str, strength: RuleStrength, context: GateContext) -> CheckResult:
        validator = self._validators.get(rule_id)
        if validator is None:
            status = (
                CheckStatus.MANUAL_REVIEW
                if strength is RuleStrength.MANUAL_REVIEW
                else CheckStatus.UNSUPPORTED
            )
            reason = "no hay validador mecánico registrado para esta regla"
            return CheckResult(id=rule_id, status=status, evidence={"reason": reason})
        outcome = validator(context)
        return CheckResult(id=rule_id, status=outcome.status, evidence=outcome.evidence)

    def _probe_media(self, path: Path) -> MediaInfo | None:
        if not path.is_file():
            return None
        try:
            return self._probe.probe(path)
        except ProbeError:
            return None


def _hash_artifact(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return sha256_file(path)
    except OSError:
        return None


def _derive_status(outcomes: Sequence[tuple[RuleStrength, CheckResult]]) -> GateStatus:
    hard = [check for strength, check in outcomes if strength is RuleStrength.HARD]
    if any(check.status is CheckStatus.FAIL for check in hard):
        return GateStatus.REJECTED
    if any(check.status is CheckStatus.UNSUPPORTED for check in hard):
        return GateStatus.UNSUPPORTED
    if any(
        strength is RuleStrength.MANUAL_REVIEW or check.status is CheckStatus.MANUAL_REVIEW
        for strength, check in outcomes
    ):
        return GateStatus.PENDING_REVIEW
    return GateStatus.PASSED
