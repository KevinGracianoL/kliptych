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

    def evaluate_piece(
        self,
        piece: Piece,
        *,
        contract: Contract,
        assets: AssetRegistry | None = None,
    ) -> GateResult:
        """Evalúa una pieza contra el contrato usando el Gate.

        Args:
            piece: Pieza final a validar (artefacto más textos).
            contract: Contrato validado que declara las reglas.
            assets: Registro opcional de assets del workspace. Si es None,
                se crea a partir del directorio del artefacto.

        Returns:
            El resultado del gate, con un check por regla declarada.
        """
        effective_assets = (
            assets if assets is not None else AssetRegistry(piece.artifact_path.parent)
        )
        return self.run(contract=contract, piece=piece, assets=effective_assets)

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
            citations = [
                unmapped.quote for unmapped in context.contract.unmapped if unmapped.rule == rule_id
            ]
            status = (
                CheckStatus.MANUAL_REVIEW
                if strength is RuleStrength.MANUAL_REVIEW or bool(citations)
                else CheckStatus.UNSUPPORTED
            )
            evidence: dict[str, object] = {
                "reason": "no hay validador mecánico registrado para esta regla"
            }
            if citations:
                evidence["citations"] = citations
            return CheckResult(id=rule_id, status=status, evidence=evidence)
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
    checks = [check for _, check in outcomes]
    if any(check.status is CheckStatus.FAIL for check in checks):
        return GateStatus.REJECTED
    hard = [check for strength, check in outcomes if strength is RuleStrength.HARD]
    if any(check.status is CheckStatus.UNSUPPORTED for check in hard):
        return GateStatus.UNSUPPORTED
    if any(
        strength is RuleStrength.MANUAL_REVIEW or check.status is CheckStatus.MANUAL_REVIEW
        for strength, check in outcomes
    ):
        return GateStatus.PENDING_REVIEW
    return GateStatus.PASSED
