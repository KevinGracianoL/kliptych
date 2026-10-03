"""Motor del gate: ejecuta validadores y deriva el estado fail-closed."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from kliptych.assets import AssetRegistry
from kliptych.contract import Contract, RuleStrength, contract_digest
from kliptych.contract.enums import AudioRule
from kliptych.contract.schema import PlatformRules, canonical_rule_id
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

# Predicados de aplicabilidad por plataforma, indexados por rule_id canonico.
# Es la UNICA tabla de aplicabilidad: la consultan el despacho de checks y el
# lookup de validadores, ambos del motor. El exportador NO la consulta: sus
# recordatorios son planos por informe y no pueden expresar el caso B, asi que
# siguen usando la lista global de manual_review (#57 abierto).
_RULE_APPLICABILITY: dict[str, Callable[[PlatformRules], bool]] = {
    "audio.present": lambda rules: rules.audio_rule is not AudioRule.ANY,
    "audio.own_clip": lambda rules: rules.audio_rule is AudioRule.OWN_CLIP,
    "audio.no_trending": lambda rules: rules.audio_rule is AudioRule.NO_TRENDING,
    "audio.official_track": lambda rules: rules.audio_rule is AudioRule.OFFICIAL_REQUIRED,
}


def rule_applies(rule_id: str, rules: PlatformRules) -> bool:
    """Indica si la regla tiene requisitos que verificar en esta plataforma.

    Los ``rule_id`` de ``contract.rules`` son globales: una sola declaracion
    cubre todas las plataformas del contrato. Su contenido, en cambio, es por
    plataforma, asi que la misma regla puede tener requisitos en una plataforma
    y en otra no. Este predicado es el unico lugar del motor donde se decide eso,
    y lo consultan el despacho de checks y el lookup de validadores. El
    exportador no lo consulta: sus recordatorios son planos por informe (#57).

    Ojo con la distincion que separa las dos situaciones:

    - Un conjunto de requisitos VACIO (por ejemplo ``caption.required_mention``
      sin menciones exigidas) NO es lo mismo que una regla NO APLICABLE. El
      check debe seguir ejecutandose y emitir su evidencia, vacia pero util,
      para que exista constancia de que se evaluo.
    - Una regla NO APLICABLE (por ejemplo ``audio.present`` sobre una
      plataforma con ``audio_rule=any``) no tiene nada que evaluar y no debe
      producir ningun outcome: hacerlo emitiria un estado, y un ``UNSUPPORTED``
      por regla dura deja la pieza inexportable sin salida humana.

    Las reglas ``audio.own_clip``, ``audio.no_trending`` y
    ``audio.official_track`` son ``manual_review`` y no tienen validador
    mecanico a proposito: ``manual_review`` significa que decide una persona, y
    eso no se automatiza. No tienen que emitir evidencia mecanica; lo que no
    pueden es exigir la revision manual de una plataforma que no declaro esa
    regla de audio.

    Generalizar esto a "omitir todo check sin requisitos" seria incorrecto y
    borraria la evidencia de los checks que si deben constancia. Solo se omite
    lo que tiene un predicado declarado aqui.

    Args:
        rule_id: Identificador de la regla declarada en el contrato.
        rules: Reglas de la plataforma de la pieza, ya resueltas.

    Returns:
        True si la regla aplica a esta plataforma.
    """
    predicate = _RULE_APPLICABILITY.get(canonical_rule_id(rule_id))
    return True if predicate is None else predicate(rules)


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

    def register_validator(self, rule_id: str, validator: Validator) -> None:
        """Registra o reemplaza un validador en el catálogo del gate."""
        self._validators[rule_id] = validator

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
            El resultado del gate, con un check por regla declarada que aplique
            a la plataforma de la pieza. Una regla declarada globalmente pero
            no aplicable a esta plataforma queda omitida, sin emitir outcome:
            ver ``rule_applies`` para el criterio y el motivo.

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
            for rule_id in rule_ids:
                if rule_applies(rule_id, context.rules):
                    # CASO A: la regla aplica a la plataforma de esta pieza.
                    outcomes.append((strength, self._run_rule(rule_id, strength, context)))
                elif rule_applies_anywhere(rule_id, contract):
                    # CASO B: no aplica aqui pero si en otra plataforma. Es regla
                    # de esa otra, y esta pieza no tiene que arrastrarla.
                    continue
                else:
                    # CASO C: declarada en el contrato y sin ninguna plataforma
                    # que la pida. El contrato es incoherente, y una declaracion
                    # incoherente NO se inventa: se despacha igual que antes del
                    # scoping, para que su resultado sea IDENTICO al de base.
                    # Omitirla en silencio seria un fail-open, y sustituirla por
                    # un UNSUPPORTED fijo tambien: `_derive_status` solo escala
                    # UNSUPPORTED cuando la fuerza es `hard`, asi que una regla
                    # recommended huerfana pasaria de `rejected` a `passed`.
                    # Se registra siempre; anadir el motivo es ademas informativo.
                    dispatched = self._run_rule(rule_id, strength, context)
                    outcomes.append(
                        (
                            strength,
                            CheckResult(
                                id=dispatched.id,
                                status=dispatched.status,
                                evidence={
                                    **dispatched.evidence,
                                    "unclaimed": (
                                        "ninguna plataforma del contrato declara esta regla"
                                    ),
                                },
                            ),
                        )
                    )
        if contract.unmapped:
            colliding = [
                entry.rule for entry in contract.unmapped if entry.rule in self._validators
            ]
            check_status = CheckStatus.FAIL if colliding else CheckStatus.MANUAL_REVIEW
            evidence: dict[str, object] = {
                "unmapped": [{"rule": u.rule, "quote": u.quote} for u in contract.unmapped],
                "rules": [u.rule for u in contract.unmapped],
                "citations": [u.quote for u in contract.unmapped],
            }
            if colliding:
                evidence["colliding_rules"] = colliding
            outcomes.append(
                (
                    RuleStrength.HARD if colliding else RuleStrength.MANUAL_REVIEW,
                    CheckResult(
                        id="rules.unmapped",
                        status=check_status,
                        evidence=evidence,
                    ),
                )
            )
        return GateResult(
            status=_derive_status(outcomes),
            checks=tuple(check for _, check in outcomes),
            artifact_sha256=artifact_sha256,
            contract_sha256=contract_digest(contract),
        )

    def _run_rule(self, rule_id: str, strength: RuleStrength, context: GateContext) -> CheckResult:
        validator = self._validators.get(canonical_rule_id(rule_id))
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


def rule_applies_anywhere(rule_id: str, contract: Contract) -> bool:
    """Indica si la regla aplica a ALGUNA plataforma del contrato.

    Es la contraparte de ``rule_applies``: aquella responde por una plataforma
    concreta, esta responde por el contrato entero. Sirve para distinguir una
    regla que es de otra plataforma (y que esta pieza no debe arrastrar) de una
    regla que no es de ninguna (y que por tanto es una declaracion incoherente
    que hay que registrar en vez de callar).

    Args:
        rule_id: Identificador de la regla declarada en el contrato.
        contract: Contrato completo, con todas sus plataformas.

    Returns:
        True si la regla aplica a al menos una plataforma del contrato.
    """
    return any(rule_applies(rule_id, rules) for rules in contract.platforms.values())


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
