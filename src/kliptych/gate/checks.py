"""Validadores deterministas del gate, indexados por rule_id.

Catálogo de reglas que el contrato puede declarar:

- ``artifact.integrity``: el artefacto existe y su hash es calculable.
- ``artifact.video_stream``: el artefacto tiene pista de video.
- ``assets.required``: los assets obligatorios están registrados, íntegros y
  coinciden con el hash y tamaño declarados en el contrato.
- ``audio.present``: el artefacto tiene pista de audio.
- ``audio.policy``: política de audio de la campaña; ``internal_official_sound``
  exige aprobación humana (``manual_review``, jamás ``pass`` ni
  ``unsupported``).
- ``audio.silence``: silencio digital real del artefacto con ``volumedetect``;
  solo aplica con ``internal_official_sound``. El gate no confía en que se
  pasara ``-af volume=0`` a ffmpeg: mide el ``max_volume`` del MP4 final
  (``<= -80.0`` dB pasa) y cualquier fallo de medición es ``fail``
  (fail-closed). Antes de medir, cuenta las pistas de audio con ffprobe:
  ffmpeg solo mide la pista por defecto, así que un MP4 con 0 o más de 1
  pista se rechaza sin medir; con exactamente 1 pista se mide con
  ``-map 0:a:0``.
- ``caption.first_line``: el caption abre con la primera línea exigida.
- ``caption.forbidden``: no aparecen términos prohibidos en el caption
  (union de ``caption_rules.forbidden`` y ``prohibitions`` de la campaña).
- ``caption.required_hashtag``: están los hashtags obligatorios, con frontera
  de token y comparación insensible a mayúsculas.
- ``caption.required_mention``: están las menciones obligatorias, con frontera
  de token y comparación insensible a mayúsculas.
- ``duration.min`` / ``duration.max``: duración dentro del rango; si no se
  pudo medir, el resultado es ``unsupported`` (jamás ``pass``).
- ``subtitles.spelling_lock``: spelling exacto en subtítulos; sin subtítulos
  y con locks declarados el resultado es ``unsupported``.
- ``watermark.full_video`` / ``watermark.present``: watermark exigido durante
  todo el video o en alguna parte; se verifica con OpenCV frame por frame
  (``cv2.matchTemplate`` contra el PNG del contrato, con zona y tamaño
  mínimos) y cualquier fallo es ``fail`` (fail-closed).

Reglas declaradas sin validador registrado jamás pasan: el motor las marca
``unsupported`` (o ``manual_review`` si el contrato las clasificó así).
"""

import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from kliptych.assets import AssetError, AssetNotFoundError
from kliptych.contract import AudioPolicy, AudioRule, Format
from kliptych.gate.models import CheckOutcome, CheckStatus, GateContext
from kliptych.gate.watermark import check_watermark_full_video, check_watermark_present

__all__ = [
    "DEFAULT_VALIDATORS",
    "CheckOutcome",
    "GateContext",
    "Validator",
    "check_artifact_integrity",
    "check_audio_policy",
    "check_audio_present",
    "check_audio_silence",
    "check_duration_max",
    "check_duration_min",
    "check_first_line",
    "check_forbidden_terms",
    "check_required_assets",
    "check_required_hashtags",
    "check_required_mentions",
    "check_spelling_locks",
    "check_video_stream",
    "check_watermark_full_video",
    "check_watermark_present",
]

_SILENCE_THRESHOLD_DB = -80.0
_VOLUMEDETECT_TIMEOUT_S = 60.0
_FFPROBE_TIMEOUT_S = 30.0
_MAX_VOLUME_RE = re.compile(r"max_volume:\s*(-inf|inf|-?\d+(?:\.\d+)?)\s*dB")
_STDERR_TAIL = 500


Validator = Callable[[GateContext], CheckOutcome]


def _pass(**evidence: object) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.PASS, evidence=dict(evidence))


def _fail(**evidence: object) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.FAIL, evidence=dict(evidence))


def _unsupported(reason: str) -> CheckOutcome:
    return CheckOutcome(status=CheckStatus.UNSUPPORTED, evidence={"reason": reason})


def check_artifact_integrity(context: GateContext) -> CheckOutcome:
    """Verifica que el artefacto exista y su hash sea calculable.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS con el sha256 del artefacto, o FAIL si no se pudo leer.
    """
    if context.artifact_sha256 is None:
        return _fail(reason="el artefacto no existe o no se pudo leer")
    return _pass(sha256=context.artifact_sha256)


def check_duration_min(context: GateContext) -> CheckOutcome:
    """Verifica que la duración del artefacto no baje del mínimo.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS/FAIL con la cota y la duración medida, o UNSUPPORTED sin probe.
    """
    return _duration_outcome(context, context.rules.duration.min_s, minimum=True)


def check_duration_max(context: GateContext) -> CheckOutcome:
    """Verifica que la duración del artefacto no supere el máximo.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS/FAIL con la cota y la duración medida, o UNSUPPORTED sin probe.
    """
    return _duration_outcome(context, context.rules.duration.max_s, minimum=False)


def check_audio_present(context: GateContext) -> CheckOutcome:
    """Verifica que el artefacto tenga pista de audio.

    La exigencia es por plataforma: ``audio_rule=any`` no exige audio.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si la plataforma no exige audio o el artefacto tiene pista;
        FAIL si lo exige y no la tiene; UNSUPPORTED sin probe.
    """
    if context.rules.audio_rule is AudioRule.ANY:
        return _pass(audio_rule="any")
    if context.media is None:
        return _unsupported("no se pudo inspeccionar el artefacto")
    if context.media.has_audio:
        return _pass(has_audio=True)
    return _fail(has_audio=False)


def check_audio_policy(context: GateContext) -> CheckOutcome:
    """Verifica la política de audio declarada por la campaña.

    El sonido oficial interno nunca pasa en silencio ni queda sin verificar:
    exige aprobación humana explícita. Las demás políticas (o la ausencia de
    política) no imponen revisión por sí mismas; el audio por plataforma lo
    siguen gobernando ``audio.present`` y las reglas de ``audio_rule``.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        MANUAL_REVIEW con ``internal_official_sound``; PASS en otro caso.
    """
    if context.contract.audio_policy is AudioPolicy.INTERNAL_OFFICIAL_SOUND:
        return CheckOutcome(
            status=CheckStatus.MANUAL_REVIEW,
            evidence={
                "audio_policy": AudioPolicy.INTERNAL_OFFICIAL_SOUND.value,
                "reason": (
                    "el sonido oficial interno exige aprobación humana "
                    "(--approve-manual-review --approved-by)"
                ),
            },
        )
    policy = context.contract.audio_policy
    return _pass(audio_policy=None if policy is None else policy.value)


def check_audio_silence(context: GateContext) -> CheckOutcome:
    """Verifica silencio digital real en el artefacto con ``volumedetect``.

    Solo aplica con ``audio_policy=internal_official_sound``: el gate no
    confía en que el render pasara ``-af volume=0`` a ffmpeg, mide el
    ``max_volume`` real del MP4 final con ``ffmpeg -af volumedetect -f null -``
    (lista de argumentos, sin shell). Fail-closed: sin ffmpeg, sin archivo o
    sin medición, el resultado es ``fail``.

    Antes de medir cuenta las pistas de audio con ffprobe: ``volumedetect``
    solo mide la pista por defecto, así que una pista extra audible pasaría
    inadvertida. Con 0 o más de 1 pista el resultado es ``fail`` sin medir;
    con exactamente 1 pista se mide con ``-map 0:a:0``.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si la política no exige silencio o el ``max_volume`` medido no
        supera los -80.0 dB; FAIL si hay audio audible, el conteo de pistas
        no es exactamente 1 o no se pudo medir.
    """
    policy = context.contract.audio_policy
    if policy is not AudioPolicy.INTERNAL_OFFICIAL_SOUND:
        return _pass(audio_policy=None if policy is None else policy.value)
    artifact = context.piece.artifact_path
    if not artifact.is_file():
        return _fail(reason="el artefacto no existe; no se pudo verificar el silencio")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return _fail(reason="ffmpeg no está disponible; no se pudo verificar el silencio")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return _fail(reason="ffprobe no está disponible; no se pudo contar las pistas de audio")
    single_track = _require_single_audio_track(ffprobe, artifact)
    if single_track is not None:
        return single_track
    return _measure_silence(ffmpeg, artifact)


def _require_single_audio_track(ffprobe: str, artifact: Path) -> CheckOutcome | None:
    """Exige exactamente 1 pista de audio antes de medir el silencio (fail-closed).

    Args:
        ffprobe: Binario ffprobe resuelto en el PATH.
        artifact: Ruta del MP4 final a inspeccionar.

    Returns:
        None si el artefacto tiene exactamente 1 pista de audio; el FAIL
        correspondiente si no se pudo contar o el conteo difiere de 1.
    """
    tracks = _count_audio_tracks(ffprobe, artifact)
    if tracks is None:
        return _fail(reason="no se pudo contar las pistas de audio del artefacto")
    if tracks != 1:
        return _fail(
            audio_tracks=tracks,
            reason=(
                f"el artefacto tiene {tracks} pistas de audio; se exige exactamente 1 "
                "para que la medición de silencio sea fiable"
            ),
        )
    return None


def _count_audio_tracks(ffprobe: str, artifact: Path) -> int | None:
    """Cuenta las pistas de audio del artefacto con ffprobe (fail-closed).

    Args:
        ffprobe: Binario ffprobe resuelto en el PATH.
        artifact: Ruta del MP4 final a inspeccionar.

    Returns:
        El número de pistas de audio, o ``None`` si no se pudo contar.
    """
    # ffprobe no acepta `-nostdin` (falla con "Option not found"): solo
    # `-hide_banner` y `-v error` lo silencian sin romper la inspección.
    argv = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(artifact),
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_S,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return sum(1 for line in completed.stdout.splitlines() if line.strip())


def _measure_silence(ffmpeg: str, artifact: Path) -> CheckOutcome:
    """Mide el ``max_volume`` del artefacto con ``volumedetect`` (fail-closed).

    Args:
        ffmpeg: Binario ffmpeg resuelto en el PATH.
        artifact: Ruta del MP4 final a medir.

    Returns:
        PASS si el ``max_volume`` no supera los -80.0 dB; FAIL si hay audio
        audible, ffmpeg falla o no se pudo medir.
    """
    # `volumedetect` reporta en nivel info: `-v info` es obligatorio para que
    # el `max_volume` aparezca en stderr (`-v error` lo silenciaría).
    # `-map 0:a:0` fija la única pista de audio (el conteo previo ya exigió
    # exactamente 1): sin el mapa, ffmpeg elegiría la pista por defecto.
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "info",
        "-i",
        str(artifact),
        "-map",
        "0:a:0",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_VOLUMEDETECT_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as error:
        return _fail(reason=f"ffmpeg no está disponible: {error}")
    except subprocess.TimeoutExpired:
        return _fail(reason=f"volumedetect excedió el timeout de {_VOLUMEDETECT_TIMEOUT_S} s")
    except OSError as error:
        return _fail(reason=f"no se pudo ejecutar ffmpeg: {error}")
    return _parse_volumedetect(completed)


def _parse_volumedetect(completed: subprocess.CompletedProcess[str]) -> CheckOutcome:
    """Interpreta la salida de ``volumedetect`` contra el umbral de silencio.

    Args:
        completed: Proceso ffmpeg ya terminado.

    Returns:
        PASS si el ``max_volume`` reportado no supera los -80.0 dB; FAIL en
        cualquier otro caso (fallo de ffmpeg, salida sin medición o audio
        audible).
    """
    if completed.returncode != 0:
        return _fail(
            reason=f"volumedetect falló con código {completed.returncode}",
            stderr=completed.stderr.strip()[-_STDERR_TAIL:],
        )
    match = _MAX_VOLUME_RE.search(completed.stderr)
    if match is None:
        return _fail(
            reason="volumedetect no reportó max_volume",
            stderr=completed.stderr.strip()[-_STDERR_TAIL:],
        )
    raw = match.group(1)
    try:
        max_volume = float(raw)
    except ValueError:
        return _fail(reason=f"max_volume no medible: {raw!r}")
    if max_volume <= _SILENCE_THRESHOLD_DB:
        return _pass(max_volume_db=raw, threshold_db=_SILENCE_THRESHOLD_DB)
    return _fail(
        max_volume_db=raw,
        threshold_db=_SILENCE_THRESHOLD_DB,
        reason="el artefacto no está en silencio digital",
    )


def check_video_stream(context: GateContext) -> CheckOutcome:
    """Verifica que el artefacto tenga pista de video.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS/FAIL según la pista de video; UNSUPPORTED sin probe o si el
        contrato no es de formato video.
    """
    if context.contract.format is not Format.VIDEO:
        return _unsupported("la validación de video no aplica a este formato")
    if context.media is None:
        return _unsupported("no se pudo inspeccionar el artefacto")
    if context.media.has_video:
        return _pass(width=context.media.width, height=context.media.height)
    return _fail(has_video=False)


def check_required_mentions(context: GateContext) -> CheckOutcome:
    """Verifica las menciones obligatorias del caption.

    La comparación exige frontera de token (``@marca`` no se satisface con
    ``@marcado`` ni con ``correo@marca.com``) y es insensible a mayúsculas.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si todas las menciones (``required_mentions`` más
        ``caption_rules.must_mention``) aparecen; FAIL con las faltantes.
    """
    required = [
        *context.rules.required_mentions,
        *context.rules.caption_rules.must_mention,
    ]
    missing = [
        mention for mention in required if not _contains_mention(context.piece.caption, mention)
    ]
    evidence: dict[str, object] = {"required": required, "missing": missing}
    return CheckOutcome(
        status=CheckStatus.FAIL if missing else CheckStatus.PASS,
        evidence=evidence,
    )


def check_required_hashtags(context: GateContext) -> CheckOutcome:
    """Verifica los hashtags obligatorios.

    El hashtag cuenta si aparece en el campo ``hashtags`` (igualdad
    normalizada) o en el caption con frontera de token (``#marca`` no se
    satisface con ``#marcado``); la comparación es insensible a mayúsculas.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si cada hashtag aparece; FAIL con los faltantes.
    """
    available = {_normalize_hashtag(tag) for tag in context.piece.hashtags}
    missing = [
        tag
        for tag in context.rules.required_hashtags
        if _normalize_hashtag(tag) not in available
        and not _contains_hashtag(context.piece.caption, tag)
    ]
    evidence: dict[str, object] = {
        "required": list(context.rules.required_hashtags),
        "missing": missing,
    }
    return CheckOutcome(
        status=CheckStatus.FAIL if missing else CheckStatus.PASS,
        evidence=evidence,
    )


def check_forbidden_terms(context: GateContext) -> CheckOutcome:
    """Verifica que no aparezcan términos prohibidos en caption ni hashtags.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si no aparece ningún término de ``caption_rules.forbidden`` ni
        de las ``prohibitions`` de la campaña en el caption ni en los
        hashtags; FAIL con los encontrados.
    """
    haystack = "\n".join((context.piece.caption, *context.piece.hashtags)).lower()
    forbidden = list(
        dict.fromkeys([*context.rules.caption_rules.forbidden, *context.contract.prohibitions])
    )
    found = [term for term in forbidden if term.lower() in haystack]
    evidence: dict[str, object] = {"forbidden": forbidden, "found": found}
    return CheckOutcome(
        status=CheckStatus.FAIL if found else CheckStatus.PASS,
        evidence=evidence,
    )


def check_first_line(context: GateContext) -> CheckOutcome:
    """Verifica que el caption abra con la primera línea exigida.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si no hay exigencia o la primera línea la contiene; FAIL en
        caso contrario.
    """
    required = context.rules.caption_rules.first_line
    if required is None:
        return _pass(required=None)
    first_line = context.piece.caption.splitlines()[0].strip() if context.piece.caption else ""
    evidence: dict[str, object] = {"required": required, "first_line": first_line}
    return CheckOutcome(
        status=CheckStatus.PASS if required.strip() in first_line else CheckStatus.FAIL,
        evidence=evidence,
    )


def check_spelling_locks(context: GateContext) -> CheckOutcome:
    """Verifica el spelling exacto de los locks en los subtítulos.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si el contrato no declara locks o todos aparecen literalmente;
        FAIL con los locks faltantes; UNSUPPORTED si hay locks declarados
        pero la pieza no trae subtítulos que verificar.
    """
    if not context.contract.spelling_locks:
        return _pass(reason="el contrato no declara spelling_locks")
    if context.piece.subtitle_text is None:
        return _unsupported("no hay subtítulos para verificar los spelling locks")
    missing = [
        lock for lock in context.contract.spelling_locks if lock not in context.piece.subtitle_text
    ]
    evidence: dict[str, object] = {
        "locks": list(context.contract.spelling_locks),
        "missing": missing,
    }
    return CheckOutcome(
        status=CheckStatus.FAIL if missing else CheckStatus.PASS,
        evidence=evidence,
    )


def check_required_assets(context: GateContext) -> CheckOutcome:
    """Verifica los assets obligatorios contra el registro y el contrato.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si cada asset requerido está registrado, su archivo está
        íntegro y su hash y tamaño coinciden con lo declarado en el
        contrato; FAIL con los ausentes, divergentes, manipulados o con
        rutas inseguras.
    """
    missing: list[str] = []
    mismatched: list[str] = []
    tampered: list[str] = []
    unsafe: list[str] = []
    for asset in context.contract.assets.required:
        try:
            registered = context.assets.get(asset.asset_id)
            intact = context.assets.verify(asset.asset_id)
        except AssetNotFoundError:
            missing.append(asset.asset_id)
            continue
        except (AssetError, OSError) as error:
            unsafe.append(f"{asset.asset_id}: {error}")
            continue
        if registered.sha256 != asset.sha256 or registered.size_bytes != asset.size_bytes:
            mismatched.append(asset.asset_id)
        elif not intact:
            tampered.append(asset.asset_id)
    evidence: dict[str, object] = {
        "missing": missing,
        "mismatched": mismatched,
        "tampered": tampered,
        "unsafe": unsafe,
    }
    failed = bool(missing or mismatched or tampered or unsafe)
    return CheckOutcome(
        status=CheckStatus.FAIL if failed else CheckStatus.PASS,
        evidence=evidence,
    )


DEFAULT_VALIDATORS: dict[str, Validator] = {
    "artifact.integrity": check_artifact_integrity,
    "artifact.video_stream": check_video_stream,
    "assets.required": check_required_assets,
    "audio.present": check_audio_present,
    "audio.policy": check_audio_policy,
    "audio.silence": check_audio_silence,
    "caption.first_line": check_first_line,
    "caption.forbidden": check_forbidden_terms,
    "caption.required_hashtag": check_required_hashtags,
    "caption.required_mention": check_required_mentions,
    "duration.max": check_duration_max,
    "duration.min": check_duration_min,
    "subtitles.spelling_lock": check_spelling_locks,
    "watermark.full_video": check_watermark_full_video,
    "watermark.present": check_watermark_present,
}


def _duration_outcome(context: GateContext, bound: int | None, *, minimum: bool) -> CheckOutcome:
    if context.media is None:
        return _unsupported("no se pudo inspeccionar el artefacto")
    if bound is None:
        return _pass(bound=None)
    duration = context.media.duration_s
    if duration is None:
        return _unsupported("no se pudo medir la duración del artefacto")
    within = duration >= bound if minimum else duration <= bound
    label = "min_s" if minimum else "max_s"
    evidence: dict[str, object] = {label: bound, "duration_s": duration}
    return CheckOutcome(
        status=CheckStatus.PASS if within else CheckStatus.FAIL,
        evidence=evidence,
    )


def _contains_mention(caption: str, mention: str) -> bool:
    pattern = rf"(?<![\w@]){re.escape(mention)}(?!\w)"
    return re.search(pattern, caption, flags=re.IGNORECASE) is not None


def _contains_hashtag(caption: str, tag: str) -> bool:
    normalized = _normalize_hashtag(tag)
    pattern = rf"(?<![\w#])#{re.escape(normalized)}(?!\w)"
    return re.search(pattern, caption, flags=re.IGNORECASE) is not None


def _normalize_hashtag(tag: str) -> str:
    return tag.strip().lstrip("#").lower()
