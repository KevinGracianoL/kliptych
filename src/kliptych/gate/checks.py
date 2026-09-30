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
- ``brand.safety``: riesgo de controversia/toxicidad solo si la campaña lo
  exige explícitamente (menciones en ``prohibitions``); con la regla
  activa un evaluador LLM decide (riesgo → ``manual_review``) y cualquier
  fallo del evaluador es ``manual_review`` (fail-closed, jamás ``pass``).
- ``caption.first_line``: el caption abre con la primera línea exigida.
- ``caption.forbidden``: no aparecen términos prohibidos en lo publicado
  por la cuenta (caption y hashtags, unión de ``caption_rules.forbidden``
  y ``prohibitions`` de la campaña) ni en lo dicho por el streamer
  (``subtitle_text`` de Whisper). La comparación normaliza (NFKD sin
  diacríticos + casefold) y exige frontera de palabra sobre la frase
  escapada (``re.escape``). La autoría decide el veredicto: coincidencia
  en lo publicado → ``fail``; solo en lo dicho → ``manual_review``.
- ``caption.required_hashtag``: están los hashtags obligatorios, con frontera
  de token y comparación insensible a mayúsculas.
- ``caption.required_mention``: están las menciones obligatorias, con frontera
  de token y comparación insensible a mayúsculas.
- ``duration.min`` / ``duration.max``: duración dentro del rango; si no se
  pudo medir, el resultado es ``unsupported`` (jamás ``pass``).
- ``hook.keyword``: la palabra clave de apertura aparece en subtítulos o
  texto en pantalla con inicio <= 3.0 s; ausente o tardía es ``fail``. El
  volumen de los primeros 3 s se adjunta como nota informativa.
- ``subtitles.spelling_lock``: spelling exacto en subtítulos; sin subtítulos
  y con locks declarados el resultado es ``unsupported``.
- ``watermark.full_video`` / ``watermark.present``: watermark exigido durante
  todo el video o en alguna parte; se verifica con OpenCV frame por frame
  (``cv2.matchTemplate`` contra el PNG del contrato, con zona, tamaño
  mínimo y opacidad del contrato) y cualquier fallo es ``fail``
  (fail-closed).

Reglas declaradas sin validador registrado jamás pasan: el motor las marca
``unsupported`` (o ``manual_review`` si el contrato las clasificó así).
"""

import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from kliptych.assets import AssetError, AssetNotFoundError, AssetRegistry
from kliptych.contract import AudioPolicy, AudioRule, Format
from kliptych.gate.brand_safety import check_brand_safety
from kliptych.gate.hook import check_hook_keyword
from kliptych.gate.models import CheckOutcome, CheckStatus, GateContext, Piece, SubtitleSegment
from kliptych.gate.text import contains_phrase, normalize_text
from kliptych.gate.watermark import check_watermark_full_video, check_watermark_present
from kliptych.lyrics import (
    LrcEmptyWindowError,
    LrcParseError,
    LyricLine,
    LyricsError,
    cut_lyric_window,
    parse_lrc,
)

__all__ = [
    "DEFAULT_VALIDATORS",
    "CheckOutcome",
    "GateContext",
    "Validator",
    "check_artifact_integrity",
    "check_audio_policy",
    "check_audio_present",
    "check_audio_silence",
    "check_brand_safety",
    "check_duration_max",
    "check_duration_min",
    "check_first_line",
    "check_forbidden_terms",
    "check_hook_keyword",
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
    """Verifica que no aparezcan términos prohibidos, con autoría separada.

    Lo publicado por la cuenta (caption y hashtags) falla el gate; lo
    dicho por el streamer (``subtitle_text`` de Whisper) exige revisión
    humana: el texto del directo no lo redacta la campaña y un falso
    positivo no debe rechazar la pieza en silencio.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        FAIL si algún término de ``caption_rules.forbidden`` o de las
        ``prohibitions`` aparece en caption o hashtags; MANUAL_REVIEW si
        solo aparece en ``subtitle_text``; PASS en caso contrario.
    """
    forbidden = list(
        dict.fromkeys([*context.rules.caption_rules.forbidden, *context.contract.prohibitions])
    )
    screen_texts = [seg.text for seg in context.piece.screen_text_segments]
    published = "\n".join((context.piece.caption, *context.piece.hashtags, *screen_texts))
    spoken = context.piece.subtitle_text or ""
    found_published = [term for term in forbidden if _contains_forbidden_term(published, term)]
    found_spoken = [
        term
        for term in forbidden
        if term not in found_published and _contains_forbidden_term(spoken, term)
    ]
    if found_published:
        return CheckOutcome(
            status=CheckStatus.FAIL,
            evidence={
                "forbidden": forbidden,
                "found": found_published,
                "authorship": "published",
            },
        )
    if found_spoken:
        return CheckOutcome(
            status=CheckStatus.MANUAL_REVIEW,
            evidence={
                "forbidden": forbidden,
                "found": found_spoken,
                "authorship": "spoken",
            },
        )
    return CheckOutcome(
        status=CheckStatus.PASS,
        evidence={"forbidden": forbidden, "found": [], "authorship": "none"},
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


def _extract_ordered_tokens(text: str) -> list[str]:
    norm = normalize_text(text)
    raw_tokens = re.split(r"[\s_]+", norm)
    tokens: list[str] = []
    for token in raw_tokens:
        cleaned = re.sub(r"^[^\wñ]+|[^\wñ]+$", "", token)
        if cleaned:
            tokens.append(cleaned)
    return tokens


_ASS_DIALOGUE_PARTS: int = 10
_ASS_SPLIT_MAX: int = 9
_ASS_TIME_TOLERANCE_SEC = 0.02
_ASS_TIME_RE = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
_SECONDS_PER_MINUTE = 60.0
_SECONDS_PER_HOUR = 3600.0


def _parse_ass_timestamp(token: str) -> float | None:
    """Parsea una marca de tiempo ``H:MM:SS.cc`` de un evento Dialogue.

    Args:
        token: Marca tal como aparece en los campos Start/End del Dialogue.

    Returns:
        Los segundos como flotante, o ``None`` si el formato es inválido.
    """
    match = _ASS_TIME_RE.match(token.strip())
    if match is None:
        return None
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    if minutes >= _SECONDS_PER_MINUTE or seconds >= _SECONDS_PER_MINUTE:
        return None
    hours = int(match.group(1))
    return float(hours) * _SECONDS_PER_HOUR + float(minutes) * _SECONDS_PER_MINUTE + seconds


def _find_ass_path(piece: Piece) -> Path | None:
    """Busca el .ass declarado o el sidecar junto al artefacto.

    Args:
        piece: Pieza con el ``ass_path`` declarado y la ruta del artefacto.

    Returns:
        La ruta existente a verificar, o ``None`` sin candidato en disco.
    """
    if piece.ass_path is not None and piece.ass_path.is_file():
        return piece.ass_path
    candidate_default = piece.artifact_path.parent / "subtitles.ass"
    if candidate_default.is_file():
        return candidate_default
    candidate_stem = piece.artifact_path.with_suffix(".ass")
    if candidate_stem.is_file():
        return candidate_stem
    return None


def _read_ass_dialogues(ass_path: Path) -> tuple[list[tuple[float, float, str]], str | None]:
    """Lee los eventos Dialogue de un .ass como (inicio, fin, texto).

    Args:
        ass_path: Ruta del archivo .ass ya resuelto.

    Returns:
        La lista de eventos y ``None``, o la lista vacía y el motivo del
        fallo cuando el archivo no se puede leer o trae eventos
        inverificables (fail-closed).
    """
    try:
        content = ass_path.read_text(encoding="utf-8")
    except OSError as error:
        return [], f"no se pudo leer el archivo .ass en {ass_path}: {error}"
    events: list[tuple[float, float, str]] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("Dialogue:"):
            continue
        parts = stripped.split(",", _ASS_SPLIT_MAX)
        if len(parts) < _ASS_DIALOGUE_PARTS:
            return [], f"evento Dialogue malformado en {ass_path} (línea {lineno})"
        start = _parse_ass_timestamp(parts[1])
        end = _parse_ass_timestamp(parts[2])
        if start is None or end is None or end <= start:
            return [], f"marca de tiempo .ass inválida en {ass_path} (línea {lineno})"
        events.append((start, end, parts[_ASS_SPLIT_MAX].strip()))
    return events, None


def _verify_ass_dialogue_events(piece: Piece, expected_lines: Sequence[LyricLine]) -> str | None:
    """Verifica tiempos y textos de los eventos Dialogue contra la ventana esperada.

    Exige el mismo número de eventos que de líneas recortadas por
    ``cut_lyric_window`` y, por cada línea, tokens idénticos e intervalos
    (inicio y fin) dentro de ``_ASS_TIME_TOLERANCE_SEC``. Cualquier
    desplazamiento, compresión o extensión falla.

    Args:
        piece: Pieza con el ``ass_path`` declarado o inferible.
        expected_lines: Líneas de la ventana temporal ya recortada y
            desplazada a 0.0 s.

    Returns:
        ``None`` si cada evento concuerda en texto y tiempos; el motivo del
        fallo en caso contrario.
    """
    ass_path = _find_ass_path(piece)
    if ass_path is None:
        return None
    events, read_err = _read_ass_dialogues(ass_path)
    if read_err is not None:
        return read_err
    if len(events) != len(expected_lines):
        return (
            f"archivo .ass con {len(events)} eventos Dialogue, "
            f"se esperaban {len(expected_lines)} líneas de la ventana .lrc"
        )
    for index, ((start, end, text), expected) in enumerate(
        zip(events, expected_lines, strict=True)
    ):
        candidate = _LyricCandidate(
            label="evento Dialogue", index=index, text=text, start=start, end=end
        )
        mismatch = _match_lyric_interval(candidate, expected)
        if mismatch is not None:
            return mismatch
    return None


@dataclass(frozen=True, slots=True)
class _LyricCandidate:
    """Texto e intervalo declarados por la pieza para comparar contra el .lrc."""

    label: str
    index: int
    text: str
    start: float
    end: float


def _match_lyric_interval(candidate: _LyricCandidate, expected: LyricLine) -> str | None:
    """Compara los tokens y el intervalo de una línea contra lo esperado.

    Args:
        candidate: Texto e intervalo declarados por la pieza.
        expected: Línea recortada por ``cut_lyric_window``.

    Returns:
        ``None`` si tokens e intervalo coinciden dentro de
        ``_ASS_TIME_TOLERANCE_SEC``; el motivo del fallo en caso contrario.
    """
    if _extract_ordered_tokens(candidate.text) != _extract_ordered_tokens(expected.text):
        return (
            f"{candidate.label} {candidate.index} no concuerda con .lrc "
            f"(esperado {expected.text!r}, obtenido {candidate.text!r})"
        )
    if abs(candidate.start - expected.start_sec) > _ASS_TIME_TOLERANCE_SEC:
        return (
            f"{candidate.label} {candidate.index} desplazado en tiempo "
            f"(esperado {expected.start_sec:.2f} s, obtenido {candidate.start:.2f} s)"
        )
    if (
        expected.end_sec is not None
        and abs(candidate.end - expected.end_sec) > _ASS_TIME_TOLERANCE_SEC
    ):
        return (
            f"{candidate.label} {candidate.index} con fin desplazado "
            f"(esperado {expected.end_sec:.2f} s, obtenido {candidate.end:.2f} s)"
        )
    return None


def _verify_subtitle_segments_exact(
    segments: Sequence[SubtitleSegment],
    expected_lines: Sequence[LyricLine],
) -> str | None:
    """Exige una línea de letra por segmento, con tokens y tiempos exactos.

    Args:
        segments: Segmentos declarados por la pieza, relativos al clip.
        expected_lines: Líneas de la ventana temporal ya recortada y
            desplazada a 0.0 s.

    Returns:
        ``None`` si el conteo, los tokens y los intervalos coinciden dentro
        de ``_ASS_TIME_TOLERANCE_SEC``; el motivo del fallo en caso contrario.
    """
    if len(segments) != len(expected_lines):
        return (
            f"pieza con {len(segments)} subtitle_segments, "
            f"se esperaban {len(expected_lines)} líneas de la ventana .lrc"
        )
    for index, (segment, expected) in enumerate(zip(segments, expected_lines, strict=True)):
        candidate = _LyricCandidate(
            label="segmento",
            index=index,
            text=segment.text,
            start=segment.start_s,
            end=segment.end_s,
        )
        mismatch = _match_lyric_interval(candidate, expected)
        if mismatch is not None:
            return mismatch
    return None


def _verify_subtitle_segments_timing(
    segments: Sequence[SubtitleSegment],
    *,
    start_sec: float | None,
    end_sec: float | None,
) -> str | None:
    for i, seg in enumerate(segments):
        if seg.start_s >= seg.end_s:
            return f"segmento {i} con start_s >= end_s ({seg.start_s} >= {seg.end_s})"
        if i > 0 and seg.start_s < segments[i - 1].start_s:
            prev_s = segments[i - 1].start_s
            return f"segmento {i} con timestamp desordenado ({seg.start_s} < {prev_s})"
        if seg.start_s < 0.0:
            return f"segmento {i} con timestamp negativo ({seg.start_s})"
        if start_sec is not None and end_sec is not None:
            max_duration = end_sec - start_sec
            if seg.end_s > max_duration + 0.1 and seg.end_s > end_sec + 0.1:
                return f"segmento {i} excede la ventana temporal ({seg.end_s})"
    return None


def _resolve_expected_lrc_lines(
    context: GateContext,
    lrc_lines: Sequence[LyricLine],
) -> tuple[tuple[LyricLine, ...] | None, str | None, float | None, float | None]:
    start_sec = context.piece.start_sec
    end_sec = context.piece.end_sec
    if (start_sec is None or end_sec is None) and len(context.contract.segments) == 1:
        start_sec = context.contract.segments[0].start_s
        end_sec = context.contract.segments[0].end_s
    if start_sec is not None and end_sec is not None:
        try:
            cut = cut_lyric_window(lrc_lines, start_sec=start_sec, end_sec=end_sec)
        except (LrcEmptyWindowError, LrcParseError, ValueError) as err:
            return None, str(err), start_sec, end_sec
        else:
            return cut, None, start_sec, end_sec
    return tuple(lrc_lines), None, start_sec, end_sec


def _load_verified_lrc_lines(
    context: GateContext, lrc_id: str
) -> tuple[tuple[LyricLine, ...] | None, str | None]:
    try:
        intact = context.assets.verify(lrc_id)
    except (AssetError, OSError) as error:
        return None, f"no se pudo verificar el asset .lrc de referencia: {error}"
    if not intact:
        return None, f"asset de letras '{lrc_id}' ausente o con integridad comprometida"
    try:
        path = context.assets.path_for(lrc_id)
        content = path.read_text(encoding="utf-8")
        return parse_lrc(content), None
    except (AssetError, OSError, ValueError, LyricsError) as error:
        return None, f"archivo .lrc de referencia inválido: {error}"


def _validate_unwindowed_segments(
    segments: Sequence[SubtitleSegment],
    expected_tokens: Sequence[str],
    *,
    start_sec: float | None,
    end_sec: float | None,
) -> str | None:
    """Valida segmentos sin ventana temporal exacta (contrato de un solo clip).

    Args:
        segments: Segmentos declarados por la pieza.
        expected_tokens: Tokens esperados del .lrc completo.
        start_sec: Inicio de la ventana, o ``None`` sin recorte.
        end_sec: Fin de la ventana, o ``None`` sin recorte.

    Returns:
        ``None`` si los tiempos son sanos y la concatenación concuerda;
        el motivo del fallo en caso contrario.
    """
    timing_err = _verify_subtitle_segments_timing(segments, start_sec=start_sec, end_sec=end_sec)
    if timing_err is not None:
        return timing_err
    seg_tokens = _extract_ordered_tokens(" ".join(seg.text for seg in segments))
    if seg_tokens != list(expected_tokens):
        return (
            "concatenación de subtitle_segments no concuerda con .lrc "
            f"(esperado {expected_tokens}, obtenido {seg_tokens})"
        )
    return None


def _validate_lyric_concordance(
    context: GateContext,
    expected_lines: Sequence[LyricLine],
    *,
    start_sec: float | None,
    end_sec: float | None,
) -> str | None:
    expected_tokens = _extract_ordered_tokens(" ".join(line.text for line in expected_lines))
    if context.piece.subtitle_text is None:
        return "pieza sin subtitle_text"
    sub_tokens = _extract_ordered_tokens(context.piece.subtitle_text)
    if sub_tokens != list(expected_tokens):
        return (
            "subtítulos no concuerdan en secuencia y multiplicidad exacta con .lrc "
            f"(esperado {expected_tokens}, obtenido {sub_tokens})"
        )

    segments = context.piece.subtitle_segments
    if not segments:
        return "pieza sin subtitle_segments"
    if start_sec is not None and end_sec is not None:
        segments_err = _verify_subtitle_segments_exact(segments, expected_lines)
    else:
        segments_err = _validate_unwindowed_segments(
            segments, expected_tokens, start_sec=start_sec, end_sec=end_sec
        )
    if segments_err is not None:
        return segments_err

    return _verify_ass_dialogue_events(context.piece, expected_lines)


def _check_lyric_ground_truth(context: GateContext) -> CheckOutcome | None:
    if (
        context.contract.format is not Format.LYRIC_VIDEO
        or not context.contract.lyric_video
        or not context.contract.lyric_video.lrc_asset_id
    ):
        return None
    lrc_id = context.contract.lyric_video.lrc_asset_id
    lrc_lines, load_err = _load_verified_lrc_lines(context, lrc_id)
    if load_err is not None or lrc_lines is None:
        return CheckOutcome(status=CheckStatus.FAIL, evidence={"reason": load_err})

    expected_lines, window_err, start_sec, end_sec = _resolve_expected_lrc_lines(context, lrc_lines)
    if window_err is not None or expected_lines is None:
        return CheckOutcome(
            status=CheckStatus.FAIL,
            evidence={"reason": f"error en ventana temporal de letras: {window_err}"},
        )
    expected_tokens = _extract_ordered_tokens(" ".join(line.text for line in expected_lines))
    if not expected_tokens:
        return CheckOutcome(
            status=CheckStatus.FAIL, evidence={"reason": "archivo .lrc sin tokens de letra"}
        )

    mismatch_reason = _validate_lyric_concordance(
        context, expected_lines, start_sec=start_sec, end_sec=end_sec
    )
    if mismatch_reason is not None:
        return CheckOutcome(status=CheckStatus.FAIL, evidence={"reason": mismatch_reason})

    return None


def check_spelling_locks(context: GateContext) -> CheckOutcome:
    """Verifica el spelling exacto de los locks en los subtítulos.

    Args:
        context: Contexto resuelto del gate.

    Returns:
        PASS si el contrato no declara locks o todos aparecen literalmente;
        FAIL con los locks faltantes o palabras alteradas; UNSUPPORTED si
        hay locks declarados pero la pieza no trae subtítulos que verificar.
    """
    is_lyric_video = context.contract.format is Format.LYRIC_VIDEO
    if is_lyric_video and (
        context.piece.subtitle_text is None
        or not context.piece.subtitle_text.strip()
        or not context.piece.subtitle_segments
    ):
        return CheckOutcome(
            status=CheckStatus.FAIL,
            evidence={"reason": "formato lyric_video sin subtítulos sincronizados verificados"},
        )

    if not context.contract.spelling_locks and not is_lyric_video:
        return _pass(reason="el contrato no declara spelling_locks")
    if context.piece.subtitle_text is None:
        return _unsupported("no hay subtítulos para verificar los spelling locks")

    missing = [
        lock
        for lock in context.contract.spelling_locks
        if not contains_phrase(context.piece.subtitle_text, lock)
    ]
    if missing:
        return CheckOutcome(
            status=CheckStatus.FAIL,
            evidence={
                "locks": list(context.contract.spelling_locks),
                "missing": missing,
            },
        )

    ground_truth_outcome = _check_lyric_ground_truth(context)
    if ground_truth_outcome is not None:
        return ground_truth_outcome

    return CheckOutcome(
        status=CheckStatus.PASS,
        evidence={
            "locks": list(context.contract.spelling_locks),
            "missing": [],
        },
    )


def _inspect_asset(
    asset_id: str,
    assets: AssetRegistry,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    try:
        registered = assets.get(asset_id)
        intact = assets.verify(asset_id)
    except AssetNotFoundError:
        return asset_id, None, None, None
    except (AssetError, OSError) as error:
        return None, None, None, f"{asset_id}: {error}"
    if expected_sha256 is not None and (
        registered.sha256 != expected_sha256 or registered.size_bytes != expected_size
    ):
        return None, asset_id, None, None
    if not intact:
        return None, None, asset_id, None
    return None, None, None, None


def _record_inspection(
    *,
    missing: list[str],
    mismatched: list[str],
    tampered: list[str],
    unsafe: list[str],
    inspection: tuple[str | None, str | None, str | None, str | None],
) -> None:
    miss, mism, tamp, uns = inspection
    if miss is not None:
        missing.append(miss)
    if mism is not None:
        mismatched.append(mism)
    if tamp is not None:
        tampered.append(tamp)
    if uns is not None:
        unsafe.append(uns)


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
        res = _inspect_asset(
            asset.asset_id,
            context.assets,
            expected_sha256=asset.sha256,
            expected_size=asset.size_bytes,
        )
        _record_inspection(
            missing=missing,
            mismatched=mismatched,
            tampered=tampered,
            unsafe=unsafe,
            inspection=res,
        )

    if context.contract.lyric_video is not None and context.contract.lyric_video.lrc_asset_id:
        lrc_id = context.contract.lyric_video.lrc_asset_id
        if not any(a.asset_id == lrc_id for a in context.contract.assets.required):
            res = _inspect_asset(lrc_id, context.assets)
            _record_inspection(
                missing=missing,
                mismatched=mismatched,
                tampered=tampered,
                unsafe=unsafe,
                inspection=res,
            )

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
    "brand.safety": check_brand_safety,
    "caption.first_line": check_first_line,
    "caption.forbidden": check_forbidden_terms,
    "caption.required_hashtag": check_required_hashtags,
    "caption.required_mention": check_required_mentions,
    "duration.max": check_duration_max,
    "duration.min": check_duration_min,
    "hook.keyword": check_hook_keyword,
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


def _contains_forbidden_term(haystack: str, term: str) -> bool:
    """Indica si una frase prohibida aparece con frontera de palabra.

    Ambos lados se normalizan (NFKD sin diacríticos salvando la 'ñ' + casefold)
    y se tratan los espacios múltiples y guiones bajos como separadores de frontera.

    Args:
        haystack: Texto donde buscar (ya incluye caption/hashtags o
            subtítulos, según la autoría evaluada).
        term: Frase prohibida tal como la declara el contrato.

    Returns:
        True si la frase normalizada aparece con fronteras de palabra.
    """
    return contains_phrase(haystack, term)


def _contains_hashtag(caption: str, tag: str) -> bool:
    normalized = _normalize_hashtag(tag)
    pattern = rf"(?<![\w#])#{re.escape(normalized)}(?!\w)"
    return re.search(pattern, caption, flags=re.IGNORECASE) is not None


def _normalize_hashtag(tag: str) -> str:
    return tag.strip().lstrip("#").lower()
