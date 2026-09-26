"""Transcripción word-level de audio con faster-whisper.

El alcance termina en ``Transcript``: no hay detección de escenas,
segmentación ni render de subtítulos aquí. La inferencia usa el modelo
``small`` con cuantización ``int8`` — la única ruta garantizada en la VRAM
objetivo de 4 GB (brief §8) — y libera el modelo siempre al terminar, incluso
ante error, para no retener VRAM entre etapas del pipeline.

``faster-whisper`` se importa de forma diferida: es una dependencia opcional
(extra ``transcription``) y el módulo base debe importarse sin ella.
"""

import gc
import importlib
import logging
import os
import shutil
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import ClassVar, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_ALLOWED_MODEL_SIZES = frozenset({"small"})
_ALLOWED_COMPUTE_TYPES = frozenset({"int8"})
_ALLOWED_DEVICES = frozenset({"auto", "cpu", "cuda"})

_MODEL_CACHE: dict[tuple[str, str, str, int], object] = {}

_WINERROR_SYMLINK_PRIVILEGE = 1314


def clear_model_cache() -> None:
    """Vacía la caché de modelos de faster-whisper.

    Se usa en teardowns de tests y para liberar referencias tras un lote;
    la VRAM se libera por transcripción en ``_release_model``.
    """
    _MODEL_CACHE.clear()


class TranscriptionError(Exception):
    """La transcripción no se pudo completar."""


class Word(BaseModel):
    """Una palabra transcrita con su marca de tiempo."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    token_id: int = Field(ge=0)


class Transcript(BaseModel):
    """Transcripción completa de un audio con palabras y texto plano."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    words: tuple[Word, ...]
    language: str = Field(min_length=1, max_length=10)
    duration_s: float = Field(ge=0.0)
    text: str = Field(min_length=1)


class _WhisperWord(Protocol):
    """Palabra tal como la entrega faster-whisper."""

    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...

    @property
    def word(self) -> str: ...

    @property
    def probability(self) -> float: ...


class _WhisperSegment(Protocol):
    """Segmento de faster-whisper con sus palabras opcionales."""

    @property
    def words(self) -> Iterable[_WhisperWord] | None: ...


class _WhisperInfo(Protocol):
    """Metadatos de la transcripción de faster-whisper."""

    @property
    def language(self) -> str: ...

    @property
    def duration(self) -> float: ...


class _WhisperEngine(Protocol):
    """Motor cargado de faster-whisper."""

    def transcribe(
        self,
        audio: str,
        *,
        word_timestamps: bool,
    ) -> tuple[Iterable[_WhisperSegment], _WhisperInfo]: ...


class _ModelFactory(Protocol):
    """Constructor del motor de faster-whisper."""

    def __call__(
        self,
        model_size_or_path: str,
        *,
        device: str,
        compute_type: str,
        num_workers: int,
    ) -> _WhisperEngine: ...


class _TorchCuda(Protocol):
    """Superficie mínima de ``torch.cuda`` que usa la liberación de VRAM."""

    def is_available(self) -> bool: ...

    def empty_cache(self) -> None: ...


class Transcriber(Protocol):
    """Interfaz del motor de transcripción."""

    def transcribe(self, audio: Path) -> Transcript:
        """Transcribe un audio a palabras con marca de tiempo.

        Args:
            audio: Ruta del audio a transcribir.

        Returns:
            La transcripción con palabras y texto plano.

        Raises:
            TranscriptionError: Si el audio no existe o la transcripción falla.
        """
        ...


class FasterWhisperTranscriber:
    """Transcripción word-level con faster-whisper (small, int8)."""

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        num_workers: int = 1,
    ) -> None:
        """Configura el motor de transcripción.

        Solo se admite ``small`` con ``int8``: es la ruta medida como segura en
        la VRAM objetivo de 4 GB (brief §8). Otros modelos o cuantizaciones no
        se aceptan hasta medirlos en ese hardware; ``medium`` queda
        explícitamente fuera.

        Args:
            model_size: Modelo de faster-whisper; debe ser ``small``.
            device: Dispositivo de inferencia (``auto``, ``cpu`` o ``cuda``).
            compute_type: Cuantización; debe ser ``int8``.
            num_workers: Número de workers del modelo; debe ser positivo.

        Raises:
            ValueError: Si el modelo, la cuantización, el dispositivo o el
                número de workers no son válidos.
        """
        if model_size not in _ALLOWED_MODEL_SIZES:
            msg = (
                f"modelo no permitido: {model_size!r}; solo 'small' está medido "
                "como seguro en la VRAM objetivo de 4 GB"
            )
            raise ValueError(msg)
        if compute_type not in _ALLOWED_COMPUTE_TYPES:
            msg = (
                f"cuantización no permitida: {compute_type!r}; solo 'int8' está "
                "medida como segura en la VRAM objetivo de 4 GB"
            )
            raise ValueError(msg)
        if device not in _ALLOWED_DEVICES:
            msg = f"dispositivo no permitido: {device!r}"
            raise ValueError(msg)
        if num_workers <= 0:
            msg = f"num_workers inválido: {num_workers}"
            raise ValueError(msg)
        self._model_size: str = model_size
        self._device: str = device
        self._compute_type: str = compute_type
        self._num_workers: int = num_workers
        self._model: _WhisperEngine | None = None

    def transcribe(self, audio: Path) -> Transcript:
        """Transcribe un audio a palabras con marca de tiempo.

        El modelo se libera siempre al terminar, incluso ante error, para no
        retener VRAM entre etapas del pipeline.

        Args:
            audio: Ruta del audio a transcribir.

        Returns:
            La transcripción con palabras y texto plano.

        Raises:
            TranscriptionError: Si el audio no existe, no se detectan palabras
                o faster-whisper falla al transcribir.
        """
        if not audio.is_file():
            msg = f"el audio no existe: {audio}"
            raise TranscriptionError(msg)
        try:
            segments, info = self._ensure_model().transcribe(str(audio), word_timestamps=True)
            return _build_transcript(segments, info, audio)
        except TranscriptionError:
            raise
        except Exception as error:
            # Frontera con faster-whisper: cualquier fallo de la librería se
            # traduce a TranscriptionError conservando la causa.
            msg = f"faster-whisper falló al transcribir {audio}: {error}"
            raise TranscriptionError(msg) from error
        finally:
            self._release_model()

    def _ensure_model(self) -> _WhisperEngine:
        if self._model is not None:
            return self._model
        _register_windows_cuda_dlls()
        _patch_windows_symlinks()
        key = (self._model_size, self._device, self._compute_type, self._num_workers)
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            self._model = cast("_WhisperEngine", cached)
            return self._model
        factory = _load_model_factory()
        try:
            model = factory(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                num_workers=self._num_workers,
            )
        except Exception as error:
            if self._device == "cuda":
                msg = f"CUDA solicitado pero falló al inicializar faster-whisper: {error}"
                raise TranscriptionError(msg) from error
            if self._device == "auto":
                logger.warning("CUDA no disponible (%s); usando CPU", error)
                try:
                    model = factory(
                        self._model_size,
                        device="cpu",
                        compute_type=self._compute_type,
                        num_workers=self._num_workers,
                    )
                except Exception as fallback_error:
                    msg = f"faster-whisper falló en CPU tras fallback de auto: {fallback_error}"
                    raise TranscriptionError(msg) from fallback_error
            else:
                raise
        _MODEL_CACHE[key] = cast("object", model)
        self._model = model
        return self._model

    def _release_model(self) -> None:
        """Libera VRAM tras la inferencia."""
        self._model = None
        _ = gc.collect()
        cuda = _load_torch_cuda()
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()


def _load_model_factory() -> _ModelFactory:
    # Import diferido: faster-whisper es una dependencia opcional.
    module = importlib.import_module("faster_whisper")
    factory = getattr(module, "WhisperModel", None)
    if factory is None:
        msg = "faster-whisper no expone WhisperModel"
        raise TranscriptionError(msg)
    return cast("_ModelFactory", factory)


def _load_torch_cuda() -> _TorchCuda | None:
    # torch es opcional (solo aporta el vaciado de caché CUDA); import diferido.
    try:
        module = importlib.import_module("torch")
    except ImportError:
        return None
    cuda = getattr(module, "cuda", None)
    if cuda is None:
        return None
    return cast("_TorchCuda", cuda)


def _register_windows_cuda_dlls() -> None:
    """Registra los DLLs CUDA del venv en Windows antes de cargar ctranslate2.

    En Windows, ``ctranslate2`` (backend de faster-whisper) necesita
    ``cublas`` y ``cudnn`` de los paquetes ``nvidia-*``. Se añaden
    ``nvidia/cublas/bin`` y ``nvidia/cudnn/bin`` del venv a ``PATH`` y a
    ``os.add_dll_directory`` para que la carga no falle con DLL faltante.

    Es best-effort: si no hay venv o no existen los directorios, no hace nada.
    """
    if sys.platform != "win32":
        return
    try:
        prefix = Path(sys.prefix)
    except (ValueError, OSError):
        return
    candidates = (
        prefix / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        prefix / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
    )
    for bin_dir in candidates:
        try:
            if not bin_dir.is_dir():
                continue
        except OSError:
            continue
        try:
            current = os.environ.get("PATH", "")
            if str(bin_dir) not in current:
                os.environ["PATH"] = str(bin_dir) + os.pathsep + current
        except (ValueError, OSError):
            continue
        add_dll = getattr(os, "add_dll_directory", None)
        if callable(add_dll):
            try:
                _ = add_dll(str(bin_dir))
            except (ValueError, OSError):
                continue


def _patch_windows_symlinks() -> None:
    """Sustituye ``os.symlink`` por una copia en Windows sin Developer Mode.

    ``huggingface_hub`` usa ``os.symlink`` para cachear modelos y falla con
    ``WinError 1314`` sin Developer Mode, lo que impide descargar
    ``large-v3-turbo`` y otros modelos. El fallback copia el archivo o el
    árbol para que la descarga complete; otros errores se relanzan.

    Es idempotente: si ya se parcheó, no vuelve a envolver.
    """
    if sys.platform != "win32":
        return
    if getattr(os.symlink, "__name__", "") == "safe_symlink":
        return
    original = os.symlink

    def safe_symlink(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
        *args: object,
        dir_fd: int | None = None,
        **kwargs: object,
    ) -> None:
        dynamic = cast("Callable[..., None]", original)
        call_kwargs: dict[str, object] = dict(kwargs)
        if dir_fd is not None:
            call_kwargs["dir_fd"] = dir_fd
        try:
            dynamic(src, dst, *args, **call_kwargs)
        except OSError as error:
            if getattr(error, "winerror", None) != _WINERROR_SYMLINK_PRIVILEGE:
                raise
            src_str = os.fspath(src)
            dst_str = os.fspath(dst)
            if Path(src_str).is_dir():
                _ = shutil.copytree(src_str, dst_str)
            else:
                _ = shutil.copy2(src_str, dst_str)
        else:
            return
        return

    os.symlink = safe_symlink


def _build_transcript(
    segments: Iterable[_WhisperSegment],
    info: _WhisperInfo,
    audio: Path,
) -> Transcript:
    words = _extract_words(segments)
    if not words:
        msg = f"no se detectaron palabras en el audio: {audio}"
        raise TranscriptionError(msg)
    return Transcript(
        words=words,
        language=info.language,
        duration_s=info.duration,
        text=" ".join(word.text for word in words),
    )


def _extract_words(segments: Iterable[_WhisperSegment]) -> tuple[Word, ...]:
    words: list[Word] = []
    for segment in segments:
        if segment.words is None:
            continue
        for raw in segment.words:
            text = raw.word.strip()
            if not text:
                continue
            words.append(
                Word(
                    start_s=raw.start,
                    end_s=raw.end,
                    text=text,
                    confidence=raw.probability,
                    token_id=len(words),
                )
            )
    return tuple(words)
