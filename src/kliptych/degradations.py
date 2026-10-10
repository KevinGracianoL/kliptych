"""Degradaciones EN CALIENTE de una corrida.

Una degradacion de pre-vuelo la detecta `detect_environment` y ya viaja al
manifiesto. Una degradacion EN CALIENTE es distinta: `run_ffmpeg_with_fallback`
pide el encoder configurado y, si falla, reintenta con libx2464. Reintentar sin
registrarlo es degradacion silenciosa: el artefacto diria que pidio NVENC
mientras el archivo es libx264, y la unica prueba seria una linea de log.

El colector vive en un `ContextVar` y no en `RenderConfig` a proposito.
`RenderConfig` es un valor congelado que se copia por todas partes; un valor que
"recuerda lo que le paso" deja de ser un valor. El colector es estado de la
CORRIDA, no de la configuracion, y el `ContextVar` lo hace explicito sin
ensanchar la firma de las seis funciones que renderizan.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

_MENSAJE_SIN_COLECTOR = (
    "el encoder pedido {pedido} cayo a {usado} y no hay colector activo: "
    "la degradacion no se puede registrar"
)
_MENSAJE_EVENTO = "encoder pedido {pedido}, usado {usado}"


@dataclass
class Degradaciones:
    """Colector mutable de eventos de degradacion de una corrida.

    Es mutable porque acumula a lo largo de la corrida. No es un valor: es el
    cuaderno de bitacora de una, y por eso nadie lo copia ni lo compara.
    """

    eventos: list[str] = field(default_factory=list)

    def agregar(self, pedido: str, usado: str) -> None:
        """Anota una caida de encoder, nombrando el pedido y el usado.

        Args:
            pedido: Codec que se pidio (p. ej. ``h264_nvenc``).
            usado: Codec con el que se acabo codificando (p. ej. ``libx2464``).
        """
        self.eventos.append(_MENSAJE_EVENTO.format(pedido=pedido, usado=usado))

    def como_tupla(self) -> tuple[str, ...]:
        """Copia inmutable de los eventos anotados.

        Returns:
            Los eventos en orden de aparicion.
        """
        return tuple(self.eventos)


_ACTIVAS: ContextVar[Degradaciones | None] = ContextVar("kliptych_degradaciones", default=None)


@contextmanager
def recolectar() -> Generator[Degradaciones]:
    """Abre un colector de degradaciones para el bloque.

    Anidar reutiliza el colector de mas afuera: una degradacion que ocurre
    dentro de otra sigue siendo de la misma corrida, y dividiria el registro
    en dos sitios que ningun lector miraria juntos.

    Yields:
        El colector donde caen los eventos del bloque.
    """
    previas = _ACTIVAS.get()
    actuales = previas if previas is not None else Degradaciones()
    token = _ACTIVAS.set(actuales)
    try:
        yield actuales
    finally:
        _ACTIVAS.reset(token)


def activas() -> Degradaciones | None:
    """Devuelve el colector de la corrida en curso, si hay uno.

    Returns:
        El colector activo, o `None` si nadie abrio uno.
    """
    return _ACTIVAS.get()


def registrar_caida(pedido: str, usado: str) -> None:
    """Anota una caida de encoder en el colector activo.

    Fallar sin colector es a proposito: si la degradacion no se puede registrar,
    la corrida revienta en vez de publicar un artefacto que miente. Una caida
    silenciosa es peor que una corrida fallida.

    Args:
        pedido: Codec que se pidio.
        usado: Codec con el que se acabo codificando.

    Raises:
        DegradacionNoRegistrableError: Si no hay colector activo.
    """
    colector = _ACTIVAS.get()
    if colector is None:
        msg = _MENSAJE_SIN_COLECTOR.format(pedido=pedido, usado=usado)
        raise DegradacionNoRegistrableError(msg)
    colector.agregar(pedido, usado)


class DegradacionNoRegistrableError(RuntimeError):
    """La degradacion ocurrio y no hay donde registrarla."""
