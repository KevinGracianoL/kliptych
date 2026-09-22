"""Configuración de logging estructurado en una línea JSON por registro."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TextIO, override

_STANDARD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Formatea registros como JSON con campos estables y atributos extra."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Serializa un registro incluyendo extras y traceback.

        Args:
            record: Registro de logging a serializar.

        Returns:
            Una línea JSON con timestamp, nivel, logger, mensaje y extras.
        """
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        record_vars: dict[str, object] = vars(record)
        payload.update(
            (key, value) for key, value in record_vars.items() if key not in _STANDARD_ATTRS
        )
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(*, level: str = "INFO", stream: TextIO | None = None) -> None:
    """Configura el logging raíz con salida JSON en una línea por registro.

    Args:
        level: Nombre del nivel de logging (por ejemplo ``"INFO"``).
        stream: Stream de salida; por defecto ``sys.stderr``.

    Raises:
        ValueError: Si ``level`` no es un nivel de logging reconocido.
    """
    normalized = level.upper()
    if normalized not in logging.getLevelNamesMapping():
        msg = f"nivel de logging inválido: {level!r}"
        raise ValueError(msg)
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(normalized)
