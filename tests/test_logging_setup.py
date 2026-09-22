"""Tests del logging estructurado."""

import io
import json
import logging
from typing import cast

import pytest

from kliptych.logging_setup import JsonFormatter, configure_logging


def _parse(line: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(line))


def _record() -> logging.LogRecord:
    return logging.LogRecord(
        name="kliptych.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="pieza %s",
        args=("lista",),
        exc_info=None,
    )


def _raise_boom() -> None:
    msg = "boom"
    raise RuntimeError(msg)


def test_formatter_emits_stable_fields() -> None:
    payload = _parse(JsonFormatter().format(_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "kliptych.test"
    assert payload["message"] == "pieza lista"
    assert isinstance(payload["timestamp"], str)


def test_formatter_includes_extra_attributes() -> None:
    record = logging.getLogger("kliptych.test").makeRecord(
        name="kliptych.test",
        level=logging.INFO,
        fn=__file__,
        lno=1,
        msg="evento",
        args=(),
        exc_info=None,
        func=None,
        extra={"run_id": "run-123"},
    )
    payload = _parse(JsonFormatter().format(record))
    assert payload["run_id"] == "run-123"


def test_configure_logging_writes_json_lines() -> None:
    stream = io.StringIO()
    configure_logging(level="info", stream=stream)
    logging.getLogger("kliptych.test").info("hola")
    payload = _parse(stream.getvalue().strip())
    assert payload["message"] == "hola"
    assert payload["level"] == "INFO"


def test_configure_logging_includes_traceback() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", stream=stream)
    try:
        _raise_boom()
    except RuntimeError:
        logging.getLogger("kliptych.test").exception("falló")
    payload = _parse(stream.getvalue().strip())
    assert "RuntimeError: boom" in str(payload["exception"])


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="nivel de logging inválido"):
        configure_logging(level="NOPE", stream=io.StringIO())
