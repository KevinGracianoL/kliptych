"""Tests de la configuración de logging."""

import logging

from kliptych.logging_setup import setup_logging


def test_setup_logging_default_info() -> None:
    setup_logging()
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_verbose_debug() -> None:
    setup_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_quiet_error() -> None:
    setup_logging(quiet=True)
    assert logging.getLogger().level == logging.ERROR


def test_setup_logging_mutes_noisy_loggers() -> None:
    setup_logging()
    for name in ("urllib3", "httpcore", "httpx", "asyncio"):
        assert logging.getLogger(name).level == logging.WARNING
