"""
    Module for project logging setup.

    Two logging scopes are supported:
    - a shared csslib library log;
    - CSS-specific logs stored in the current result directory.
"""

__all__ = [
    "get_css_logger",
    "get_tools_logger",
    "get_config_logger",
    "get_exceptions_logger",
    "get_supercell_worker_logger",
    "get_collect_worker_logger",
]

import logging
import os
import threading
from multiprocessing import current_process


_LOCK = threading.RLock()
_ROOT_LOGGER_NAME = "csslib"
_LIBRARY_LOGGER_NAME = "csslib.library"
_TOOLS_LOGGER_NAME = "csslib.tools"
_CONFIG_LOGGER_NAME = "csslib.config"
_EXCEPTIONS_LOGGER_NAME = "csslib.exceptions"
_CSS_LOGGER_NAME = "csslib.css"
_SUPERCELL_WORKER_LOGGER_NAME = "csslib.supercell_worker"
_COLLECT_WORKER_LOGGER_NAME = "csslib.collect_worker"
_LIBRARY_LOG_ENV_VAR = "CSSLIB_LIBRARY_LOG_PATH"
_DEFAULT_LIBRARY_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csslib.log")

_CONSOLE_FORMATTER = logging.Formatter("%(levelname)s - %(message)s")
_FILE_FORMATTER = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] - %(message)s", "%Y-%m-%d %H:%M:%S")


class _ExactLoggerNameFilter(logging.Filter):
    """Allows records only from one exact logger name."""

    def __init__(self, logger_name: str):
        super().__init__()
        self._logger_name = logger_name

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == self._logger_name


def _get_library_log_path() -> str:
    """Returns the shared csslib log path."""

    configured_path = os.getenv(_LIBRARY_LOG_ENV_VAR)
    return configured_path if configured_path else _DEFAULT_LIBRARY_LOG_PATH


def _configure_handler(handler: logging.Handler, key: str, formatter: logging.Formatter, level: int) -> logging.Handler:
    """Applies shared configuration and labels a handler for idempotent reuse."""

    handler.setLevel(level)
    handler.setFormatter(formatter)
    setattr(handler, "_csslib_handler_key", key)
    return handler


def _ensure_logger(logger_name: str, *, level: int = logging.DEBUG, propagate: bool = True) -> logging.Logger:
    """Returns a configured logger with stable propagation rules."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = propagate
    return logger


def _ensure_handler(logger: logging.Logger, key: str, factory) -> logging.Handler:
    """Attaches a handler once per logger and key."""

    with _LOCK:
        for handler in logger.handlers:
            if getattr(handler, "_csslib_handler_key", None) == key:
                return handler
        handler = factory()
        logger.addHandler(handler)
        return handler


def _ensure_root_logger() -> logging.Logger:
    """Configures the csslib root logger that writes the shared library log."""

    root_logger = _ensure_logger(_ROOT_LOGGER_NAME, propagate=False)

    def _build_library_handler():
        library_log_path = _get_library_log_path()
        os.makedirs(os.path.dirname(library_log_path), exist_ok=True)
        return _configure_handler(
            logging.FileHandler(library_log_path, mode="a", encoding="utf-8"),
            f"library-file::{os.path.abspath(library_log_path)}",
            _FILE_FORMATTER,
            logging.DEBUG,
        )

    _ensure_handler(root_logger, f"library-file::{os.path.abspath(_get_library_log_path())}", _build_library_handler)
    return root_logger


def _ensure_console_logger(logger_name: str, *, exact_name_only: bool) -> logging.Logger:
    """Configures a logger that should emit user-facing console messages."""

    _ensure_root_logger()
    logger = _ensure_logger(logger_name, propagate=True)

    def _build_console_handler():
        handler = _configure_handler(
            logging.StreamHandler(),
            f"console::{logger_name}::{int(exact_name_only)}",
            _CONSOLE_FORMATTER,
            logging.INFO,
        )
        if exact_name_only:
            handler.addFilter(_ExactLoggerNameFilter(logger_name))
        return handler

    _ensure_handler(logger, f"console::{logger_name}::{int(exact_name_only)}", _build_console_handler)
    return logger


def _attach_result_file_handler(logger_name: str, result_path: str, log_filename: str) -> logging.Logger:
    """Attaches a result-scoped file handler to a logger."""

    _ensure_root_logger()
    logger = _ensure_logger(logger_name, propagate=True)
    logs_dir = os.path.join(result_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{log_filename}.log")
    handler_key = f"result-file::{logger_name}::{os.path.abspath(log_path)}"

    def _build_result_handler():
        return _configure_handler(
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            handler_key,
            _FILE_FORMATTER,
            logging.DEBUG,
        )

    _ensure_handler(logger, handler_key, _build_result_handler)
    return logger


def _resolve_logger_name(base_logger_name: str, name: str | None) -> str:
    """Builds a child logger name from a short or fully qualified name."""

    if not name:
        return base_logger_name
    if name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return name
    if name == _ROOT_LOGGER_NAME:
        return name
    if name.startswith("csslib_"):
        return name.replace("_", ".", 1)
    if name.startswith("csslib."):
        return name
    return f"{base_logger_name}.{name}"


def get_css_logger(result_path: str) -> logging.Logger:
    """
        Configures the CSS logger and returns it.

        Args:
            result_path (str): path to the CSS results directory.

        Return:
            Logger.
    """

    logger = _ensure_console_logger(_CSS_LOGGER_NAME, exact_name_only=True)
    return _attach_result_file_handler(logger.name, result_path, "main")


def get_tools_logger(name: str | None = None) -> logging.Logger:
    """
        Configures the tools logger and returns it.

        Args:
            name (str | None, optional): optional child logger name.

        Return:
            Logger.
    """

    _ensure_console_logger(_TOOLS_LOGGER_NAME, exact_name_only=False)
    return _ensure_logger(_resolve_logger_name(_TOOLS_LOGGER_NAME, name), propagate=True)


def get_config_logger() -> logging.Logger:
    """
        Configures the config logger and returns it.

        Return:
            Logger.
    """

    return _ensure_console_logger(_CONFIG_LOGGER_NAME, exact_name_only=True)


def get_exceptions_logger() -> logging.Logger:
    """
        Configures the exceptions logger and returns it.

        Return:
            Logger.
    """

    _ensure_root_logger()
    return _ensure_logger(_EXCEPTIONS_LOGGER_NAME, propagate=True)


def get_supercell_worker_logger(result_path: str) -> logging.Logger:
    """
        Configures the supercell worker logger and returns it.

        Args:
            result_path (str): path to the CSS results directory.

        Return:
            Logger.
    """

    return _attach_result_file_handler(
        _SUPERCELL_WORKER_LOGGER_NAME,
        result_path,
        f"supercell_{current_process().name}",
    )


def get_collect_worker_logger(result_path: str) -> logging.Logger:
    """
        Configures the collect worker logger and returns it.

        Args:
            result_path (str): path to the CSS results directory.

        Return:
            Logger.
    """

    return _attach_result_file_handler(
        _COLLECT_WORKER_LOGGER_NAME,
        result_path,
        f"collect_{current_process().name}",
    )

