"""Structlog configuration with JSON file output and console stdout output."""

import logging
import re
import sys
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Regime display: color + symbol prefix
REGIME_FMT: dict[str, str] = {
    "bear":    "\033[91m",   # bright red
    "neutral": "\033[93m",   # bright yellow
    "bull":    "\033[92m",   # bright green
}
_RESET = "\033[0m"


def regime_banner(label: str, ticker: str) -> str:
    """Return an ANSI-colored regime banner string for console logging."""
    color = REGIME_FMT.get(label.lower(), "")
    symbol = {"bear": "▼", "neutral": "◆", "bull": "▲"}.get(label.lower(), "●")
    return f"{color}{symbol} REGIME [{ticker}]: {label.upper()}{_RESET}"


def _strip_ansi(
    logger: Any, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor — remove ANSI escape codes from all string values."""
    for key, val in event_dict.items():
        if isinstance(val, str):
            event_dict[key] = _ANSI_RE.sub("", val)
    return event_dict


def configure_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """
    Call once at application startup.

    Always uses stdlib LoggerFactory so that add_logger_name (which reads
    ``logger.name``) works correctly in every code path.

    Parameters
    ----------
    log_level:
        Standard level string, e.g. ``"INFO"``, ``"DEBUG"``.
    log_file:
        Optional path to a newline-delimited JSON log file.
        If *None* only console (stdout) output is produced.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared pre-processors — add level, module name, ISO timestamp
    shared_pre: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # Console handler — always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(),
            ]
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    _strip_ansi,
                    structlog.processors.JSONRenderer(),
                ]
            )
        )
        root.addHandler(file_handler)

    structlog.configure(
        processors=shared_pre,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog bound logger tagged with *name* as the module field.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)
