"""
Configuration and utilities for consistent logging across the application.
"""

import logging
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel

# Create a shared console instance
console = Console()

# Configure the root logger with Rich formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)
    ],
)

# Silence verbose logging from external libraries
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.WARNING)


def get_logger(name: str):
    """
    Get a configured logger with the specified name.

    Args:
        name: Name for the logger, typically the module name

    Returns:
        A configured logger instance
    """
    return logging.getLogger(f"hinbox.{name}")


def log(
    message: str,
    level: str = "info",
    exception: Optional[Exception] = None,
    logger=None,
):
    """
    Unified logging function with rich formatting.

    Args:
        message: The message to log
        level: Log level (info, warning, error, debug, success, processing)
        exception: Optional exception to include in error messages
        logger: Logger to use (if None, uses the root hinbox logger)
    """
    if logger is None:
        logger = logging.getLogger("hinbox")

    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        if exception:
            logger.error(f"{message}: {str(exception)}")
        else:
            logger.error(message)
    elif level == "debug":
        logger.debug(message)
    elif level == "success":
        # Custom success level (still logged as INFO)
        logger.info(f"[bold green]✓[/] {message}")
    elif level == "processing":
        # Custom processing level for workflow steps
        logger.info(f"[blue]{message}[/]")
    else:
        raise ValueError(
            f"Unsupported log level '{level}'. Supported levels are: "
            "'info', 'warning', 'error', 'debug', 'success', 'processing'."
        )


class DecisionKind(str, Enum):
    """Merge-loop outcome for a single entity."""

    NEW = "NEW"
    MERGE = "MERGE"
    SKIP = "SKIP"
    DISPUTE = "DISPUTE"
    DEFER = "DEFER"
    ERROR = "ERROR"


_DECISION_COLORS = {
    DecisionKind.NEW: "green",
    DecisionKind.MERGE: "yellow",
    DecisionKind.SKIP: "dim",
    DecisionKind.DISPUTE: "magenta",
    DecisionKind.DEFER: "cyan",
    DecisionKind.ERROR: "red",
}


def log_decision(
    kind: DecisionKind,
    entity_type: str,
    name: str,
    detail: str = "",
) -> None:
    """Emit a single colour-coded decision line for one entity.

    Example output:
      MERGE   person  "John Smith"  similarity=0.87
    """
    color = _DECISION_COLORS.get(kind, "white")
    badge = f"[bold {color}]{kind.value:<8}[/]"
    singular = entity_type.rstrip("s") if entity_type.endswith("s") else entity_type
    parts = [badge, f"{singular:<13}", f'"{name}"']
    if detail:
        parts.append(f" {detail}")
    line = "  ".join(parts)
    logging.getLogger("hinbox").info(line)


# Module-level flag controlling Rich profile panels.
# Off by default; set via set_show_profiles() or --show-profiles CLI flag.
_show_profiles = False


def set_show_profiles(enabled: bool = True) -> None:
    """Toggle display of full-profile Rich panels."""
    global _show_profiles
    _show_profiles = enabled


def display_markdown(content: str, title: str = None, style: str = "green"):
    """
    Display content with markdown formatting in a panel.

    Gated behind the --show-profiles flag: when disabled (the default),
    this function is a no-op to keep output compact.

    Args:
        content: Markdown content to display
        title: Optional title for the panel
        style: Border style color for the panel
    """
    if not _show_profiles:
        return
    console.print(
        Panel(
            Markdown(content),
            title=title,
            border_style=style,
        )
    )


# Configure default verbosity
def set_verbose(verbose: bool = False):
    """
    Set the verbosity level for all hinbox loggers.

    Args:
        verbose: If True, sets logging level to DEBUG; otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("hinbox").setLevel(level)
