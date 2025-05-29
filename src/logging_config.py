"""
Configuration and utilities for consistent logging across the application.
"""

import logging
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


def display_markdown(content: str, title: str = None, style: str = "green"):
    """
    Display content with markdown formatting in a panel.

    Args:
        content: Markdown content to display
        title: Optional title for the panel
        style: Border style color for the panel
    """
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
