"""
Enhanced logging utilities for the iterative improvement process.
"""

from datetime import datetime
from typing import Optional

from src.logging_config import get_logger, log

logger = get_logger("utils.iterative")


def log_iterative_start(prompt: str, model: str, max_iterations: int) -> datetime:
    """
    Log the start of an iterative improvement process.

    Args:
        prompt: The user prompt being processed
        model: The model being used
        max_iterations: Maximum iterations allowed

    Returns:
        Start time for timing calculations
    """
    # Truncate the prompt if it's too long for logging
    prompt_truncated = prompt[:100] + "..." if len(prompt) > 100 else prompt

    log(f"Starting iterative improvement with model: {model}", level="processing")
    log(f"Max iterations: {max_iterations}", level="debug")
    log(f"Prompt: {prompt_truncated}", level="debug")

    return datetime.now()


def log_generation_attempt(iteration: int, model: str) -> datetime:
    """
    Log a generation attempt in the iterative process.

    Args:
        iteration: Current iteration number
        model: The model being used

    Returns:
        Start time for timing the generation
    """
    log(f"Iteration {iteration + 1}: Generating response with {model}", level="debug")
    return datetime.now()


def log_generation_result(
    start_time: datetime, iteration: int, success: bool, error: Optional[str] = None
):
    """
    Log the result of a generation attempt.

    Args:
        start_time: When the generation started
        iteration: Current iteration number
        success: Whether generation succeeded
        error: Optional error message if generation failed
    """
    duration = (datetime.now() - start_time).total_seconds()

    if success:
        log(
            f"Iteration {iteration + 1}: Generated response in {duration:.2f}s",
            level="debug",
        )
    else:
        log(
            f"Iteration {iteration + 1}: Generation failed after {duration:.2f}s: {error}",
            level="error",
        )


def log_evaluation_start(iteration: int, model: str) -> datetime:
    """
    Log the start of an evaluation in the iterative process.

    Args:
        iteration: Current iteration number
        model: The evaluation model being used

    Returns:
        Start time for timing the evaluation
    """
    log(f"Iteration {iteration + 1}: Evaluating response with {model}", level="debug")
    return datetime.now()


def log_evaluation_result(
    start_time: datetime,
    iteration: int,
    passed: bool,
    reason: str,
    feedback: Optional[str] = None,
):
    """
    Log the result of an evaluation.

    Args:
        start_time: When the evaluation started
        iteration: Current iteration number
        passed: Whether the evaluation passed
        reason: Reason for the pass/fail
        feedback: Feedback for improvement if failed
    """
    duration = (datetime.now() - start_time).total_seconds()

    if passed:
        log(
            f"Iteration {iteration + 1}: ✓ Evaluation passed in {duration:.2f}s",
            level="success",
        )
        log(f"Reason: {reason}", level="debug")
    else:
        log(
            f"Iteration {iteration + 1}: ✗ Evaluation failed in {duration:.2f}s",
            level="warning",
        )
        log(f"Reason: {reason}", level="debug")
        log(f"Feedback: {feedback}", level="debug")


def log_iterative_complete(
    start_time: datetime, iterations_completed: int, max_iterations: int, success: bool
):
    """
    Log the completion of the iterative improvement process.

    Args:
        start_time: When the process started
        iterations_completed: How many iterations were performed
        max_iterations: Maximum iterations allowed
        success: Whether a successful response was generated
    """
    duration = (datetime.now() - start_time).total_seconds()

    if success:
        log(
            f"Iterative improvement completed successfully in {duration:.2f}s "
            f"after {iterations_completed}/{max_iterations} iterations",
            level="success",
        )
    else:
        log(
            f"Iterative improvement failed to converge in {duration:.2f}s "
            f"after {iterations_completed}/{max_iterations} iterations",
            level="warning",
        )
