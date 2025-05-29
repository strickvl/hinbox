"""Profile extraction and management utilities."""

from typing import Any, Dict, Optional

from src.logging_config import get_logger

# Get logger for this module
logger = get_logger("utils.profiles")


def extract_profile_text(
    profile_response: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Extract profile text from either a simple dict or nested API response.

    Args:
        profile_response: Response from create_profile() which could be either:
            - A simple dict with 'text' key
            - A nested API response with parsed content

    Returns:
        Dict with 'text' and other fields, or None if text cannot be extracted
    """
    if not profile_response:
        return None

    # Handle simple dict case
    if isinstance(profile_response, dict):
        if "text" in profile_response:
            return profile_response
        if "choices" in profile_response and len(profile_response["choices"]) > 0:
            message = profile_response["choices"][0].get("message", {})
            if "parsed" in message:
                return message["parsed"]

    return None


def merge_profile_data(
    existing_profile: Optional[Dict[str, Any]], new_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge new profile data with existing profile.

    Args:
        existing_profile: Existing profile data (may be None)
        new_data: New data to merge

    Returns:
        Merged profile data
    """
    if not existing_profile:
        return new_data

    # Create a copy to avoid mutation
    merged = dict(existing_profile)

    # Merge simple fields
    for key, value in new_data.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(value, list) and isinstance(merged[key], list):
            # Merge lists (avoiding duplicates if items are hashable)
            try:
                merged[key] = list(set(merged[key] + value))
            except TypeError:
                # If items aren't hashable, just concatenate
                merged[key] = merged[key] + value
        elif isinstance(value, dict) and isinstance(merged[key], dict):
            # Recursively merge dictionaries
            merged[key] = merge_profile_data(merged[key], value)
        else:
            # For other types, prefer the new value
            merged[key] = value

    return merged


def format_profile_for_display(profile: Dict[str, Any]) -> str:
    """
    Format a profile dictionary for human-readable display.

    Args:
        profile: Profile dictionary

    Returns:
        Formatted string representation
    """
    lines = []

    # Add name/title if present
    if "name" in profile:
        lines.append(f"# {profile['name']}")
        lines.append("")
    elif "title" in profile:
        lines.append(f"# {profile['title']}")
        lines.append("")

    # Add type/category if present
    if "type" in profile:
        lines.append(f"**Type:** {profile['type']}")
        lines.append("")

    # Add main text if present
    if "text" in profile:
        lines.append(profile["text"])
        lines.append("")

    # Add other fields
    skip_fields = {"name", "title", "type", "text", "reflection_history"}
    for key, value in profile.items():
        if key not in skip_fields and value:
            if isinstance(value, list):
                lines.append(f"**{key.replace('_', ' ').title()}:**")
                for item in value:
                    lines.append(f"- {item}")
                lines.append("")
            elif isinstance(value, dict):
                lines.append(f"**{key.replace('_', ' ').title()}:**")
                for k, v in value.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")
            else:
                lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
                lines.append("")

    return "\n".join(lines).strip()
