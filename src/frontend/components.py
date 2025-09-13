"""Reusable UI components following FastHTML patterns.

This module contains reusable components that can be used across different entity routes
to improve consistency and reduce code duplication. Components support HTMX interactions
and follow a hybrid UX approach where form controls require explicit Apply actions.
"""

from typing import Any

from fasthtml.common import *

import src.constants as constants

from . import utils as frontend_utils
from .app_config import main_layout
from .utils import random_pastel_color


def EmptyState(message: str, cls: str = "empty-state") -> Div:
    """Consistent empty state component for when no results are found.

    Creates a styled message container to display when filters or searches
    return no results, providing user feedback and guidance.

    Args:
        message: Text message to display to the user
        cls: CSS class name for styling the empty state

    Returns:
        Div component with the empty state message
    """
    return Div(
        message,
        cls=cls,
    )


def SearchInput(
    name: str,
    value: str = "",
    placeholder: str = "Search...",
    cls: str = "search-box",
    **attrs: Any,
) -> Div:
    """Reusable search input component that respects our hybrid UX pattern.

    Creates a labeled text input for search functionality. Does NOT include
    auto-trigger HTMX attributes by default, following the hybrid UX approach
    where text inputs require an explicit Apply button.

    Args:
        name: Form input name attribute
        value: Initial value for the input field
        placeholder: Placeholder text for the input
        cls: CSS class name for styling
        **attrs: Additional HTML attributes to pass to the input element

    Returns:
        Div component containing label and input elements

    Note:
        Additional HTMX attributes can be added via **attrs for specific use cases.
    """
    return Div(
        Label("Search: "),
        Input(
            type="text",
            name=name,
            value=value,
            placeholder=placeholder,
            style="width:100%; margin-top:5px;",
            **attrs,
        ),
        cls=cls,
    )


def EntityCount(count: int, entity_type_name: str) -> Div:
    """Consistent count display for entity results.

    Creates a styled count display showing how many entities match the current
    filters, providing users with feedback about result set size.

    Args:
        count: Number of entities found
        entity_type_name: Display name for the entity type (e.g., "People", "Events")

    Returns:
        Div component with formatted count text
    """
    return Div(
        f"{count} {entity_type_name.lower()} found",
        style="margin-bottom:15px; color:var(--text-light);",
    )


def ClearFiltersButton(
    route: str,
    cls: str = "contrast outline",
    style: str = "width:100%; margin-bottom:15px; font-weight:bold;",
) -> Button:
    """Consistent clear filters button across entity types.

    Creates a button that redirects to the base route to clear all applied filters,
    providing users with a quick way to reset their search.

    Args:
        route: Base route URL to redirect to for clearing filters
        cls: CSS classes for button styling
        style: Inline CSS styles for the button

    Returns:
        Button component with onclick navigation to clear filters
    """
    return Button(
        "Clear Filters",
        cls=cls,
        onclick=f"window.location.href='{route}'",
        style=style,
    )


def ApplyFiltersButton(
    route: str,
    cls: str = "primary",
    style: str = "width:100%; margin-top:15px; font-weight:bold;",
) -> Button:
    """Consistent apply filters button for our hybrid UX approach.

    Creates a submit button for filter forms, implementing the hybrid UX pattern
    where users must explicitly apply their filter selections.

    Args:
        route: Form action route (passed for consistency, but uses form submission)
        cls: CSS classes for button styling
        style: Inline CSS styles for the button

    Returns:
        Button component configured as form submit button
    """
    return Button(
        "Apply Filters",
        type="submit",
        cls=cls,
        style=style,
    )


def FilterForm(route: str, *content: Any, **attrs: Any) -> Form:
    """Wrapper for filter forms with consistent HTMX configuration.

    Creates a form element with standard HTMX attributes for dynamic content updates,
    targeting the content area and using GET method for filter submissions.

    Args:
        route: Form action URL for filter submission
        *content: Child components to include in the form
        **attrs: Additional form attributes (merged with defaults)

    Returns:
        Form component with HTMX configuration for dynamic updates

    Note:
        Default attributes can be overridden by passing them in **attrs.
    """
    default_attrs = {
        "method": "get",
        "action": route,
        "hx_get": route,
        "hx_target": ".content-area",
        "hx_swap": "innerHTML",
    }
    default_attrs.update(attrs)

    return Form(*content, **default_attrs)


def TypeChip(chip_name, chip_value, is_checked, target_route, param_name="type"):
    """Reusable type chip component following our chip_checkbox pattern."""
    return Label(
        Input(
            type="checkbox",
            name=param_name,
            value=chip_value,
            checked="checked" if is_checked else None,
            style="display:none;",
            onchange="this.parentElement.classList.toggle('selected', this.checked);",
            hx_trigger="change",
            hx_get=target_route,
            hx_target=".content-area",
            hx_swap="innerHTML",
            hx_include="[name]",
        ),
        chip_name.capitalize(),
        cls=f"filter-chip{' selected' if is_checked else ''}",
        style=f"background-color: {random_pastel_color(chip_value)};",
    )


def TypeChipsSection(
    title, chips_data, target_route, param_name="type", style="margin-bottom:15px;"
):
    """Section containing multiple type chips."""
    if not chips_data:
        return ""

    return Div(
        H4(title),
        *[
            TypeChip(chip_name, chip_value, is_checked, target_route, param_name)
            for chip_name, chip_value, is_checked in chips_data
        ],
        style=style,
    )


def EntityCard(title, subtitle="", badge="", content="", href="#", cls="entity-card"):
    """Reusable card component for entity display."""
    card_content = [
        A(title, href=href, cls="entity-link") if href != "#" else H4(title)
    ]

    if subtitle:
        card_content.append(P(subtitle, cls="entity-subtitle"))

    if badge:
        card_content.append(Span(badge, cls="tag"))

    if content:
        card_content.append(Div(content, cls="entity-content"))

    return Div(*card_content, cls=cls)


def EntityList(entities, entity_type_name, route_prefix, render_func=None):
    """Generic entity list renderer with consistent structure."""
    if not entities:
        return EmptyState(
            f"No {entity_type_name.lower()} match your filters. Try adjusting your criteria."
        )

    items = []
    for entity_data in entities:
        if render_func:
            items.append(render_func(entity_data))
        else:
            # Default simple rendering
            if isinstance(entity_data, tuple) and len(entity_data) >= 2:
                k, entity = entity_data[0], entity_data[1]
            else:
                k, entity = entity_data, entity_data

            # Create default list item
            type_badge = ""
            if entity.get("type"):
                type_badge = Span(entity.get("type"), cls="tag")

            name = entity.get("name") or entity.get("title", "Unknown")
            link = A(name, href=f"/{route_prefix}/{frontend_utils.encode_key(k)}")

            items.append(Li(link, " ", type_badge))

    return Div(EntityCount(len(entities), entity_type_name), Ul(*items), id="results")


def PageWithSidebar(title, sidebar_content, main_content, **layout_attrs):
    """Layout wrapper for pages with sidebar (filter panel) and main content."""
    return main_layout(title, sidebar_content, main_content, **layout_attrs)


def DateRangeInputs(start_date="", end_date="", cls="date-range"):
    """Reusable date range inputs for filtering."""
    return Div(
        H4("Date Range"),
        Div(
            Label("From:"),
            Input(
                type="date",
                name="start_date",
                value=start_date if start_date else None,
            ),
            style="margin-bottom:10px;",
        ),
        Div(
            Label("To:"),
            Input(
                type="date",
                name="end_date",
                value=end_date if end_date else None,
            ),
            style="margin-bottom:10px;",
        ),
        cls=cls,
    )


def ProfileVersionSelector(
    entity_name: str,
    entity_type: str,
    current_version: int,
    total_versions: int,
    route_prefix: str,
    entity_key: str,
    selected_version: int = None,
):
    """Dropdown selector for profile versions."""
    if not constants.ENABLE_PROFILE_VERSIONING or total_versions <= 1:
        return ""

    selected_version = selected_version or current_version

    options = []
    for v in range(total_versions, 0, -1):  # Latest first
        label = f"Version {v}"
        if v == current_version:
            label += " (Current)"

        options.append(
            Option(
                label,
                value=str(v),
                selected="selected" if v == selected_version else None,
            )
        )

    return Div(
        Label("Profile Version:", style="font-weight: bold; margin-right: 10px;"),
        Select(
            *options,
            name="version",
            onchange=f"window.location.href = '/{route_prefix}/{frontend_utils.encode_key(entity_key)}?version=' + this.value",
            style="margin-right: 10px;",
        ),
        style="margin-bottom: 20px; padding: 15px; background: var(--surface-2); border-radius: 8px;",
    )
