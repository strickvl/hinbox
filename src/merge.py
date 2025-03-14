from typing import Any, Dict, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.constants import (
    CLOUD_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
)
from src.embeddings import embed_text
from src.profiles import create_profile, update_profile
from src.utils import write_entity_to_file

console = Console()


def merge_people(
    extracted_people: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
):
    embedding_model = (
        LOCAL_EMBEDDING_MODEL if model_type == "ollama" else CLOUD_EMBEDDING_MODEL
    )
    for p in extracted_people:
        person_name = p.get("name", "")
        if not person_name:
            continue

        entity_updated = False

        if person_name in entities["people"]:
            existing_person = entities["people"][person_name]
            article_exists = any(
                a.get("article_id") == article_id for a in existing_person["articles"]
            )
            if not article_exists:
                existing_person["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile with new information
                if "profile" in existing_person:
                    console.print(
                        f"\n[yellow]Updating profile for person:[/] {person_name}"
                    )
                    existing_person["profile"] = update_profile(
                        "person",
                        person_name,
                        existing_person["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    existing_person["profile_embedding"] = embed_text(
                        existing_person["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_person["profile"]["text"]),
                            title=f"Updated Profile: {person_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for person:[/] {person_name}"
                    )
                    existing_person["profile"] = create_profile(
                        "person", person_name, article_content, article_id, model_type
                    )
                    console.print(
                        Panel(
                            Markdown(existing_person["profile"]["text"]),
                            title=f"New Profile: {person_name}",
                            border_style="green",
                        )
                    )

            # Update extraction timestamp to earliest
            existing_timestamp = existing_person.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_person["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("people", person_name, existing_person)
                entities["people"][person_name] = existing_person
                console.print(
                    f"[blue]Updated person entity saved to file:[/] {person_name}"
                )
        else:
            # Create new entry with initial profile
            console.print(f"\n[green]Creating profile for new person:[/] {person_name}")
            profile = create_profile(
                "person", person_name, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {person_name}",
                    border_style="green",
                )
            )

            new_person = {
                "name": person_name,
                "type": p.get("type", ""),
                "profile": profile,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "profile_embedding": embed_text(
                    profile["text"], model_name=embedding_model
                ),
                "extraction_timestamp": extraction_timestamp,
            }

            entities["people"][person_name] = new_person
            write_entity_to_file("people", person_name, new_person)
            console.print(f"[green]New person entity saved to file:[/] {person_name}")


def merge_locations(
    extracted_locations: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
):
    embedding_model = (
        LOCAL_EMBEDDING_MODEL if model_type == "ollama" else CLOUD_EMBEDDING_MODEL
    )
    for loc in extracted_locations:
        loc_name = loc.get("name", "")
        loc_type = loc.get("type", "")
        if not loc_name:
            continue
        location_key = (loc_name, loc_type)

        entity_updated = False

        if location_key in entities["locations"]:
            existing_loc = entities["locations"][location_key]
            article_exists = any(
                a.get("article_id") == article_id for a in existing_loc["articles"]
            )
            if not article_exists:
                existing_loc["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile
                if "profile" in existing_loc:
                    console.print(
                        f"\n[yellow]Updating profile for location:[/] {loc_name}"
                    )
                    existing_loc["profile"] = update_profile(
                        "location",
                        loc_name,
                        existing_loc["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    existing_loc["profile_embedding"] = embed_text(
                        existing_loc["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_loc["profile"]["text"]),
                            title=f"Updated Profile: {loc_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for location:[/] {loc_name}"
                    )
                    existing_loc["profile"] = create_profile(
                        "location", loc_name, article_content, article_id, model_type
                    )
                    console.print(
                        Panel(
                            Markdown(existing_loc["profile"]["text"]),
                            title=f"New Profile: {loc_name}",
                            border_style="green",
                        )
                    )

            existing_timestamp = existing_loc.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_loc["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("locations", location_key, existing_loc)
                entities["locations"][location_key] = existing_loc
                console.print(
                    f"[blue]Updated location entity saved to file:[/] {loc_name}"
                )
        else:
            # Create new entry
            console.print(f"\n[green]Creating profile for new location:[/] {loc_name}")
            profile = create_profile(
                "location", loc_name, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {loc_name}",
                    border_style="green",
                )
            )

            new_location = {
                "name": loc_name,
                "type": loc_type,
                "profile": profile,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "extraction_timestamp": extraction_timestamp,
            }

            entities["locations"][location_key] = new_location
            write_entity_to_file("locations", location_key, new_location)
            console.print(f"[green]New location entity saved to file:[/] {loc_name}")


def merge_organizations(
    extracted_orgs: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
):
    embedding_model = (
        LOCAL_EMBEDDING_MODEL if model_type == "ollama" else CLOUD_EMBEDDING_MODEL
    )
    for org in extracted_orgs:
        org_name = org.get("name", "")
        org_type = org.get("type", "")
        if not org_name:
            continue
        org_key = (org_name, org_type)

        entity_updated = False

        if org_key in entities["organizations"]:
            existing_org = entities["organizations"][org_key]
            article_exists = any(
                a.get("article_id") == article_id for a in existing_org["articles"]
            )
            if not article_exists:
                existing_org["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile
                if "profile" in existing_org:
                    console.print(
                        f"\n[yellow]Updating profile for organization:[/] {org_name}"
                    )
                    existing_org["profile"] = update_profile(
                        "organization",
                        org_name,
                        existing_org["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    # Embed updated profile text using appropriate model
                    embedding_model = (
                        LOCAL_EMBEDDING_MODEL
                        if model_type == "ollama"
                        else CLOUD_EMBEDDING_MODEL
                    )
                    existing_org["profile_embedding"] = embed_text(
                        existing_org["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_org["profile"]["text"]),
                            title=f"Updated Profile: {org_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for organization:[/] {org_name}"
                    )
                    existing_org["profile"] = create_profile(
                        "organization",
                        org_name,
                        article_content,
                        article_id,
                        model_type,
                    )
                    console.print(
                        Panel(
                            Markdown(existing_org["profile"]["text"]),
                            title=f"New Profile: {org_name}",
                            border_style="green",
                        )
                    )

            existing_timestamp = existing_org.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_org["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("organizations", org_key, existing_org)
                entities["organizations"][org_key] = existing_org
                console.print(
                    f"[blue]Updated organization entity saved to file:[/] {org_name}"
                )
        else:
            # Create new entry
            console.print(
                f"\n[green]Creating profile for new organization:[/] {org_name}"
            )
            profile = create_profile(
                "organization", org_name, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {org_name}",
                    border_style="green",
                )
            )

            new_org = {
                "name": org_name,
                "type": org_type,
                "profile": profile,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "extraction_timestamp": extraction_timestamp,
            }

            entities["organizations"][org_key] = new_org
            write_entity_to_file("organizations", org_key, new_org)
            console.print(
                f"[green]New organization entity saved to file:[/] {org_name}"
            )


def merge_events(
    extracted_events: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
):
    embedding_model = (
        LOCAL_EMBEDDING_MODEL if model_type == "ollama" else CLOUD_EMBEDDING_MODEL
    )
    for e in extracted_events:
        event_title = e.get("title", "")
        event_start_date = e.get("start_date", "")
        event_key = (event_title, event_start_date)
        if not event_title:
            continue

        entity_updated = False

        if event_key in entities["events"]:
            existing_event = entities["events"][event_key]
            article_exists = any(
                a.get("article_id") == article_id for a in existing_event["articles"]
            )
            if not article_exists:
                existing_event["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile
                if "profile" in existing_event:
                    console.print(
                        f"\n[yellow]Updating profile for event:[/] {event_title}"
                    )
                    existing_event["profile"] = update_profile(
                        "event",
                        event_title,
                        existing_event["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    # Embed updated profile text using appropriate model
                    embedding_model = (
                        LOCAL_EMBEDDING_MODEL
                        if model_type == "ollama"
                        else CLOUD_EMBEDDING_MODEL
                    )
                    existing_event["profile_embedding"] = embed_text(
                        existing_event["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_event["profile"]["text"]),
                            title=f"Updated Profile: {event_title}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for event:[/] {event_title}"
                    )
                    existing_event["profile"] = create_profile(
                        "event", event_title, article_content, article_id, model_type
                    )
                    console.print(
                        Panel(
                            Markdown(existing_event["profile"]["text"]),
                            title=f"New Profile: {event_title}",
                            border_style="green",
                        )
                    )

            existing_timestamp = existing_event.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_event["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("events", event_key, existing_event)
                entities["events"][event_key] = existing_event
                console.print(
                    f"[blue]Updated event entity saved to file:[/] {event_title}"
                )
        else:
            # Create new entry
            console.print(f"\n[green]Creating profile for new event:[/] {event_title}")
            profile = create_profile(
                "event", event_title, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {event_title}",
                    border_style="green",
                )
            )

            new_event = {
                "title": event_title,
                "description": e.get("description", ""),
                "event_type": e.get("event_type", ""),
                "start_date": event_start_date,
                "end_date": e.get("end_date", ""),
                "is_fuzzy_date": e.get("is_fuzzy_date", False),
                "tags": e.get("tags", []),
                "profile": profile,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "extraction_timestamp": extraction_timestamp,
            }

            entities["events"][event_key] = new_event
            write_entity_to_file("events", event_key, new_event)
            console.print(f"[green]New event entity saved to file:[/] {event_title}")
