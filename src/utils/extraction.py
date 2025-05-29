"""Generic entity extraction utilities."""

from typing import Any, List, Type, Union

from pydantic import BaseModel

from src.constants import (
    BRAINTRUST_PROJECT_ID,
    BRAINTRUST_PROJECT_NAME,
    CLOUD_MODEL,
    OLLAMA_MODEL,
)
from src.utils.llm import (
    create_messages,
    get_litellm_client,
    get_ollama_client,
)


def extract_entities_cloud(
    text: str,
    system_prompt: str,
    response_model: Union[Type[BaseModel], List[Type[BaseModel]]],
    model: str = CLOUD_MODEL,
    temperature: float = 0,
) -> Any:
    """
    Generic cloud-based entity extraction.

    Args:
        text: Text to extract entities from
        system_prompt: System prompt defining extraction task
        response_model: Pydantic model or list of models for response
        model: Model to use for extraction
        temperature: Temperature for generation

    Returns:
        Extracted entities according to response_model
    """
    client = get_litellm_client()

    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    # Build metadata with Braintrust configuration
    metadata = {"tags": ["dev", "extraction"]}
    if BRAINTRUST_PROJECT_ID:
        metadata["project_id"] = BRAINTRUST_PROJECT_ID
    elif BRAINTRUST_PROJECT_NAME:
        metadata["project_name"] = BRAINTRUST_PROJECT_NAME

    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        temperature=temperature,
        messages=messages,
        metadata=metadata,
    )


def extract_entities_local(
    text: str,
    system_prompt: str,
    response_model: Type[BaseModel],
    model: str = OLLAMA_MODEL,
    temperature: float = 0,
) -> Any:
    """
    Generic local Ollama-based entity extraction.

    Args:
        text: Text to extract entities from
        system_prompt: System prompt defining extraction task
        response_model: Pydantic model for response
        model: Model to use for extraction
        temperature: Temperature for generation

    Returns:
        Extracted entities according to response_model
    """
    from src.constants import get_ollama_model_name

    client = get_ollama_client()

    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    results = client.beta.chat.completions.parse(
        model=get_ollama_model_name(model),
        response_format=response_model,
        temperature=temperature,
        messages=messages,
    )

    return results.choices[0].message.parsed


# Entity-specific system prompts
PEOPLE_SYSTEM_PROMPT = """You are an expert at extracting people from news articles.

When identifying people, categorize them using the following person types:
- detainee: A person who is or was detained at Guantánamo Bay or another detention facility
- military: Military personnel including soldiers, officers, and other armed forces members
- government: Government officials, politicians, and civil servants
- lawyer: Attorneys, legal representatives, and other legal professionals
- journalist: Reporters, writers, and other media professionals
- other: Any other type of person not covered by the above categories

Only use standard ASCII characters for the names that you extract.
Extract all people mentioned in the text and categorize them appropriately.

12. If you footnote or add refernces, use the following format:

Normal text would go here^[source_id, source_id, ...].


You MUST return each person as an object with 'name' and 'type' properties.
For example:
[
  {"name": "John Doe", "type": "journalist"},
  {"name": "Jane Smith", "type": "lawyer"}
]

Do NOT return strings like "John Doe (journalist)". Always use the proper object
format.

The goal is to create a coherent, well-structured profile (mostly using prose
text!) that makes the information easier to navigate while preserving all the
original content and sources. Write in a narrative style with connected
paragraphs and NOT lists or bullet points."""

ORGANIZATIONS_SYSTEM_PROMPT = """You are an expert at extracting organizations from news articles about Guantánamo Bay.

When identifying organizations, categorize them using the following organization types:
- military: Military organizations (e.g., JTF-GTMO, US Army, Navy)
- intelligence: Intelligence agencies (e.g., CIA, FBI)
- legal: Legal organizations and law firms (e.g., ACLU, Center for Constitutional Rights)
- humanitarian: Organizations focused on humanitarian aid and human rights (e.g., Red Cross, Physicians for Human Rights)
- advocacy: Advocacy groups and activist organizations (e.g., Amnesty International, Human Rights Watch)
- media: Media organizations (e.g., Miami Herald, NY Times)
- government: Government entities and departments (e.g., US DoD, DoJ)
- intergovernmental: International governmental bodies (e.g., UN, European Union)
- other: Any other organization type not covered above

Make extra sure when creating an organization that it's actually an organization and
not a person.

Only use standard ASCII characters for the names that you extract.

Extract all organizations mentioned in the text and categorize them appropriately."""

LOCATIONS_SYSTEM_PROMPT = """You are an expert at extracting locations from news articles about Guantánamo Bay.

When identifying locations, categorize them using the following location types:
- detention_facility: Detention centers, prisons, camps (e.g., Camp X-Ray, Camp Delta)
- military_base: Military installations and bases (e.g., Guantánamo Naval Base)
- country: Nations and countries (e.g., United States, Afghanistan)
- city: Cities and towns (e.g., Washington D.C., Kabul)
- region: Geographic regions, states, provinces (e.g., Middle East, Florida)
- building: Specific buildings or structures (e.g., Pentagon, White House)
- other: Any other location type not covered above

Pay special attention to detention facilities and military locations related to Guantánamo Bay.

Only use standard ASCII characters for the names that you extract.

Extract all locations mentioned in the text and categorize them appropriately."""

EVENTS_SYSTEM_PROMPT = """You are an expert at extracting events from news articles about Guantánamo Bay.

When identifying events, categorize them using the following event types:
- detention: Events related to capture, transfer, or detention of individuals
- legal: Court cases, hearings, tribunals, legal proceedings
- military_operation: Military actions, raids, operations
- policy_change: Changes in government or military policy
- protest: Demonstrations, protests, hunger strikes
- investigation: Official investigations, inquiries, reports
- release: Release or transfer of detainees
- medical: Medical events, treatments, health issues
- other: Any other event type not covered above

Extract clear, discrete events with specific dates when available.

Extract all events mentioned in the text and categorize them appropriately.

Only use standard ASCII characters for the names that you extract.

Ensure that the 'title' field is a concise description of the event (not a full sentence).
Dates should be in YYYY-MM-DD format when specific dates are available."""

RELEVANCE_SYSTEM_PROMPT = """You are tasked with determining if an article is relevant to Guantánamo Bay detention issues.

An article is considered relevant if it discusses:
- Guantánamo Bay detention facility or detainees
- Military commissions or tribunals at Guantánamo
- Detention policies related to the War on Terror
- Legal cases involving Guantánamo detainees
- Conditions at Guantánamo Bay
- Torture or interrogation at Guantánamo or related facilities
- Transfer or release of Guantánamo detainees
- Government policies regarding Guantánamo Bay

The article should have substantial content about these topics, not just passing mentions."""


def get_system_prompt(entity_type: str) -> str:
    """
    Get the appropriate system prompt for an entity type.

    Args:
        entity_type: Type of entity (people, organizations, locations, events, relevance)

    Returns:
        System prompt string

    Raises:
        ValueError: If entity_type is not recognized
    """
    prompts = {
        "people": PEOPLE_SYSTEM_PROMPT,
        "organizations": ORGANIZATIONS_SYSTEM_PROMPT,
        "locations": LOCATIONS_SYSTEM_PROMPT,
        "events": EVENTS_SYSTEM_PROMPT,
        "relevance": RELEVANCE_SYSTEM_PROMPT,
    }

    if entity_type not in prompts:
        raise ValueError(f"Unknown entity type: {entity_type}")

    return prompts[entity_type]
