from typing import Any, Dict, List, Literal, Optional

import instructor
import litellm
from pydantic import BaseModel

from src.constants import CLOUD_MODEL

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


class Header(BaseModel):
    h2_header_str: str


class ArticleHeaders(BaseModel):
    headers: List[Header]


# Define the possible profile types
ProfileType = Literal["person", "location", "organization", "event"]


def restructure_text_with_headers(
    original_text: str, headers: List[Header], model: str = CLOUD_MODEL
) -> str:
    """
    Restructure the original text using the provided headers.

    Args:
        original_text: The original text to restructure
        headers: The headers to use for restructuring
        model: The model to use for restructuring

    Returns:
        The restructured text
    """
    client = instructor.from_litellm(litellm.completion)

    # Convert headers to a string format
    headers_text = "\n".join([f"## {header.h2_header_str}" for header in headers])

    system_prompt = """You are an expert at restructuring text into a well-organized profile document.

Your task is to take the original text and reorganize it using the provided headers. Follow these guidelines:

1. Preserve as much of the original information as possible
2. Maintain all source references and citations from the original text
3. Ensure all content is placed under the most appropriate header
4. Do not add new information that wasn't in the original text
5. Do not remove important details or context
6. Maintain the original tone and style of writing
7. If information fits under multiple headers, you may include it in the most relevant section
8. Format the output as a markdown document with the provided headers
9. IMPORTANT: Use flowing paragraphs of prose rather than bullet points
10. Connect related information into coherent narratives within each section
11. Ensure smooth transitions between ideas within each section
12. If you footnote or add refernces, use the following format:

Normal text would go here^[source_id, source_id, ...].

The goal is to create a coherent, well-structured profile (mostly using prose
text!) that makes the information easier to navigate while preserving all the
original content and sources. Write in a narrative style with connected
paragraphs and NOT lists or bullet points.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Here are the headers to use for restructuring:\n\n{headers_text}\n\nHere is the original text to restructure:\n\n{original_text}",
        },
    ]

    # For text generation, we can use str as the response_model
    response = client.chat.completions.create(
        model=model,
        response_model=str,
        messages=messages,
        temperature=0.3,
    )

    # When using response_model=str, the instructor client directly returns the string
    return response


def gemini_extract_headers(
    text: str, profile_type: Optional[ProfileType] = None, model: str = CLOUD_MODEL
) -> List[Dict[str, Any]]:
    """
    Extract structural headers from the provided text using Gemini.

    Args:
        text: The text to extract headers from
        profile_type: The type of profile (person, location, organization, event)
        model: The model to use for extraction

    Returns:
        A list of extracted headers
    """
    client = instructor.from_litellm(litellm.completion)

    # Base system prompt
    system_prompt = """You are an expert at extracting structural headers from text to organize information into coherent profiles.

Your task is to identify the main topics and themes that could serve as headers in a profile document. Focus on creating a comprehensive structural framework that covers all major information categories present in the text.

For each text sample:
1. Identify the main topics and themes covered
2. Create headers that would organize this information effectively
3. Ensure headers are clear, concise, and representative of the content
4. Focus only on creating meaningful structural elements (no content details)

"""

    # Add type-specific instructions
    if profile_type == "person":
        system_prompt += """For biographical profiles about individuals, consider headers such as:
- Family History and Background
- Cultural and Social Context
- Childhood and Early Years
- Trauma History and Adverse Experiences
- Educational Challenges and Opportunities
- Socioeconomic Factors
- Mental Health History and Conditions
- Substance Use History
- Developmental and Cognitive Factors
- Support Systems and Protective Factors
- Rehabilitation Efforts and Potential
- Community Ties and Social Capital
- Employment History and Skills
- Military Service (if applicable)
- Religious and Spiritual Beliefs
- Remorse and Acceptance of Responsibility
- Risk Factors for Criminal Behavior
- Mitigating Circumstances Related to the Offense
- Professional Background
- Affiliations and Associations
- Key Relationships and Networks
- Notable Activities and Events
- Relevant Incidents
- Witness Accounts and Testimonies
- Credibility Factors
- Behavioral Patterns
- Public Statements and Communications
- Financial Information
- Travel History
- Medical and Psychological Information
- Legal History
- Current Status and Circumstances
- Reliability Assessment
- Motivations and Potential Biases
- Corroborating Evidence
"""
    elif profile_type == "location":
        system_prompt += """For location profiles, consider headers such as:
- Geographic and Physical Context
- Historical Background and Significance
- Political and Legal Framework
- Security Situation and Risks
- Socioeconomic Conditions
- Cultural and Religious Dynamics
- Access to Resources and Services
- Environmental Factors and Challenges
- Human Rights Concerns
- Notable Incidents and Events
- Current Status and Developments
"""
    elif profile_type == "organization":
        system_prompt += """For organization profiles, consider headers such as:
- Origins and Development
- Structure and Leadership
- Mission and Stated Objectives
- Operational Methods and Activities
- Resources and Funding
- Membership and Recruitment
- Ideological Framework
- External Relationships and Affiliations
- Legal Status and Challenges
- Public Perception and Media Coverage
- Notable Actions and Incidents
- Internal Dynamics and Conflicts
- Current Status and Trajectory
"""
    elif profile_type == "event":
        system_prompt += """For event profiles, consider headers such as:
- Context and Background
- Chronology and Timeline
- Key Participants and Their Roles
- Precipitating Factors
- Environmental and Situational Conditions
- Witness Accounts and Perspectives
- Immediate Consequences and Impact
- Official Responses and Reactions
- Legal Proceedings and Outcomes
- Media Coverage and Public Narrative
- Long-term Implications
- Related Events and Patterns
"""
    else:
        # Generic headers for when no specific type is provided
        system_prompt += """Consider headers that would best organize the specific type of information present in the text, which might include categories such as:
- Background and History
- Key Figures/Features
- Timeline of Events
- Current Status
- Legal/Political Aspects
- Controversies
- Relationships and Connections
- Impact and Significance
- Media Coverage
- Future Outlook
"""

    system_prompt += "\nExtract all meaningful headers that would help organize the information into a coherent profile structure."

    results = client.chat.completions.create(
        model=model,
        response_model=ArticleHeaders,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    # The instructor client directly returns the parsed model
    return results.headers


if __name__ == "__main__":
    text = """Khalid Sheikh Mohammed is the alleged mastermind of the September 11 attacks95bbac2e-6e19-416a-94ca-9040353bda99. Military prosecutors claim to have tapes of telephone calls between Mohammed and three accused co-conspirators, discussing the plot in code months before the attacks95bbac2e-6e19-416a-94ca-9040353bda99. These calls were reportedly made between April and October 200195bbac2e-6e19-416a-94ca-9040353bda99. He is currently held at Guantánamo Bay awaiting trial95bbac2e-6e19-416a-94ca-9040353bda99. He was arraigned in 201295bbac2e-6e19-416a-94ca-9040353bda99. Terry McDermott, co-author of "The Hunt for KSM," said that U.S. satellites "randomly scooped up" calls between Mohammed and an alleged deputy, Ramzi Binalshibh95bbac2e-6e19-416a-94ca-9040353bda99. The article mentions that defense lawyers questioned whether voice samples of Mohammed were recorded during the years he was held in the CIA's secret prison system95bbac2e-6e19-416a-94ca-9040353bda99.

Confidence: 0.9

Articles
Article ID: 95bbac2e-6e19-416a-94ca-9040353bda99, Title: U.S. says it has tapes of alleged 9/11 mastermind plotting with co-conspirators (link)

Khalid Sheik Mohammed is the accused architect of the Sept. 11 attacks^[fb46f019-6ead-40d1-a9db-0f6198d5e262, 05057ae5-e727-4813-976b-1992cc57e4fa]. He is among five men charged in a death-penalty case alleging their conspiracy in the Sept. 11, 2001 hijackings that killed 2,976 people in New York, at the Pentagon and aboard an airliner that crashed in a Pennsylvania field^[fb46f019-6ead-40d1-a9db-0f6198d5e262, 05057ae5-e727-4813-976b-1992cc57e4fa]. Mohammed declared that he oversaw the 9/11 attacks "from A to Z" after three and a half years in CIA detention, including 183 rounds of waterboardingfb46f019-6ead-40d1-a9db-0f6198d5e262. He was held for years in covert CIA prisons^[fb46f019-6ead-40d1-a9db-0f6198d5e262, 05057ae5-e727-4813-976b-1992cc57e4fa] before being moved to military detention at Guantánamo Bay in September 2006^[fb46f019-6ead-40d1-a9db-0f6198d5e262, 05057ae5-e727-4813-976b-1992cc57e4fa].

His lawyer, David Nevin, threatened to refuse participation in a pretrial hearing due to concerns about the FBI questioning a former 9/11 defense team paralegal05057ae5-e727-4813-976b-1992cc57e4fa. Nevin cited concerns about attorney-client confidentiality, alleging intelligence agency interference, including listening devices and recruitment of defense team informants05057ae5-e727-4813-976b-1992cc57e4fa. The paralegal, Army Staff Sgt. Brent Skeete, stated that FBI agents sought information about the defense team's work, personalities, and communications during questioning at Fort Hood, Texas05057ae5-e727-4813-976b-1992cc57e4fa.

Mohammed's attorneys are also seeking to remove the new judge in the case, Marine Col. Keith Parrella, because of his prior work with the National Security Division of the Department of Justice05057ae5-e727-4813-976b-1992cc57e4fa. Nevin stated that the FBI's investigation of the sergeant appeared to be part of an unending effort to intimidate the defense teams05057ae5-e727-4813-976b-1992cc57e4fa.

Confidence: 0.85

Articles
Article ID: fb46f019-6ead-40d1-a9db-0f6198d5e262, Title: Did CIA Director Gina Haspel run a black site at Guantánamo? (link)
Article ID: 05057ae5-e727-4813-976b-1992cc57e4fa, Title: Sept. 11 defense lawyers threaten hearing boycott over FBI interview of paralegal (link)

Khalid Shaikh Mohammed is a prisoner at the military prison in Guantánamo Bay, Cuba0532a14f-a9af-4665-b955-293fe6aa874f. He is accused of plotting the 9/11 attacks0532a14f-a9af-4665-b955-293fe6aa874f. Mohammed has agreed to never disclose secret aspects of his torture by the CIA if he is allowed to plead guilty rather than face a death-penalty trial0532a14f-a9af-4665-b955-293fe6aa874f.

A clause regarding the non-disclosure of his torture was included in the latest portions of his deal to be unsealed at a federal appeals court in Washington0532a14f-a9af-4665-b955-293fe6aa874f. The agreement states he will not disclose information about his capture, detention, or confinement while in U.S. custody0532a14f-a9af-4665-b955-293fe6aa874f. He signed the agreement on July 310532a14f-a9af-4665-b955-293fe6aa874f.

Mohammed was waterboarded 183 times by the CIA 0532a14f-a9af-4665-b955-293fe6aa874f. The waterboarding was done by a three-person interrogation team led by Bruce Jessen and James E. Mitchell, two former contract psychologists for the agency0532a14f-a9af-4665-b955-293fe6aa874f. He was held in CIA "black site" prisons overseas from 2003-060532a14f-a9af-4665-b955-293fe6aa874f, and was shuttled between prisons in Afghanistan, Poland and other locations not acknowledged by the CIA as former black sites0532a14f-a9af-4665-b955-293fe6aa874f.

Mohammed was first charged in this case in 2012 0532a14f-a9af-4665-b955-293fe6aa874f. A sentencing hearing may be held at Guantánamo Bay0532a14f-a9af-4665-b955-293fe6aa874f.

Co-defendants: Walid bin Attash and Mustafa al-Hawsawi0532a14f-a9af-4665-b955-293fe6aa874f
Confidence: 0.9

Articles
Article ID: 0532a14f-a9af-4665-b955-293fe6aa874f, Title: 9/11 plea deal includes lifetime gag order on CIA torture secrets (link)"""

    # Example usage with different profile types
    print("Extracting headers for person profile...")
    headers = gemini_extract_headers(text, profile_type="person")
    for header in headers:
        print(f"- {header.h2_header_str}")

    print("\nRestructuring text with extracted headers...")
    restructured_text = restructure_text_with_headers(
        text, headers, model="openrouter/anthropic/claude-3-5-haiku"
    )
    print(restructured_text)
