import json

import instructor
import litellm
from openai import OpenAI

from src.constants import (
    OLLAMA_API_KEY,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    get_ollama_model_name,
)
from src.models import ArticleTag, ArticleTags


def gemini_extract_tags(article_text: str) -> ArticleTags:
    """
    Extract article tags using Google's Gemini model.

    Args:
        article_text: The text of the article to analyze

    Returns:
        ArticleTags object containing the extracted tags
    """
    prompt = f"""
    You are an expert analyst specializing in articles about Guantánamo Bay detention camp.
    
    Analyze the following article and identify the main themes or topics it covers.
    Select the most relevant tags from the provided list that accurately represent the article's content.
    
    Article:
    ```
    {article_text}
    ```
    
    Available tags:
    {[tag.value for tag in ArticleTag]}
    
    Return your analysis as a JSON object with a single field "tags" containing an array of tag values.
    Only include tags that are strongly represented in the article.
    Include at least 2 tags but no more than 5 tags.
    
    Example response format:
    {{
      "tags": ["detainee_treatment", "legal_proceedings"]
    }}
    """

    try:
        response = litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert analyst of Guantánamo Bay detention camp articles.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            metadata={"project_name": "hinbox", "tags": ["dev", "article_tags"]},
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        # Validate that the tags are in the enum
        valid_tags = []
        for tag in data.get("tags", []):
            try:
                valid_tag = ArticleTag(tag)
                valid_tags.append(valid_tag)
            except ValueError:
                # Skip invalid tags
                continue

        return ArticleTags(tags=valid_tags)
    except Exception as e:
        print(f"Error extracting tags with Gemini: {e}")
        return ArticleTags(tags=[])


def ollama_extract_tags(article_text: str, model: str = OLLAMA_MODEL) -> ArticleTags:
    """
    Extract article tags using Ollama models.

    Args:
        article_text: The text of the article to analyze
        model: The Ollama model to use

    Returns:
        ArticleTags object containing the extracted tags
    """
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)
    instructor_client = instructor.from_openai(client)

    prompt = f"""
    You are an expert analyst specializing in articles about Guantánamo Bay detention camp.
    
    Analyze the following article and identify the main themes or topics it covers.
    Select the most relevant tags from the provided list that accurately represent the article's content.
    
    Article:
    ```
    {article_text}
    ```

    Available tags:
    {[tag.value for tag in ArticleTag]}

    Only include tags that are strongly represented in the article.
    Include at least 2 tags but no more than 5 tags.
    """

    try:
        results = client.beta.chat.completions.parse(
            model=get_ollama_model_name(model),  # Strip ollama/ prefix for API call
            response_format=ArticleTags,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert analyst of Guantánamo Bay detention camp articles.",
                },
                {"role": "user", "content": prompt},
            ],
            metadata={"project_name": "hinbox", "tags": ["dev", "article_tags"]},
        )
        return results.choices[0].message.parsed.tags
    except Exception as e:
        print(f"Error extracting tags with Ollama: {e}")
        return ArticleTags(tags=[])
