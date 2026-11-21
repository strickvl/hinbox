"""Deterministic locators for immutable textual representations."""

from pydantic import BaseModel, Field, model_validator


class TextLocator(BaseModel):
    """A half-open span measured in Python/Unicode code points."""

    representation_id: str
    coordinate_system: str = "unicode_codepoint"
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TextLocator":
        if self.coordinate_system != "unicode_codepoint":
            raise ValueError("unsupported text locator coordinate system")
        if self.end <= self.start:
            raise ValueError("locator end must be greater than start")
        return self


class RetrievedPassage(BaseModel):
    locator: TextLocator
    quotation: str
    textual_form: str
    checksum: str
