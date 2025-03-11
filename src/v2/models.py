from pydantic import BaseModel
from enum import Enum


class PlaceType(str, Enum):
    PROVINCE = "province"
    DISTRICT = "district"
    CITY = "city"
    OTHER = "other"


class Place(BaseModel):
    name: str
    type: PlaceType
