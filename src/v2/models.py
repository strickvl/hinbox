from pydantic import BaseModel
from enum import Enum


class PlaceType(str, Enum):
    COUNTRY = "country"
    PROVINCE = "province"
    STATE = "state"
    DISTRICT = "district"
    CITY = "city"
    OTHER = "other"


class Place(BaseModel):
    name: str
    type: PlaceType
