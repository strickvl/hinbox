from enum import Enum

from pydantic import BaseModel


class PlaceType(str, Enum):
    COUNTRY = "country"
    PROVINCE = "province"
    STATE = "state"
    DISTRICT = "district"
    CITY = "city"
    PRISON_LOCATION = "prison_location"
    OTHER = "other"


class Place(BaseModel):
    name: str
    type: PlaceType
