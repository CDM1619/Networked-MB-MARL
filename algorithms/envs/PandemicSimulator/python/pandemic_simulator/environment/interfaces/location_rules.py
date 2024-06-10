# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import dataclasses
from typing import Union, Optional, Type

from .location_states import ContactRate
from .pandemic_types import Default, DEFAULT
from .sim_time import SimTimeTuple

__all__ = ['LocationRule', 'BusinessLocationRule', 'NonEssentialBusinessLocationRule']


@dataclasses.dataclass(frozen=True)
class LocationRule:
    """A rule to modify the location's operation."""
    contact_rate: Union[ContactRate, Default, None] = None
    visitor_time: Union[SimTimeTuple, Default, None] = None
    visitor_capacity: Union[Default, int, None] = None

    @classmethod
    def get_default(cls: Type['LocationRule']) -> 'LocationRule':
        return LocationRule(**{f.name: DEFAULT for f in dataclasses.fields(cls)})


@dataclasses.dataclass(frozen=True)
class BusinessLocationRule(LocationRule):
    open_time: Union[SimTimeTuple, Default, None] = None


@dataclasses.dataclass(frozen=True)
class NonEssentialBusinessLocationRule(BusinessLocationRule):
    lock: Optional[bool] = None
