# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from dataclasses import dataclass

from ..interfaces import BusinessLocationState, ContactRate, SimTimeTuple, NonEssentialBusinessLocationState, \
    EssentialBusinessBaseLocation, NonEssentialBusinessBaseLocation

__all__ = ['GroceryStore', 'RetailStore', 'GroceryStoreState', 'RetailStoreState']


@dataclass
class GroceryStoreState(BusinessLocationState):
    contact_rate: ContactRate = ContactRate(0, 1, 0, 0.2, 0.25, 0.3)
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(7, 21)), week_days=tuple(range(0, 6)))


class GroceryStore(EssentialBusinessBaseLocation[GroceryStoreState]):
    """Implements a grocery store location."""

    state_type = GroceryStoreState


@dataclass
class RetailStoreState(NonEssentialBusinessLocationState):
    contact_rate: ContactRate = ContactRate(0, 1, 0, 0.2, 0.25, 0.3)
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(7, 21)), week_days=tuple(range(0, 6)))


class RetailStore(NonEssentialBusinessBaseLocation[RetailStoreState]):
    """Implements a retail store location."""

    state_type = RetailStoreState
