# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from dataclasses import dataclass

from ..interfaces import NonEssentialBusinessLocationState, ContactRate, SimTimeTuple, \
    AgeRestrictedBusinessBaseLocation, NonEssentialBusinessBaseLocation

__all__ = ['Bar', 'Restaurant', 'RestaurantState', 'BarState']


@dataclass
class RestaurantState(NonEssentialBusinessLocationState):
    contact_rate: ContactRate = ContactRate(1, 1, 0, 0.3, 0.35, 0.1)
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(11, 16)) + tuple(range(19, 24)),
                                           week_days=tuple(range(1, 7)))


class Restaurant(NonEssentialBusinessBaseLocation[RestaurantState]):
    """Implements a restaurant location."""
    state_type = RestaurantState


@dataclass
class BarState(NonEssentialBusinessLocationState):
    contact_rate: ContactRate = ContactRate(1, 1, 0, 0.7, 0.2, 0.1)
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(21, 24)), week_days=tuple(range(1, 7)))


class Bar(AgeRestrictedBusinessBaseLocation[BarState]):
    """Implements a Bar"""
    state_type = BarState
    age_limits = (21, 110)
