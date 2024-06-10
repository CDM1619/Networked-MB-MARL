# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from dataclasses import dataclass

from ..interfaces import NonEssentialBusinessLocationState, ContactRate, SimTimeTuple, NonEssentialBusinessBaseLocation

__all__ = ['HairSalon', 'HairSalonState']


@dataclass
class HairSalonState(NonEssentialBusinessLocationState):
    contact_rate: ContactRate = ContactRate(1, 1, 0, 0.5, 0.3, 0.1)
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(9, 17)), week_days=tuple(range(1, 7)))


class HairSalon(NonEssentialBusinessBaseLocation[HairSalonState]):
    """Implements a hair salon."""
    state_type = HairSalonState
