# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from dataclasses import dataclass

from ..interfaces import NonEssentialBusinessLocationState, ContactRate, SimTimeTuple, NonEssentialBusinessBaseLocation

__all__ = ['School', 'SchoolState']


@dataclass
class SchoolState(NonEssentialBusinessLocationState):
    contact_rate: ContactRate = ContactRate(5, 1, 0, 0.1, 0., 0.1)
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(7, 15)), week_days=tuple(range(0, 5)))


class School(NonEssentialBusinessBaseLocation[SchoolState]):
    """Implements a simple school"""

    state_type = SchoolState
