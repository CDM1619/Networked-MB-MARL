# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.


from dataclasses import dataclass, field
from typing import Set

from orderedset import OrderedSet

from .ids import PersonID
from .sim_time import SimTimeTuple

__all__ = ['LocationState', 'ContactRate', 'NonEssentialBusinessLocationState',
           'BusinessLocationState']


@dataclass(frozen=True)
class ContactRate:
    """Defines contact rates in a location."""

    min_assignees: int
    """Minimum number of contacts between assignees in the location"""

    min_assignees_visitors: int
    """Minimum number of contacts between assignees and visitors in the location"""

    min_visitors: int
    """Minimum number of contacts between visitors in the location"""

    fraction_assignees: float
    """A fraction of contacts between all assignees currently in the location. A value in [0, 1]"""

    fraction_assignees_visitors: float
    """A fraction of contacts between assignees and visitors currently in the location. A value in [0, 1]"""

    fraction_visitors: float
    """A fraction of contacts between all visitors currently in the location. A value in [0, 1]"""

    def __post_init__(self) -> None:
        assert 0 <= self.fraction_assignees <= 1
        assert 0 <= self.fraction_assignees_visitors <= 1
        assert 0 <= self.fraction_visitors <= 1


@dataclass
class LocationState:
    """State of the location."""

    contact_rate: ContactRate = ContactRate(1, 1, 0, 0.5, 0., 0.)
    """Rate at which assignees interact with other persons at that location."""

    visitor_capacity: int = -1
    """Number of visitors allowed during the visitor_time"""

    visitor_time: SimTimeTuple = SimTimeTuple()
    """Time when visitors are allowed to enter."""

    # state values that are updated by the simulator

    is_open: bool = field(init=False, default=True)
    """A boolean that indicates if the location is open or closed."""

    assignees: OrderedSet = field(default_factory=OrderedSet, init=False)
    """A set of ids of persons assigned to the location. Default is an empty set where nobody is assigned."""

    assignees_in_location: OrderedSet = field(default_factory=OrderedSet, init=False)
    """A set of ids of assignes who are currently in the location. Default is an empty set."""

    visitors_in_location: OrderedSet = field(default_factory=OrderedSet, init=False)
    """A set of ids of visitors who are currently in the location. Default is an empty set."""

    social_gathering_event: bool = field(default=False, init=False)
    """Set to True to advertise a social gathering at the location."""

    @property
    def persons_in_location(self) -> Set[PersonID]:
        """
        Property that returns the set of person ids in the location.

        :return: ID of the persons in the location.
        """
        persons = set(self.assignees_in_location)
        persons.union(self.visitors_in_location)
        return persons

    @property
    def num_persons_in_location(self) -> int:
        """Returns the number of persons in the location."""
        return len(self.assignees_in_location) + len(self.visitors_in_location)


@dataclass
class BusinessLocationState(LocationState):
    open_time: SimTimeTuple = SimTimeTuple(hours=tuple(range(9, 18)), week_days=tuple(range(0, 5)))
    """Specifies the time during which the location is open, during which the state variable is_open is True."""


@dataclass
class NonEssentialBusinessLocationState(BusinessLocationState):
    locked: bool = False
    """Specifies if the location is locked. When locked, is_open will be False independent of the open_time."""
