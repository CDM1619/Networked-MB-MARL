# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, Generic, TypeVar, ClassVar

from .ids import PersonID, LocationID
from .location_rules import LocationRule
from .location_states import LocationState
from .sim_time import SimTime
from ...utils import abstract_class_property

__all__ = ['Location', 'LocationError', 'LocationSummary']


class LocationError(Exception):
    """Generic location error"""
    pass


@dataclass(frozen=True)
class LocationSummary:
    """Dataclass that holds the location summary stats."""

    entry_count: float = 0
    """Entries to the location by a person (assignees and visitors)"""

    visitor_count: float = 0
    """Entries to the location by a visitor"""


_State = TypeVar('_State', bound=LocationState)


class Location(ABC, Generic[_State]):
    """Class that implements a location with a pre-defined operating rules"""

    location_rule_type: Type = abstract_class_property()  # The type of the location rule used by the location
    state_type: ClassVar[Type[_State]]

    @property
    @abstractmethod
    def id(self) -> LocationID:
        """
        Method that returns the id of the location.

        :return: ID of the location.
        """
        pass

    @property
    @abstractmethod
    def state(self) -> _State:
        """
        Property that returns the current state of the location.

        :return: Current state of the location.
        """
        pass

    @property
    @abstractmethod
    def init_state(self) -> _State:
        """
        Property that returns the init state of the location.

        :return: Init state of the location.
        """
        pass

    @abstractmethod
    def sync(self, sim_time: SimTime) -> None:
        """
        Sync location time with simulation time.

        :param sim_time: Current simulation time.
        """
        pass

    @abstractmethod
    def update_rules(self, new_rule: LocationRule) -> None:
        """Update operating rules based on the given location instruction."""
        pass

    @abstractmethod
    def is_entry_allowed(self, person_id: PersonID) -> bool:
        """
        Checks if a person with the given ID is allowed to enter the location at the current time.

        :param person_id: PersonID instance
        :return: Return True if entry is allowed else False
        """

    @abstractmethod
    def assign_person(self, person_id: PersonID) -> None:
        """
        Method that assigns a person to the location.

        :param person_id: PersonID instance
        """
        pass

    @abstractmethod
    def add_person_to_location(self, person_id: PersonID) -> None:
        """Adds a person with the given ID to the location"""
        pass

    @abstractmethod
    def remove_person_from_location(self, person_id: PersonID) -> None:
        """Removes a person with the given ID from the location"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset location to its initial state."""
