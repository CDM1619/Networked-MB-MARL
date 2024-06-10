# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import dataclasses
from typing import Dict, List, Optional, cast, Set, Type, Mapping, Tuple, Union

from cachetools import cached

from .interfaces import LocationID, Location, PersonID, Person, Registry, RegistrationError, InfectionSummary, \
    IndividualInfectionState, BusinessLocationState, PandemicTestResult, LocationSummary, SimTimeTuple, SimTime, \
    LocationState
from pandemic_simulator.environment.interfaces.location_base_business import BusinessBaseLocation
from .location.cemetery import Cemetery

__all__ = ['CityRegistry']


class CityRegistry(Registry):
    """A global registry for all persons and location instances"""

    _location_register: Dict[LocationID, Location]
    _person_register: Dict[PersonID, Person]

    _location_ids: Set[LocationID]
    _business_location_ids: Set[LocationID]
    _person_ids: Set[PersonID]

    _quarantined: Set[PersonID]

    _location_ids_with_social_events: List[LocationID]
    _global_location_summary: Dict[Tuple[str, str], LocationSummary]
    _location_types: Set[str]
    _person_type_to_count: Dict[str, int]

    IGNORE_LOCS_SUMMARY: Set[Type] = {Cemetery}

    def __init__(self) -> None:
        self._location_register = {}
        self._person_register = {}

        self._location_ids = set()
        self._business_location_ids = set()
        self._person_ids = set()

        self._quarantined = set()
        self._global_location_summary = dict()
        self._location_types = set()
        self._person_type_to_count = dict()

    def register_location(self, location: Location) -> None:
        if location.id in self._location_register:
            raise RegistrationError(f'Location {location.id.name} is already registered.')
        self._location_register[location.id] = location
        self._location_ids.add(location.id)
        if isinstance(location.state, BusinessLocationState):
            self._business_location_ids.add(location.id)

        if type(location) not in self.IGNORE_LOCS_SUMMARY:
            self._location_types.add(type(location).__name__)

    def register_person(self, person: Person) -> None:
        if person.id in self._person_register:
            raise RegistrationError(f'Person {person.id.name} is already registered.')

        if person.state.current_location not in self._location_register:
            raise RegistrationError(f'Unable to register the person {person.id.name} because the current'
                                    f'location is not registered yet.')

        for loc_id in person.assigned_locations:
            if loc_id not in self._location_register:
                raise RegistrationError(f'Unable to register the person {person.id.name} because the assigned'
                                        f'location {loc_id.name} is not registered yet.')

        current_location = self._location_register[person.state.current_location]
        assigned_locations = [self._location_register[loc_id] for loc_id in person.assigned_locations]

        if (current_location.id not in person.assigned_locations and
                person.id not in current_location.state.persons_in_location):
            if not current_location.is_entry_allowed(person.id):
                raise RegistrationError('Unable to register the person because the person is not allowed in his/her'
                                        'claimed current location.')

        # everything checks out, register the person
        for loc in assigned_locations:
            loc.assign_person(person.id)
        current_location.add_person_to_location(person.id)
        self._person_register[person.id] = person
        self._person_ids.add(person.id)

        # init entry in global_location_summary
        person_type = type(person).__name__
        if person_type not in self._person_type_to_count:
            self._person_type_to_count[person_type] = 0
            for loc_type in self._location_types:
                self._global_location_summary[(loc_type, person_type)] = LocationSummary()
        self._person_type_to_count[person_type] += 1

    def register_person_entry_in_location(self, person_id: PersonID, location_id: LocationID) -> bool:
        person = self._person_register[person_id]
        current_location = self._location_register[person.state.current_location]
        next_location = self._location_register[location_id]

        if next_location.id != current_location.id and not next_location.is_entry_allowed(person_id):
            # if entry to the next location is not allowed
            return False

        # update person and location state
        current_location.remove_person_from_location(person_id)  # exit current
        next_location.add_person_to_location(person_id)  # enter next
        person.state.current_location = next_location.id  # update person state

        # update global location summary
        if type(next_location) not in self.IGNORE_LOCS_SUMMARY:
            location_type = type(next_location).__name__
            person_type = type(person).__name__
            key = (location_type, person_type)
            summary = self._global_location_summary[key]
            is_visitor = person_id not in next_location.state.assignees
            person_cnt = self._person_type_to_count[person_type]
            self._global_location_summary[key] = dataclasses.replace(
                summary,
                entry_count=(person_cnt * summary.entry_count + 1) / person_cnt,
                visitor_count=(person_cnt * summary.visitor_count + is_visitor) / person_cnt
            )
        return True

    def update_location_specific_information(self) -> None:
        self._location_ids_with_social_events = [loc_id for loc_id, loc in self._location_register.items()
                                                 if loc.state.social_gathering_event]

    def reassign_locations(self, person: Person) -> None:
        assigned_locations = [self._location_register[loc_id] for loc_id in person.assigned_locations]
        for loc in assigned_locations:
            loc.assign_person(person.id)

    # ----------------public attributes-----------------

    @property
    def person_ids(self) -> Set[PersonID]:
        return self._person_ids

    @property
    def location_ids(self) -> Set[LocationID]:
        return self._location_ids

    @property
    def location_ids_with_social_events(self) -> List[LocationID]:
        return self._location_ids_with_social_events

    @property
    def location_types(self) -> Set[str]:
        return self._location_types

    @property
    def global_location_summary(self) -> Mapping[Tuple[str, str], LocationSummary]:
        return self._global_location_summary

    # ----------------location utility methods-----------------

    @cached(cache={})
    def location_ids_of_type(self, location_type: Union[type, Tuple[type, ...]]) -> Tuple[LocationID, ...]:
        return tuple(loc_id for loc_id, loc in self._location_register.items() if isinstance(loc, location_type))

    def get_persons_in_location(self, location_id: LocationID) -> Set[PersonID]:
        return cast(LocationState, self._location_register[location_id].state).persons_in_location

    def location_id_to_type(self, location_id: LocationID) -> Type:
        return type(self._location_register[location_id])

    def get_location_work_time(self, location_id: LocationID) -> SimTimeTuple:
        assert location_id in self._business_location_ids, 'The given location id is not a business location.'
        return cast(BusinessBaseLocation, self._location_register[location_id]).get_worker_work_time()

    def is_location_open_for_visitors(self, location_id: LocationID, sim_time: SimTime) -> bool:
        location_state = self._location_register[location_id].state
        return location_state.is_open and sim_time in location_state.visitor_time

    # ----------------person utility methods-----------------

    def get_person_home_id(self, person_id: PersonID) -> LocationID:
        return self._person_register[person_id].home

    def get_households(self, person_id: PersonID) -> Set[PersonID]:
        home_id = self._person_register[person_id].home
        home = self._location_register[home_id]
        assignees = home.state.assignees

        return cast(Set, assignees)

    def get_person_infection_summary(self, person_id: PersonID) -> Optional[InfectionSummary]:
        if self._person_register[person_id].state.infection_state is not None:
            state = cast(IndividualInfectionState, self._person_register[person_id].state.infection_state)
            return state.summary

        return None

    def get_person_test_result(self, person_id: PersonID) -> PandemicTestResult:
        state = self._person_register[person_id].state
        return state.test_result

    def quarantine_person(self, person_id: PersonID) -> None:
        self._quarantined.add(person_id)

    def clear_quarantined(self, person_id: PersonID) -> None:
        if person_id in self._quarantined:
            self._quarantined.remove(person_id)

    def get_person_quarantined_state(self, person_id: PersonID) -> bool:
        return person_id in self._quarantined
