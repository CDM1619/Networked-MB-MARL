# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from collections import defaultdict, OrderedDict
from itertools import product as cartesianproduct, combinations
from typing import DefaultDict, Dict, List, Optional, Sequence, cast, Type

import numpy as np
from orderedset import OrderedSet

from .contact_tracing import MaxSlotContactTracer
from .infection_model import SEIRModel, SpreadProbabilityParams
from .interfaces import ContactRate, ContactTracer, PandemicRegulation, PandemicSimState, PandemicTesting, \
    PandemicTestResult, \
    DEFAULT, GlobalTestingState, InfectionModel, InfectionSummary, Location, LocationID, Person, PersonID, Registry, \
    SimTime, SimTimeInterval, sorted_infection_summary, globals, PersonRoutineAssignment
from .location import Hospital
from .make_population import make_population
from .pandemic_testing_strategies import RandomPandemicTesting
from .simulator_config import PandemicSimConfig
from .simulator_opts import PandemicSimOpts

__all__ = ['PandemicSim', 'make_locations']


def make_locations(sim_config: PandemicSimConfig) -> List[Location]:
    return [config.location_type(loc_id=f'{config.location_type.__name__}_{i}',
                                 init_state=config.location_type.state_type(**config.state_opts),
                                 **config.extra_opts)  # type: ignore
            for config in sim_config.location_configs for i in range(config.num)]


class PandemicSim:
    """Class that implements the pandemic simulator."""

    _id_to_person: Dict[PersonID, Person]
    _id_to_location: Dict[LocationID, Location]
    _infection_model: InfectionModel
    _pandemic_testing: PandemicTesting
    _registry: Registry
    _contact_tracer: Optional[ContactTracer]
    _new_time_slot_interval: SimTimeInterval
    _infection_update_interval: SimTimeInterval
    _infection_threshold: int
    _numpy_rng: np.random.RandomState

    _type_to_locations: DefaultDict
    _hospital_ids: List[LocationID]
    _persons: Sequence[Person]
    _state: PandemicSimState

    def __init__(self,
                 locations: Sequence[Location],
                 persons: Sequence[Person],
                 infection_model: Optional[InfectionModel] = None,
                 pandemic_testing: Optional[PandemicTesting] = None,
                 contact_tracer: Optional[ContactTracer] = None,
                 new_time_slot_interval: SimTimeInterval = SimTimeInterval(day=1),
                 infection_update_interval: SimTimeInterval = SimTimeInterval(day=1),
                 person_routine_assignment: Optional[PersonRoutineAssignment] = None,
                 infection_threshold: int = 0):
        """
        :param locations: A sequence of Location instances.
        :param persons: A sequence of Person instances.
        :param infection_model: Infection model instance, if None SEIR default infection model is used.
        :param pandemic_testing: PandemicTesting instance, if None RandomPandemicTesting default instance is used.
        :param contact_tracer: Optional ContactTracer instance.
        :param new_time_slot_interval: interval for updating contact tracer if that is not None. Default is set daily.
        :param infection_update_interval: interval for updating infection states. Default is set once daily.
        :param person_routine_assignment: An optional PersonRoutineAssignment instance that assign PersonRoutines to
            each person
        :param infection_threshold: If the infection summary is greater than the specified threshold, a
            boolean in PandemicSimState is set to True.
        """
        assert globals.registry, 'No registry found. Create the repo wide registry first by calling init_globals()'
        self._registry = globals.registry
        self._numpy_rng = globals.numpy_rng

        self._id_to_location = OrderedDict({loc.id: loc for loc in locations})
        assert self._registry.location_ids.issuperset(self._id_to_location)
        self._id_to_person = OrderedDict({p.id: p for p in persons})
        assert self._registry.person_ids.issuperset(self._id_to_person)

        self._infection_model = infection_model or SEIRModel()
        self._pandemic_testing = pandemic_testing or RandomPandemicTesting()
        self._contact_tracer = contact_tracer
        self._new_time_slot_interval = new_time_slot_interval
        self._infection_update_interval = infection_update_interval
        self._infection_threshold = infection_threshold

        self._type_to_locations = defaultdict(list)
        for loc in locations:
            self._type_to_locations[type(loc)].append(loc)
        self._hospital_ids = [loc.id for loc in locations if isinstance(loc, Hospital)]

        self._persons = persons

        # assign routines
        if person_routine_assignment is not None:
            for _loc in person_routine_assignment.required_location_types:
                assert _loc.__name__ in globals.registry.location_types, (
                    f'Required location type {_loc.__name__} not found. Modify sim_config to include it.')
            person_routine_assignment.assign_routines(persons)

        self._state = PandemicSimState(
            id_to_person_state={person.id: person.state for person in persons},
            id_to_location_state={location.id: location.state for location in locations},
            location_type_infection_summary={type(location): 0 for location in locations},
            global_infection_summary={s: 0 for s in sorted_infection_summary},
            global_testing_state=GlobalTestingState(summary={s: len(persons) if s == InfectionSummary.NONE else 0
                                                             for s in sorted_infection_summary},
                                                    num_tests=0),
            global_location_summary=self._registry.global_location_summary,
            sim_time=SimTime(),
            regulation_stage=0,
            infection_above_threshold=False
        )

    @classmethod
    def from_config(cls: Type['PandemicSim'],
                    sim_config: PandemicSimConfig,
                    sim_opts: PandemicSimOpts = PandemicSimOpts()) -> 'PandemicSim':
        """
        Creates an instance using config

        :param sim_config: Simulator config
        :param sim_opts: Simulator opts
        :return: PandemicSim instance
        """
        assert globals.registry, 'No registry found. Create the repo wide registry first by calling init_globals()'

        # make locations
        locations = make_locations(sim_config)

        # make population
        persons = make_population(sim_config)

        # make infection model
        infection_model = SEIRModel(
            spread_probability_params=SpreadProbabilityParams(sim_opts.infection_spread_rate_mean,
                                                              sim_opts.infection_spread_rate_sigma))

        # setup pandemic testing
        pandemic_testing = RandomPandemicTesting(spontaneous_testing_rate=sim_opts.spontaneous_testing_rate,
                                                 symp_testing_rate=sim_opts.symp_testing_rate,
                                                 critical_testing_rate=sim_opts.critical_testing_rate,
                                                 testing_false_positive_rate=sim_opts.testing_false_positive_rate,
                                                 testing_false_negative_rate=sim_opts.testing_false_negative_rate,
                                                 retest_rate=sim_opts.retest_rate)

        # create contact tracing app (optional)
        contact_tracer = MaxSlotContactTracer(
            storage_slots=sim_opts.contact_tracer_history_size) if sim_opts.use_contact_tracer else None

        # setup sim
        return PandemicSim(persons=persons,
                           locations=locations,
                           infection_model=infection_model,
                           pandemic_testing=pandemic_testing,
                           contact_tracer=contact_tracer,
                           infection_threshold=sim_opts.infection_threshold,
                           person_routine_assignment=sim_config.person_routine_assignment)

    @property
    def registry(self) -> Registry:
        """Return registry"""
        return self._registry

    def _compute_contacts(self, location: Location) -> OrderedSet:
        assignees = location.state.assignees_in_location
        visitors = location.state.visitors_in_location
        cr = location.state.contact_rate

        groups = [(assignees, assignees),
                  (assignees, visitors),
                  (visitors, visitors)]
        constraints = [(cr.min_assignees, cr.fraction_assignees),
                       (cr.min_assignees_visitors, cr.fraction_assignees_visitors),
                       (cr.min_visitors, cr.fraction_visitors)]

        contacts: OrderedSet = OrderedSet()

        for grp, cst in zip(groups, constraints):
            grp1, grp2 = grp
            minimum, fraction = cst

            possible_contacts = list(combinations(grp1, 2) if grp1 == grp2 else cartesianproduct(grp1, grp2))
            num_possible_contacts = len(possible_contacts)

            if len(possible_contacts) == 0:
                continue

            fraction_sample = min(1., max(0., self._numpy_rng.normal(fraction, 1e-2)))
            real_fraction = max(minimum, int(fraction_sample * num_possible_contacts))

            # we are using an orderedset, it's repeatable
            contact_idx = self._numpy_rng.randint(0, num_possible_contacts, real_fraction)
            contacts.update([possible_contacts[idx] for idx in contact_idx])

        return contacts

    def _compute_infection_probabilities(self, contacts: OrderedSet) -> None:
        infectious_states = {InfectionSummary.INFECTED, InfectionSummary.CRITICAL}

        for c in contacts:
            id_person1 = c[0]
            id_person2 = c[1]
            person1_state = self._id_to_person[id_person1].state
            person2_state = self._id_to_person[id_person2].state
            person1_inf_state = person1_state.infection_state
            person2_inf_state = person2_state.infection_state

            if (
                    # both are not infectious
                    (person1_inf_state is None and person2_inf_state is None) or

                    # both are already infected
                    (person1_inf_state is not None and person2_inf_state is not None and

                     person1_inf_state.summary in infectious_states and person2_inf_state.summary in infectious_states)
            ):
                continue
            elif person1_inf_state is not None and person1_inf_state.summary in infectious_states:
                spread_probability = (person1_inf_state.spread_probability *
                                      person1_state.infection_spread_multiplier)
                person2_state.not_infection_probability *= 1 - spread_probability
                person2_state.not_infection_probability_history.append((person2_state.current_location,
                                                                        person2_state.not_infection_probability))
            elif person2_inf_state is not None and person2_inf_state.summary in infectious_states:
                spread_probability = (person2_inf_state.spread_probability *
                                      person2_state.infection_spread_multiplier)
                person1_state.not_infection_probability *= 1 - spread_probability
                person1_state.not_infection_probability_history.append((person1_state.current_location,
                                                                        person1_state.not_infection_probability))

    def _update_global_testing_state(self, new_result: PandemicTestResult, prev_result: PandemicTestResult) -> None:
        if new_result == prev_result:
            # nothing to update
            return

            # person died - just update the test summary and __not__ the num_tests
        if new_result == PandemicTestResult.DEAD and prev_result != PandemicTestResult.DEAD:
            prv = InfectionSummary.CRITICAL if prev_result == PandemicTestResult.CRITICAL else \
                InfectionSummary.INFECTED if prev_result == PandemicTestResult.POSITIVE else InfectionSummary.NONE
            self._state.global_testing_state.summary[InfectionSummary.DEAD] += 1
            self._state.global_testing_state.summary[prv] -= 1

        # person tested positive/critical
        elif (new_result in {PandemicTestResult.POSITIVE, PandemicTestResult.CRITICAL} and
              prev_result in {PandemicTestResult.POSITIVE, PandemicTestResult.NEGATIVE, PandemicTestResult.UNTESTED}):
            new = InfectionSummary.CRITICAL if new_result == PandemicTestResult.CRITICAL else InfectionSummary.INFECTED
            prv = InfectionSummary.INFECTED if prev_result == PandemicTestResult.POSITIVE else InfectionSummary.NONE
            self._state.global_testing_state.summary[new] += 1
            self._state.global_testing_state.summary[prv] -= 1
            self._state.global_testing_state.num_tests += 1  # update number of tests

        # person tested negative after having tested as infected before
        elif (new_result == PandemicTestResult.NEGATIVE and
              prev_result in {PandemicTestResult.POSITIVE, PandemicTestResult.CRITICAL}):
            prv = InfectionSummary.CRITICAL if prev_result == PandemicTestResult.CRITICAL else InfectionSummary.INFECTED
            self._state.global_testing_state.summary[InfectionSummary.RECOVERED] += 1
            self._state.global_testing_state.summary[prv] -= 1
            self._state.global_testing_state.num_tests += 1  # update number of tests

    def step(self) -> None:
        """Method that advances one step through the simulator"""
        # sync all locations
        for location in self._id_to_location.values():
            location.sync(self._state.sim_time)
        self._registry.update_location_specific_information()

        # call person steps (randomize order)
        for i in self._numpy_rng.randint(0, len(self._persons), len(self._persons)):
            self._persons[i].step(self._state.sim_time, self._contact_tracer)

        # update person contacts
        for location in self._id_to_location.values():
            contacts = self._compute_contacts(location)

            if self._contact_tracer:
                self._contact_tracer.add_contacts(contacts)

            self._compute_infection_probabilities(contacts)

        # call infection model steps
        if self._infection_update_interval.trigger_at_interval(self._state.sim_time):
            global_infection_summary = {s: 0 for s in sorted_infection_summary}
            for person in self._id_to_person.values():
                # infection model step
                person.state.infection_state = self._infection_model.step(person.state.infection_state,
                                                                          person.id.age,
                                                                          person.state.risk,
                                                                          1 - person.state.not_infection_probability)

                if person.state.infection_state.exposed_rnb != -1.:
                    for vals in person.state.not_infection_probability_history:
                        if person.state.infection_state.exposed_rnb < 1 - vals[1]:
                            infection_location = vals[0]
                            break

                    person_location_type = self._registry.location_id_to_type(infection_location)
                    self._state.location_type_infection_summary[person_location_type] += 1

                global_infection_summary[person.state.infection_state.summary] += 1
                person.state.not_infection_probability = 1.
                person.state.not_infection_probability_history = []

                # test the person for infection
                if self._pandemic_testing.admit_person(person.state):
                    new_test_result = self._pandemic_testing.test_person(person.state)
                    self._update_global_testing_state(new_test_result, person.state.test_result)
                    person.state.test_result = new_test_result

            self._state.global_infection_summary = global_infection_summary
        self._state.infection_above_threshold = (self._state.global_testing_state.summary[InfectionSummary.INFECTED]
                                                 >= self._infection_threshold)

        self._state.global_location_summary = self._registry.global_location_summary

        if self._contact_tracer and self._new_time_slot_interval.trigger_at_interval(self._state.sim_time):
            self._contact_tracer.new_time_slot()

        # call sim time step
        self._state.sim_time.step()

    def step_day(self, hours_in_a_day: int = 24) -> None:
        for _ in range(hours_in_a_day):
            self.step()

    @staticmethod
    def _get_cr_from_social_distancing(location: Location,
                                       social_distancing: float) -> ContactRate:
        new_fraction = 1 - social_distancing
        cr = location.state.contact_rate
        init_cr = location.init_state.contact_rate
        new_cr = ContactRate(cr.min_assignees,
                             cr.min_assignees_visitors,
                             cr.min_visitors,
                             new_fraction * init_cr.fraction_assignees,
                             new_fraction * init_cr.fraction_assignees_visitors,
                             new_fraction * init_cr.fraction_visitors)

        return new_cr

    def impose_regulation(self, regulation: PandemicRegulation) -> None:
        """
        Receive a regulation that updates the simulator dynamics

        :param regulation: a PandemicRegulation instance
        """
        # update location rules
        sd = [regulation[i].social_distancing for i in range(len(regulation)-1)]
        loc_type_rk = [regulation[i].location_type_to_rule_kwargs for i in range(len(regulation)-1)]
        
        #print("regulation=",regulation)
        #print("_type_to_locations=",self._type_to_locations)
        
        #print("loc_type_rk=",loc_type_rk)
        #print("sd=",sd)

        #print("location_length=",len(self._type_to_locations.items()))
        p = 0
        for loc_type, locations in self._type_to_locations.items():
            #print("loc_type=",loc_type)
            #print("loaction",locations)
            rule_kwargs = {}
            if loc_type_rk[p] is not None and loc_type in loc_type_rk[p]:
                rule_kwargs.update(loc_type_rk[p][loc_type])

            if sd[p] is not None:
                # cr is the same for all locations of a given type. So just use one location to compute the new cr.
                cr = (DEFAULT if sd[p] == DEFAULT
                      else self._get_cr_from_social_distancing(locations[0], cast(float, sd[p])))
                rule_kwargs.update(dict(contact_rate=cr))
            #print("locations_size=",len(locations))
            for loc in locations:
                loc.update_rules(loc.location_rule_type(**rule_kwargs))
            
            p += 1

        # update person policy

        #print("person_size=",len(self._id_to_person.values()))
        for person in self._id_to_person.values():
            person.receive_regulation(regulation[-1])

        self._state.regulation_stage = regulation[-1].stage

    @property
    def state(self) -> PandemicSimState:
        """
        Property that returns the current state of the simulator.

        :return: Current state of the simulator.
        """

        return self._state

    def reset(self) -> None:
        for location in self._id_to_location.values():
            location.reset()
        for person in self._id_to_person.values():
            person.reset()

        self._infection_model.reset()

        num_persons = len(self._id_to_person)
        self._state = PandemicSimState(
            id_to_person_state={person_id: person.state for person_id, person in self._id_to_person.items()},
            id_to_location_state={loc_id: loc.state for loc_id, loc in self._id_to_location.items()},
            location_type_infection_summary={type(location): 0 for location in self._id_to_location.values()},

            global_infection_summary={s: 0 for s in sorted_infection_summary},
            global_location_summary=self._registry.global_location_summary,
            global_testing_state=GlobalTestingState(summary={s: num_persons if s == InfectionSummary.NONE else 0
                                                             for s in sorted_infection_summary},
                                                    num_tests=0),
            sim_time=SimTime(),
            regulation_stage=0,
            infection_above_threshold=False,
        )
