# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import Sequence, Optional, cast, Type, Tuple

from .base import BasePerson
from ..interfaces import PersonRoutineWithStatus, NoOP, NOOP, LocationID, SpecialEndLoc, globals, PersonRoutine, \
    SimTimeTuple, SimTimeRoutineTrigger, RoutineTrigger

__all__ = ['execute_routines', 'triggered_routine', 'weekend_routine', 'mid_day_during_week_routine', 'social_routine']


def execute_routines(person: BasePerson, routines_with_status: Sequence[PersonRoutineWithStatus]) -> Optional[NoOP]:
    """
    Executes the given routines of the person in the simulator.
    Note that this function updates (mutates) status flags in the routines_with_status instances.

    :param person: person to execute the routine
    :param routines_with_status: a sequence of PersonRoutineWithStatus instances
    :return: returns a NOOP if none of the routines were executed (typically happens when the routines
        conditions are not met), otherwise None.
    """
    numpy_rng = globals.numpy_rng
    # the overall flow is that if a routine is due, start it and block the execution of other routines
    # until it has completed

    # first check if there are any ongoing routines that need to be completed
    for rws in routines_with_status:
        if rws.started and not rws.done:
            if (
                    # person left the end_location typically if the location closed due to external factors
                    (person.state.current_location != rws.end_loc_selected) or

                    # duration elapsed
                    (rws.duration >= rws.routine.duration_of_stay_at_end_loc)
            ):
                rws.done = True
            else:
                # block execution of other routines until the routine is complete
                return None

    # at this point there are no ongoing routines, now start the next routine that is due
    for rws in routines_with_status:
        routine = rws.routine
        if rws.due:
            if (
                    # check if we are in the correct location to start the routine
                    (routine.start_loc is None or routine.start_loc == person.state.current_location) and

                    # randomized start probability
                    numpy_rng.uniform() < routine.start_hour_probability
            ):
                # get the end location
                if routine.end_loc == SpecialEndLoc.social:
                    end_loc = person.get_social_gathering_location()
                    if end_loc is None:
                        # no social gatherings to attend, skip routine
                        continue
                else:
                    end_loc = cast(LocationID, routine.end_loc)

                if (len(routine.explorable_end_locs) > 0) and (numpy_rng.uniform() < routine.explore_probability):
                    end_loc = routine.explorable_end_locs[numpy_rng.randint(0, len(routine.explorable_end_locs))]

                assert end_loc
                if person.enter_location(end_loc):
                    # successfully started the routine
                    rws.due = False
                    rws.started = True
                    rws.duration = 1
                    rws.end_loc_selected = end_loc
                    return None
    return NOOP


def _get_locations_from_type(location_type: Type) -> Tuple[LocationID, Sequence[LocationID]]:
    assert globals.registry, 'No registry found. Create the repo wide registry first by calling init_globals()'
    explorable_end_locs = globals.registry.location_ids_of_type(location_type)
    assert len(explorable_end_locs) > 0, f'{location_type.__name__}'
    end_loc = explorable_end_locs[globals.numpy_rng.randint(0, len(explorable_end_locs))]
    return end_loc, explorable_end_locs


def triggered_routine(start_loc: Optional[LocationID],
                      end_location_type: type,
                      interval_in_days: int,
                      explore_probability: float = 0.05) -> PersonRoutine:
    end_loc, explorable_end_locs = _get_locations_from_type(end_location_type)
    return PersonRoutine(start_loc=start_loc,
                         end_loc=end_loc,
                         start_trigger=SimTimeRoutineTrigger(day=interval_in_days,
                                                             offset_day=globals.numpy_rng.randint(0, interval_in_days)),
                         explorable_end_locs=explorable_end_locs,
                         explore_probability=explore_probability)


def weekend_routine(start_loc: Optional[LocationID],
                    end_location_type: type,
                    explore_probability: float = 0.05,
                    reset_when_done: RoutineTrigger = SimTimeRoutineTrigger(day=1)) -> PersonRoutine:
    end_loc, explorable_end_locs = _get_locations_from_type(end_location_type)
    return PersonRoutine(start_loc=start_loc,
                         end_loc=end_loc,
                         valid_time=SimTimeTuple(week_days=(5, 6)),
                         explorable_end_locs=explorable_end_locs,
                         explore_probability=explore_probability,
                         reset_when_done_trigger=reset_when_done)


def mid_day_during_week_routine(start_loc: Optional[LocationID],
                                end_location_type: type,
                                explore_probability: float = 0.05) -> PersonRoutine:
    end_loc, explorable_end_locs = _get_locations_from_type(end_location_type)
    return PersonRoutine(start_loc=start_loc,
                         end_loc=end_loc,
                         valid_time=SimTimeTuple(hours=tuple(range(11, 14)), week_days=tuple(range(0, 5))),
                         explorable_end_locs=explorable_end_locs,
                         explore_probability=explore_probability)


def social_routine(start_loc: Optional[LocationID]) -> PersonRoutine:
    return PersonRoutine(start_loc=start_loc,
                         end_loc=SpecialEndLoc.social,
                         valid_time=SimTimeTuple(hours=tuple(range(15, 20))),
                         duration_of_stay_at_end_loc=globals.numpy_rng.randint(1, 3),
                         reset_when_done_trigger=SimTimeRoutineTrigger(day=7))
