# Confidential, Copyright 2021, Sony Corporation of America, All rights reserved.
from typing import Optional

import pandemic_simulator as ps
from pandemic_simulator.environment import SimTime, PersonState


def test_person_routine_with_status_sim_time_trigger() -> None:
    ps.init_globals()

    home = ps.env.Home()
    store = ps.env.GroceryStore()

    r = ps.env.PersonRoutine(start_loc=home.id,
                             end_loc=store.id,
                             start_trigger=ps.env.SimTimeRoutineTrigger(day=1, offset_hour=10),
                             reset_when_done_trigger=ps.env.SimTimeRoutineTrigger(day=1, offset_hour=14))
    rws = ps.env.PersonRoutineWithStatus(r)

    # at start all flags are false
    rws.sync(ps.env.SimTime(hour=0))
    assert not rws.started
    assert not rws.due
    assert not rws.done

    # due should be true at trigger
    rws.sync(ps.env.SimTime(hour=10))
    assert rws.due

    # due should remain true even after trigger
    rws.sync(ps.env.SimTime(hour=11))
    assert rws.due

    # due should switch to false when routine has started
    rws.started = True
    rws.sync(ps.env.SimTime(hour=12))
    assert not rws.due

    # same when done is True
    rws.done = True
    rws.sync(ps.env.SimTime(hour=13))
    assert not rws.due

    # flags reset at reset trigger
    rws.sync(ps.env.SimTime(hour=14))
    assert not rws.started
    assert not rws.due
    assert not rws.done


class StateRoutineTrigger(ps.env.RoutineTrigger):

    def trigger(self, sim_time: SimTime, person_state: Optional[PersonState] = None) -> bool:
        assert person_state
        return person_state.risk == ps.env.Risk.HIGH


def test_person_routine_with_status_state_trigger() -> None:
    ps.init_globals()

    home = ps.env.Home()
    store = ps.env.GroceryStore()

    r = ps.env.PersonRoutine(start_loc=home.id,
                             end_loc=store.id,
                             start_trigger=StateRoutineTrigger(),
                             reset_when_done_trigger=ps.env.SimTimeRoutineTrigger(day=1))
    rws = ps.env.PersonRoutineWithStatus(r)

    # check flags for non-trigger state
    rws.sync(ps.env.SimTime(hour=5), person_state=PersonState(home.id, risk=ps.env.Risk.LOW))
    assert not rws.started
    assert not rws.due
    assert not rws.done

    # due should be true at state trigger
    rws.sync(ps.env.SimTime(hour=5), person_state=PersonState(home.id, risk=ps.env.Risk.HIGH))
    assert rws.due

    # due should remain true even after trigger
    rws.sync(ps.env.SimTime(hour=11), person_state=PersonState(home.id, risk=ps.env.Risk.LOW))
    assert rws.due

    # due should switch to false when routine has started
    rws.started = True
    rws.sync(ps.env.SimTime(hour=12), person_state=PersonState(home.id, risk=ps.env.Risk.HIGH))
    assert not rws.due

    # same when done is True
    rws.done = True
    rws.sync(ps.env.SimTime(hour=13), person_state=PersonState(home.id, risk=ps.env.Risk.HIGH))
    assert not rws.due

    # flags reset at reset trigger and due is again set (because of state trigger)
    rws.sync(ps.env.SimTime(day=1, hour=0), person_state=PersonState(home.id, risk=ps.env.Risk.HIGH))
    assert not rws.started
    assert rws.due
    assert not rws.done
