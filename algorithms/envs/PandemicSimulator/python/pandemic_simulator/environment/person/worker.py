# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import Optional, Sequence, List

from .base import BasePerson
from .routine_utils import execute_routines
from ..interfaces import PersonState, LocationID, SimTime, NoOP, SimTimeTuple, NOOP, PersonRoutine, \
    ContactTracer, PersonID, PersonRoutineWithStatus

__all__ = ['Worker']


class Worker(BasePerson):
    """Class that implements a basic worker."""

    _work: LocationID
    _work_time: SimTimeTuple

    _routines: List[PersonRoutine]
    _during_work_rs: List[PersonRoutineWithStatus]
    _outside_work_rs: List[PersonRoutineWithStatus]

    def __init__(self,
                 person_id: PersonID,
                 home: LocationID,
                 work: LocationID,
                 work_time: Optional[SimTimeTuple] = None,
                 regulation_compliance_prob: float = 1.0,
                 init_state: Optional[PersonState] = None):
        """
        :param person_id: PersonID instance
        :param home: Home location id
        :param work: Work location id
        :param work_time: Work time specified in SimTimeTuples. Default - 9am-5pm and Mon-Fri
        :param regulation_compliance_prob: probability of complying to a regulation
        :param init_state: Optional initial state of the person
        """
        assert person_id.age >= 18, "Workers's age must be >= 18"
        self._work = work
        self._work_time = work_time or SimTimeTuple(hours=tuple(range(9, 18)), week_days=tuple(range(0, 5)))

        self._routines = []
        self._during_work_rs = []
        self._outside_work_rs = []

        super().__init__(person_id=person_id,
                         home=home,
                         regulation_compliance_prob=regulation_compliance_prob,
                         init_state=init_state)

    @property
    def work(self) -> LocationID:
        return self._work

    @property
    def assigned_locations(self) -> Sequence[LocationID]:
        return self._home, self._work

    @property
    def at_work(self) -> bool:
        """Return True if the person is at work and False otherwise"""
        return self._state.current_location == self.work

    def set_during_work_routines(self, routines: Sequence[PersonRoutine]) -> None:
        """A sequence of person routines to run during work time"""
        for routine in routines:
            if routine not in self._routines:
                self._routines.append(routine)
                self._during_work_rs.append(PersonRoutineWithStatus(routine))

    def set_outside_work_routines(self, routines: Sequence[PersonRoutine]) -> None:
        """A sequence of person routines to run outside work time"""
        for routine in routines:
            if routine not in self._routines:
                self._routines.append(routine)
                self._outside_work_rs.append(PersonRoutineWithStatus(routine))

    def _sync(self, sim_time: SimTime) -> None:
        super()._sync(sim_time)

        for rws in self._during_work_rs + self._outside_work_rs:
            rws.sync(sim_time=sim_time, person_state=self.state)

    def step(self, sim_time: SimTime, contact_tracer: Optional[ContactTracer] = None) -> Optional[NoOP]:
        step_ret = super().step(sim_time, contact_tracer)
        if step_ret != NOOP:
            return step_ret

        if sim_time in self._work_time:
            # execute during work routines
            ret = execute_routines(person=self, routines_with_status=self._during_work_rs)
            if ret != NOOP:
                return ret

            # no more routines to execute, go back to office
            if not self.at_work and self.enter_location(self.work):
                return None
        else:
            # execute outside work time routines
            ret = execute_routines(person=self, routines_with_status=self._outside_work_rs)
            if ret != NOOP:
                return ret

            # no more routines to execute, go home
            if not self.at_home:
                self.enter_location(self.home)
                return None

        return NOOP

    def reset(self) -> None:
        super().reset()
        for rws in self._during_work_rs + self._outside_work_rs:
            rws.reset()
