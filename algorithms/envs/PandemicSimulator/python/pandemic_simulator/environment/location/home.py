# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from dataclasses import dataclass

from ..interfaces import LocationState, ContactRate, SimTime, SimTimeTuple, LocationRule, globals, BaseLocation

__all__ = ['Home', 'HomeState']


@dataclass
class HomeState(LocationState):
    contact_rate: ContactRate = ContactRate(0, 1, 0, 0.5, 0.3, 0.3)
    visitor_time = SimTimeTuple(hours=tuple(range(15, 20)), days=tuple(globals.numpy_rng.randint(0, 365, 12)))


class Home(BaseLocation[HomeState]):
    """Class that implements a standard Home location. """
    state_type = HomeState

    def sync(self, sim_time: SimTime) -> None:
        super().sync(sim_time)
        self._state.social_gathering_event = sim_time in self._state.visitor_time

    def update_rules(self, new_rule: LocationRule) -> None:
        pass
