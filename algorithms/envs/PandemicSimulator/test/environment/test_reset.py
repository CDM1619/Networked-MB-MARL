# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import copy

import pandemic_simulator as ps


def test_location_and_person_reset() -> None:
    ps.init_globals()
    config = ps.sh.tiny_town_config

    locations = ps.env.make_locations(config)
    persons = ps.env.make_population(config)

    loc_states = [copy.deepcopy(loc.state) for loc in locations]
    per_states = [copy.deepcopy(per.state) for per in persons]

    for loc in locations:
        loc.reset()

    for per in persons:
        per.reset()

    new_loc_states = [copy.deepcopy(loc.state) for loc in locations]
    new_per_states = [copy.deepcopy(per.state) for per in persons]

    for st1, st2 in zip(loc_states, new_loc_states):
        assert st1 == st2

    for st1, st2 in zip(per_states, new_per_states):
        assert st1 == st2
