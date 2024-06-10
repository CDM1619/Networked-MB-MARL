import numpy as np

import pandemic_simulator as ps


def test_family_households() -> None:
    ps.init_globals()
    config = ps.sh.small_town_config
    cr = ps.env.globals.registry
    assert cr

    ps.env.make_locations(config)
    ps.env.make_population(config)

    retiree_homes_list = []
    minor_homes_list = []
    adult_homes_list = []

    homes = cr.location_ids_of_type(ps.env.Home)

    tot_persons = 0
    for home in homes:
        household = cr.get_persons_in_location(home)

        adults = 0
        minors = 0
        retirees = 0
        for member in household:
            if member.age <= 18:
                minors += 1
            elif 18 < member.age <= 65:
                adults += 1
            else:
                retirees += 1
        if minors > 0:
            minor_homes_list.append([minors, adults, retirees])
        elif adults > 0:
            adult_homes_list.append([minors, adults, retirees])
        elif retirees > 0:
            retiree_homes_list.append([minors, adults, retirees])
        tot_persons += len(household)

    assert len(minor_homes_list)
    assert len(adult_homes_list)
    assert len(retiree_homes_list)
    assert tot_persons == config.num_persons

    minor_homes = np.asarray(minor_homes_list)
    adult_homes = np.asarray(adult_homes_list)
    retiree_homes = np.asarray(retiree_homes_list)

    # there should be non-zero homes with 1, 2, and 3 children
    for i in range(1, 4):
        assert len(minor_homes[minor_homes[:, 0] == i]) > 0

    # each minor home must contain an adult
    assert all(minor_homes[:, 1] > 0)

    # minor homes in general must also have retirees for small town config
    assert np.sum(minor_homes, axis=0)[2] > 0

    # there should be non-zeros homes with 1 and >1 adults
    assert len(adult_homes[adult_homes[:, 1] == 1]) > 0
    assert len(adult_homes[adult_homes[:, 1] > 1]) > 0

    # no minors in adult homes
    assert np.sum(adult_homes, axis=0)[0] == 0

    # adult homes in general must also have retirees for small town config
    assert np.sum(adult_homes, axis=0)[2] > 0

    # there should be non-zeros with only retirees and no adults and minors
    assert np.sum(retiree_homes, axis=0)[2] > 0
    assert np.sum(retiree_homes, axis=0)[1] == 0
    assert np.sum(retiree_homes, axis=0)[0] == 0


def test_retiree_households() -> None:
    ps.init_globals()
    config = ps.sh.small_town_config

    ps.env.make_locations(config)
    ps.env.make_population(config)

    cr = ps.env.globals.registry
    assert cr

    home_ids = cr.location_ids_of_type(ps.env.Home)
    retirees_in_nursing_home = 0
    all_retirees = 0
    for home in home_ids:
        household = cr.get_persons_in_location(home)
        is_nursing_home = True
        for member in household:
            if member.age <= 65:
                is_nursing_home = False
            else:
                all_retirees += 1
        if is_nursing_home:
            retirees_in_nursing_home += len(household)

    assert (retirees_in_nursing_home / all_retirees) >= 0.065


def test_single_parent_households() -> None:
    ps.init_globals()
    config = ps.sh.small_town_config

    ps.env.make_locations(config)
    ps.env.make_population(config)

    cr = ps.env.globals.registry
    assert cr

    minor_homes_list = []

    homes = cr.location_ids_of_type(ps.env.Home)

    for home in homes:
        household = cr.get_persons_in_location(home)
        minors = 0
        adults = 0
        retirees = 0
        for member in household:
            if member.age <= 18:
                minors += 1
            elif 18 < member.age <= 65:
                adults += 1
            else:
                retirees += 1
        if minors > 0:
            minor_homes_list.append([minors, adults, retirees])

    minor_homes = np.asarray(minor_homes_list)
    nm = minor_homes[:, 1] + minor_homes[:, 2]
    single_parent_homes = sum(nm == 1)
    assert (single_parent_homes / len(minor_homes)) > 0.22
