# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List, cast
from uuid import uuid4

import numpy as np

from .interfaces import globals, Risk, Person, PersonID, PersonState, BusinessBaseLocation
from .job_counselor import JobCounselor
from .location import Home, School
from .person import Retired, Worker, Minor
from .simulator_config import PandemicSimConfig
from ..utils import cluster_into_random_sized_groups

__all__ = ['make_population']

age_group = range(2, 101)


def get_us_age_distribution(num_persons: int) -> List[int]:
    age_p = np.zeros(100)
    for i, age in enumerate(age_group):
        if age < 60:
            age_p[i] = globals.numpy_rng.normal(1, 0.05)
        else:
            age_p[i] = (1 + (age - 60) * (0.05 - 1) / (100 - 60)) * globals.numpy_rng.normal(1, 0.05)
    age_p /= np.sum(age_p)
    ages = [int(globals.numpy_rng.choice(np.arange(1, 101), p=age_p)) for _ in range(num_persons)]
    # print(f'Average age: {np.average(ages)}')
    return ages


def infection_risk(age: int) -> Risk:
    return cast(Risk,
                globals.numpy_rng.choice([Risk.LOW, Risk.HIGH], p=[1 - age / age_group.stop, age / age_group.stop]))


def make_population(sim_config: PandemicSimConfig) -> List[Person]:
    """
    Creates a realistic us-age distributed population with home assignment and returns a list of persons.

    The overview of home assignment is as follows:

    a) "Only 4.5 percent of older adults live in nursing homes and
    2 percent in assisted living facilities. The majority of older adults (93.5 percent) live in the community."
    - https://www.ncbi.nlm.nih.gov/books/NBK51841/

    Select 6.5% of older adults (age > 65) and cluster them as groups of 1 or 2 and assign each
    group to a nursing home.

    b) "In 2019, there was an average of 1.93 children under 18 per family in the United States"
    - https://www.statista.com/statistics/718084/average-number-of-own-children-per-family/

    Cluster minors into 1-3 sized uniform groups and assign each group to a non-nursing home

    c) "Almost a quarter of U.S. children under the age of 18 live with one parent and no other adults (23%)"
    - https://www.pewresearch.org
        /fact-tank/2019/12/12/u-s-children-more-likely-than-children-in-other-countries-to-live-with-just-one-parent/

    one adult to 23% minor homes,
    distribute the remaining adults and retirees in the remaining minor and non-nursing homes

    :param sim_config: PandemicSimConfig instance
    :return: a list of person instances
    """
    assert globals.registry, 'No registry found. Create the repo wide registry first by calling init_globals()'
    registry = globals.registry
    numpy_rng = globals.numpy_rng

    persons: List[Person] = []

    # ages based on the age profile of USA
    ages = get_us_age_distribution(sim_config.num_persons)
    numpy_rng.shuffle(ages)
    minor_ages = []
    adult_ages = []
    retiree_ages = []
    for age in ages:
        if age <= 18:
            minor_ages.append(age)
        elif 18 < age <= 65:
            adult_ages.append(age)
        else:
            retiree_ages.append(age)

    all_homes = list(registry.location_ids_of_type(Home))
    numpy_rng.shuffle(all_homes)
    unassigned_homes = all_homes

    # a) Select 6.5% of retirees (age > 65) and cluster them as groups of 1 or 2 and assign each
    # group to a nursing home.
    num_retirees_in_nursing = np.ceil(len(retiree_ages) * 0.065).astype('int')
    retirees_in_nursing_ages = retiree_ages[:num_retirees_in_nursing]
    clustered_nursing_ages = cluster_into_random_sized_groups(retirees_in_nursing_ages, 1, 2, numpy_rng)
    retiree_homes_ages = [(unassigned_homes[_i], _a) for _i, _g in enumerate(clustered_nursing_ages) for _a in _g]
    nursing_homes = unassigned_homes[:len(clustered_nursing_ages)]
    unassigned_homes = unassigned_homes[len(nursing_homes):]  # remove assigned nursing homes
    unassigned_retiree_ages = retiree_ages[num_retirees_in_nursing:]
    # create retirees in nursing homes
    for home, age in retiree_homes_ages:
        persons.append(Retired(person_id=PersonID(f'retired_{str(uuid4())}', age),
                               home=home,
                               regulation_compliance_prob=sim_config.regulation_compliance_prob,
                               init_state=PersonState(current_location=home, risk=infection_risk(age))))

    # b) Cluster minors into 1-3 sized uniform groups and assign each group to a home
    schools = registry.location_ids_of_type(School)
    clustered_minor_ages = cluster_into_random_sized_groups(minor_ages, 1, 3, numpy_rng)
    assert len(unassigned_homes) >= len(clustered_minor_ages), 'not enough homes to assign all people'
    minor_homes_ages = [(unassigned_homes[_i], _a) for _i, _g in enumerate(clustered_minor_ages) for _a in _g]
    minor_homes = unassigned_homes[:len(clustered_minor_ages)]
    unassigned_homes = unassigned_homes[len(minor_homes):]  # remove assigned minor homes
    # create all minor
    for home, age in minor_homes_ages:
        persons.append(Minor(person_id=PersonID(f'minor_{str(uuid4())}', age),
                             home=home,
                             school=numpy_rng.choice(schools) if len(schools) > 0 else None,
                             regulation_compliance_prob=sim_config.regulation_compliance_prob,
                             init_state=PersonState(current_location=home, risk=infection_risk(age))))

    # c) Assign one adult to each minor-included homes and then distribute the remaining uniformly across
    # all non-nursing homes
    # nursing_homes, minor_homes, unassigned_homes
    required_num_adults = len(minor_homes) + len(unassigned_homes) - len(unassigned_retiree_ages)
    assert len(adult_ages) >= required_num_adults, (
        f'not enough adults {required_num_adults} to ensure each minor home has at least a single '
        f'adult and all the homes are filled.')
    adult_homes_ages = [(_h, adult_ages[_i]) for _i, _h in enumerate(minor_homes)]
    non_nursing_homes_ages = []

    non_single_parent_minor_homes = minor_homes[int(len(minor_homes) * 0.23):]
    homes_to_distribute = unassigned_homes + non_single_parent_minor_homes
    numpy_rng.shuffle(homes_to_distribute)
    unassigned_adult_ages = adult_ages[len(minor_homes):]
    for i in range(len(unassigned_retiree_ages) + len(unassigned_adult_ages)):
        home = homes_to_distribute[i % len(homes_to_distribute)]
        if len(unassigned_retiree_ages) > 0:
            age = unassigned_retiree_ages.pop(0)
            non_nursing_homes_ages.append((home, age))
        else:
            age = unassigned_adult_ages.pop(0)
            adult_homes_ages.append((home, age))

    work_ids = registry.location_ids_of_type(BusinessBaseLocation)
    assert len(work_ids) > 0, 'no business locations found!'
    for home, age in adult_homes_ages:
        job_counselor = JobCounselor(sim_config.location_configs)
        work_package = job_counselor.next_available_work()
        assert work_package, 'Not enough available jobs, increase the capacity of certain businesses'
        persons.append(Worker(person_id=PersonID(f'worker_{str(uuid4())}', age),
                              home=home,
                              work=work_package.work,
                              work_time=work_package.work_time,
                              regulation_compliance_prob=sim_config.regulation_compliance_prob,
                              init_state=PersonState(current_location=home, risk=infection_risk(age))))

    for home, age in non_nursing_homes_ages:
        persons.append(Retired(person_id=PersonID(f'retired_{str(uuid4())}', age),
                               home=home,
                               regulation_compliance_prob=sim_config.regulation_compliance_prob,
                               init_state=PersonState(current_location=home, risk=infection_risk(age))))

    return persons
