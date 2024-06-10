# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from .person_routines import DefaultPersonRoutineAssignment
from ..environment import Home, GroceryStore, Office, School, Hospital, RetailStore, HairSalon, Restaurant, Bar, \
    PandemicSimConfig, LocationConfig

__all__ = ['town_config', 'small_town_config', 'test_config',
           'tiny_town_config', 'medium_town_config',
           'above_medium_town_config']

"""
A few references for the numbers selected:

http://www.worldcitiescultureforum.com/data/number-of-restaurants-per-100.000-population (Austin)

"""

town_config = PandemicSimConfig(
    num_persons=10000,
    location_configs=[
        LocationConfig(Home, num=3000),
        LocationConfig(GroceryStore, num=40, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Office, num=50, num_assignees=150, state_opts=dict(visitor_capacity=0)),
        LocationConfig(School, num=100, num_assignees=4, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Hospital, num=10, num_assignees=30, state_opts=dict(patient_capacity=10)),
        LocationConfig(RetailStore, num=40, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(HairSalon, num=40, num_assignees=3, state_opts=dict(visitor_capacity=5)),
        LocationConfig(Restaurant, num=20, num_assignees=6, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Bar, num=20, num_assignees=5, state_opts=dict(visitor_capacity=30)),
    ],
    person_routine_assignment=DefaultPersonRoutineAssignment())


above_medium_town_config = PandemicSimConfig(
    num_persons=5000,
    location_configs=[
        LocationConfig(Home, num=1200),
        LocationConfig(GroceryStore, num=16, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Office, num=20, num_assignees=150, state_opts=dict(visitor_capacity=0)),
        LocationConfig(School, num=40, num_assignees=4, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Hospital, num=4, num_assignees=30, state_opts=dict(patient_capacity=10)),
        LocationConfig(RetailStore, num=16, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(HairSalon, num=16, num_assignees=3, state_opts=dict(visitor_capacity=5)),
        LocationConfig(Restaurant, num=8, num_assignees=6, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Bar, num=8, num_assignees=4, state_opts=dict(visitor_capacity=30))
    ],
    person_routine_assignment=DefaultPersonRoutineAssignment())

medium_town_config = PandemicSimConfig(
    num_persons=2000,
    location_configs=[
        LocationConfig(Home, num=600),
        LocationConfig(GroceryStore, num=8, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Office, num=10, num_assignees=150, state_opts=dict(visitor_capacity=0)),
        LocationConfig(School, num=20, num_assignees=4, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Hospital, num=2, num_assignees=30, state_opts=dict(patient_capacity=10)),
        LocationConfig(RetailStore, num=8, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(HairSalon, num=8, num_assignees=3, state_opts=dict(visitor_capacity=5)),
        LocationConfig(Restaurant, num=4, num_assignees=6, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Bar, num=4, num_assignees=3, state_opts=dict(visitor_capacity=30))
    ],
    person_routine_assignment=DefaultPersonRoutineAssignment())

small_town_config = PandemicSimConfig(
    num_persons=1000,
    location_configs=[
        LocationConfig(Home, num=300),
        LocationConfig(GroceryStore, num=4, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Office, num=5, num_assignees=150, state_opts=dict(visitor_capacity=0)),
        LocationConfig(School, num=10, num_assignees=4, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Hospital, num=1, num_assignees=30, state_opts=dict(patient_capacity=10)),
        LocationConfig(RetailStore, num=4, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(HairSalon, num=4, num_assignees=3, state_opts=dict(visitor_capacity=5)),
        LocationConfig(Restaurant, num=2, num_assignees=6, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Bar, num=2, num_assignees=5, state_opts=dict(visitor_capacity=30)),
    ],
    person_routine_assignment=DefaultPersonRoutineAssignment())

tiny_town_config = PandemicSimConfig(
    num_persons=500,
    location_configs=[
        LocationConfig(Home, num=150),
        LocationConfig(GroceryStore, num=2, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Office, num=2, num_assignees=150, state_opts=dict(visitor_capacity=0)),
        LocationConfig(School, num=10, num_assignees=2, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Hospital, num=1, num_assignees=15, state_opts=dict(patient_capacity=5)),
        LocationConfig(RetailStore, num=2, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(HairSalon, num=2, num_assignees=3, state_opts=dict(visitor_capacity=5)),
        LocationConfig(Restaurant, num=1, num_assignees=6, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Bar, num=1, num_assignees=3, state_opts=dict(visitor_capacity=30))
    ],
    person_routine_assignment=DefaultPersonRoutineAssignment())

test_config = PandemicSimConfig(
    num_persons=100,
    location_configs=[
        LocationConfig(Home, num=30),
        LocationConfig(GroceryStore, num=1, num_assignees=5, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Office, num=1, num_assignees=150, state_opts=dict(visitor_capacity=0)),
        LocationConfig(School, num=10, num_assignees=2, state_opts=dict(visitor_capacity=30)),
        LocationConfig(Hospital, num=1, num_assignees=30, state_opts=dict(patient_capacity=2)),
        LocationConfig(Restaurant, num=1, num_assignees=3, state_opts=dict(visitor_capacity=10)),
        LocationConfig(Bar, num=1, num_assignees=3, state_opts=dict(visitor_capacity=10)),
    ],
    person_routine_assignment=DefaultPersonRoutineAssignment())
