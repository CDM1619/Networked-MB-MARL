# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import pytest

from pandemic_simulator.environment import SimTime, SimTimeInterval


def test_sim_time_interval_trigger_hour() -> None:
    hour = 5
    interval = SimTimeInterval(hour=hour)
    for i in range(1, 3):
        assert interval.trigger_at_interval(SimTime(hour=hour * i))
        assert not interval.trigger_at_interval(SimTime(hour=hour * i + 1))
        assert not interval.trigger_at_interval(SimTime(hour=hour * i - 1))


def test_sim_time_interval_trigger_day() -> None:
    day = 5
    hour = 6
    interval = SimTimeInterval(day=day, hour=hour)
    for i in range(1, 3):
        assert interval.trigger_at_interval(SimTime(day=day * i, hour=hour * i))
        assert not interval.trigger_at_interval(SimTime(day=day * i + 1))
        assert not interval.trigger_at_interval(SimTime(day=day * i - 1))
        assert not interval.trigger_at_interval(SimTime(day=day, hour=1))


@pytest.mark.parametrize(['hour', 'day', 'offset_hour', 'offset_day'],
                         [
                             [1, 3, 2, 1],
                             [0, 1, 5, 0],
                             [2, 0, 1, 0],
                             [0, 2, 0, 1]
                         ]
                         )
def test_sim_time_interval_trigger_offset(hour: int, day: int, offset_hour: int, offset_day: int) -> None:
    interval = SimTimeInterval(hour=hour, day=day, offset_hour=offset_hour, offset_day=offset_day)

    if offset_day > 0:
        for i in range(offset_day + 1):
            for j in range(24 if i < offset_day else offset_hour):
                assert not interval.trigger_at_interval(SimTime(day=day, hour=i))
    else:
        for i in range(offset_hour):
            assert not interval.trigger_at_interval(SimTime(hour=i))

    for i in range(1, 3):
        assert interval.trigger_at_interval(SimTime(day=day * i + offset_day, hour=hour * i + offset_hour))
        assert not interval.trigger_at_interval(SimTime(day=day * i, hour=hour * i))
