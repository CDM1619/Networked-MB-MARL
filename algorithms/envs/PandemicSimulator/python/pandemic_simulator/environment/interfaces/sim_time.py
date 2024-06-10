# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Type, Union

__all__ = ['SimTime', 'SimTimeInterval', 'SimTimeTuple']


@dataclass(frozen=True)
class SimTime:
    hour: int = 0
    week_day: int = 0
    day: int = 0
    year: int = 0

    def __post_init__(self) -> None:
        assert self.hour in range(0, 24), 'hour must be in (0, 23)'
        assert self.week_day in range(0, 7), 'Weekday must be in (0, 6)'
        assert self.day in range(0, 365), 'day must be in (0, 364)'

    def now(self, frmt: str = 'ydwh') -> List[int]:
        """Returns current time as list of ints in the specified format"""
        _map = {'h': self.hour, 'w': self.week_day, 'd': self.day, 'y': self.year}
        ret_list = []
        for _t in frmt:
            assert _t in _map, f'Unrecognized frmt string given {frmt}. frmt must be a ' \
                               f'combination string of {list(_map.keys())}.'
            ret_list.append(_map[_t])
        return ret_list

    def step(self) -> None:
        """Increments time by one discrete step"""
        h, w, d, y = self.now('hwdy')
        h += 1
        if h >= 24:
            h = 0
            w = (w + 1) % 7
            d += 1
            if d >= 365:
                d = 0
                y += 1
        object.__setattr__(self, 'hour', h)
        object.__setattr__(self, 'week_day', w)
        object.__setattr__(self, 'day', d)
        object.__setattr__(self, 'year', y)

    def in_hours(self) -> int:
        return self.year * 365 * 24 + self.day * 24 + self.hour

    @classmethod
    def from_hours(cls: Type, hours: int) -> 'SimTime':
        y = hours // (365 * 24)
        d = hours % (365 * 24) // 24
        h = hours % (365 * 24) % 24
        w = d % 7
        return SimTime(h, w, d, y)

    def __add__(self, other: Union['SimTime', 'SimTimeInterval']) -> 'SimTime':
        return SimTime.from_hours(other.in_hours() + self.in_hours())


@dataclass(frozen=True)
class SimTimeInterval:
    """Interval specified in hours/week_day/day/year"""

    hour: int = 0
    """Set a value in [0, 23] to indicate an interval in hours."""

    day: int = 0
    """Set a value in [0, 365] to indicate an interval in days"""

    year: int = 0
    """Set a value in >0 to indicate an interval in years"""

    offset_hour: int = 0
    """An offset in hours [0, 23]. Example - day = 1 and offset_hour = 12 would trigger at Noon everyday."""

    offset_day: int = 0
    """An offset in days [0, 365]. Example - day = 3 and offset_day = 1 would trigger once in 3 days starting a day
        later."""

    _trigger_hr: int = field(init=False)
    _offset_hr: int = field(init=False)

    def __post_init__(self) -> None:
        assert self.hour in range(24), 'Set a value in [1, 23] for an interval in hours'
        assert self.day in range(365), 'Set a value in [1, 365] for an interval in days'
        assert self.hour + self.day + self.year > 0, 'Provide a non-zero value at least for either of hour/day/year'
        object.__setattr__(self, '_trigger_hr', self.in_hours())
        object.__setattr__(self, '_offset_hr', self.offset_day * 24 + self.offset_hour)

    def trigger_at_interval(self, sim_time: SimTime) -> bool:
        """Return True at sim time interval and False otherwise."""
        sim_hr = sim_time.year * 365 * 24 + sim_time.day * 24 + sim_time.hour
        return (sim_hr - self._offset_hr) % self._trigger_hr == 0 if sim_hr >= self._offset_hr else False

    def in_hours(self) -> int:
        return self.year * 365 * 24 + self.day * 24 + self.hour


@dataclass(frozen=True)
class SimTimeTuple:
    hours: Optional[Tuple[int, ...]] = None
    week_days: Optional[Tuple[int, ...]] = None
    days: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.hours:
            for hour in self.hours:
                assert hour in range(0, 24), 'hour must be in (0, 23)'
        if self.week_days:
            for wd in self.week_days:
                assert wd in range(0, 7), 'Weekday must be in (0, 6)'
        if self.days:
            for d in self.days:
                assert d in range(0, 365), 'day must be in (0, 364)'

    def __contains__(self, item: SimTime) -> bool:
        contains = True
        if self.hours is not None:
            contains = contains and (item.hour in self.hours)
            if not contains:
                return False

        if self.week_days is not None:
            contains = contains and (item.week_day in self.week_days)
            if not contains:
                return False

        if self.days is not None:
            contains = contains and (item.day in self.days)
            if not contains:
                return False

        return True
