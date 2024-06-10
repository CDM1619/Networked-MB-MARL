# Confidential, Copyright 2021, Sony Corporation of America, All rights reserved.

from ..interfaces import globals, SimTimeTuple

__all__ = ['get_work_time_for_24_7_open_locations']


def get_work_time_for_24_7_open_locations() -> SimTimeTuple:
    """Return a work time for a worker working at a 24x7 open location."""
    # roll the dice for day shift or night shift
    if globals.numpy_rng.random() < 0.5:
        # night shift
        hours = (22, 23) + tuple(range(0, 7))
    else:
        # distribute the work hours of the day shifts between 7 am to 10pm
        start = globals.numpy_rng.randint(7, 13)
        hours = tuple(range(start, start + 9))

    start = globals.numpy_rng.randint(0, 2)
    week_days = tuple(range(start, start + 6))
    return SimTimeTuple(hours, week_days)
