# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import Sequence, Type

from tqdm import trange

import pandemic_simulator as ps
from pandemic_simulator.environment import Person, Location


def using_person_routine_assignment() -> None:
    """In the tutorial 2_simple_worker_loop_with_routines.py, we showed how to add routines manually for each person.
    However, this can become tedious if we want to add for an entire population. To this end, the simulator provides
    a person routine assignment interface and this tutorial shows how to use it."""

    print('\nA tutorial to use PersonRoutineAssignment', flush=True)

    # the first thing to do at the start of any experiment is to initialize a few global parameters
    # these parameters are shared across the entire repo
    ps.init_globals(seed=0)

    # define an implementation of the PersonRoutineAssignment interface (see also
    # ps.sh.DefaultPersonRoutineAssignment() for a more realistic example)
    class MyAssignment(ps.env.PersonRoutineAssignment):
        """This is a callable class that gets used by the simulator to assign a routine for each person."""

        @property
        def required_location_types(self) -> Sequence[Type[Location]]:
            """Specify the a tuple of location types that are required for this routine assignment"""
            return ps.env.GroceryStore,

        def assign_routines(self, persons: Sequence[Person]) -> None:
            """
            Here, we implement a person routine for each person in the simulator.

            :param persons: A sequence of all person in the simulator
            :return: None
            """
            # iterate through each person
            for p in persons:
                # we check the category of the person
                if isinstance(p, ps.env.Retired):
                    # routines for retirees
                    routines = [
                        # we use a helper that sets up a triggered routine once every 7 days to visit the grocery store
                        ps.env.triggered_routine(None, ps.env.GroceryStore, 7),
                    ]
                    p.set_routines(routines)  # set the routine

                elif isinstance(p, ps.env.Minor):
                    # routines for minors, not implemented in this example
                    pass
                elif isinstance(p, ps.env.Worker):
                    # routines for workers
                    routines = [
                        # we use a helper that sets up a triggered routine once every 7 days to visit the grocery store
                        ps.env.triggered_routine(None, ps.env.GroceryStore, 7),
                    ]
                    p.set_outside_work_routines(routines)  # set as a outside work routine

    # generate a simulator config (see `python/pandemic_simulator/script_helpers/sim_configs.py` for more configs)
    sim_config = ps.env.PandemicSimConfig(
        num_persons=20,
        location_configs=[
            ps.env.LocationConfig(location_type=ps.env.Home, num=3),
            ps.env.LocationConfig(location_type=ps.env.GroceryStore, num=1),
            ps.env.LocationConfig(location_type=ps.env.Office, num=1),
            ps.env.LocationConfig(location_type=ps.env.School, num=1)
        ],
        person_routine_assignment=MyAssignment(),
    )

    # Init simulator by passing the person routine assignment instance
    sim = ps.env.PandemicSim.from_config(sim_config)

    # setup viz to show plots
    viz = ps.viz.SimViz.from_config(sim_config)

    # Iterate by advancing in days by calling step_day in the simulator
    for _ in trange(20, desc='Simulating day'):
        sim.step_day()
        viz.record(sim.state)

    # display plots to show grocery store (visitor visits)
    viz.plot([ps.viz.PlotType.global_infection_summary,
              ps.viz.PlotType.location_assignee_visits,
              ps.viz.PlotType.location_visitor_visits])


if __name__ == '__main__':
    using_person_routine_assignment()
