# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from tqdm import trange

import pandemic_simulator as ps


def impose_regulations() -> None:
    """In all the tutorial so far, we ran the simulator under no movement restrictions (regulations). This tutorial
    shows how to impose regulations."""

    print('\nA tutorial to impose pandemic regulations in the environment', flush=True)

    # the first thing to do at the start of any experiment is to initialize a few global parameters
    # these parameters are shared across the entire repo
    ps.init_globals(seed=0)

    # generate a simulator config (see `python/pandemic_simulator/script_helpers/sim_configs.py` for more configs)
    sim_config = ps.env.PandemicSimConfig(
        num_persons=10,
        location_configs=[
            ps.env.LocationConfig(location_type=ps.env.Home, num=4),
            ps.env.LocationConfig(location_type=ps.env.GroceryStore, num=1),
            ps.env.LocationConfig(location_type=ps.env.Office, num=1),
            ps.env.LocationConfig(location_type=ps.env.School, num=1)
        ])

    # init simulator
    sim = ps.env.PandemicSim.from_config(sim_config)

    # setup viz to show plots
    viz = ps.viz.SimViz.from_config(sim_config)

    # define two custom pandemic regulations (see ps.sh.austin_regulations for realistic regulations)
    regulation_1 = ps.env.PandemicRegulation(  # moderate restriction
        stay_home_if_sick=True,  # stay home if sick
        location_type_to_rule_kwargs={
            ps.env.Office: {'lock': False},  # unlock office (if locked)
        },
        stage=1  # a discrete identifier for this regulation
    )
    regulation_2 = ps.env.PandemicRegulation(  # restricted movement
        stay_home_if_sick=True,  # stay home if sick
        location_type_to_rule_kwargs={
            ps.env.Office: {'lock': True},  # also lock office
        },
        stage=2  # a discrete identifier for this regulation
    )

    # Iterate with no restrictions
    for _ in trange(3, desc='Simulating day (no restrictions)'):
        sim.step_day()
        viz.record_state(sim.state)

    # Iterate after imposing stage 1 restrictions
    sim.impose_regulation(regulation_1)
    for _ in trange(3, desc='Simulating day (stage 1)'):
        sim.step_day()
        viz.record_state(sim.state)

    # Iterate after imposing stage 2 restrictions
    sim.impose_regulation(regulation_2)
    for _ in trange(3, desc='Simulating day (stage 2)'):
        sim.step_day()
        viz.record(sim.state)

    # display plots to show grocery store (visitor visits)
    viz.plot([ps.viz.PlotType.global_infection_summary, ps.viz.PlotType.stages])


if __name__ == '__main__':
    impose_regulations()
