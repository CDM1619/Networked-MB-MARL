# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List

from matplotlib import pyplot as plt
from numpy import max
from tqdm import trange

import pandemic_simulator as ps


def run(days: int, stage: int, days_per_interval: int) -> List[int]:
    # init globals
    ps.init_globals(seed=100)

    # setup simulator config and options
    sim_config = ps.sh.town_config                                               #  ['town_config', 'small_town_config', 'test_config', 'tiny_town_config', 'medium_town_config', 'above_medium_town_config']
    sim_opts = ps.env.PandemicSimOpts(use_contact_tracer=True)  # use defaults

    # make sim
    sim = ps.env.PandemicSim.from_config(sim_config, sim_opts)

    # setup viz
    viz = ps.viz.GraphViz(sim, num_stages=len(ps.sh.austin_regulations), days_per_interval=days_per_interval)

    # execute the given regulation
    sim.impose_regulation(regulation=ps.sh.austin_regulations[stage])  # stage 0
    print(f'Stage {stage}:')

    # run regulation steps in the simulator
    for _ in trange(days, desc='Simulating day'):
        sim.step_day()
        viz.record(sim.state)

    return viz.num_components_per_interval


if __name__ == '__main__':
    title = "2-Week Interval"
    days_per_interval = 14

    stage_results = []
    for stage in range(0, 5):
        stage_results.append(run(days=100, stage=stage, days_per_interval=days_per_interval))

    plt.figure(figsize=(12, 8))

    for i in range(0, len(stage_results)):
        plt.plot(stage_results[i], label="Stage " + str(i))

    plt.ylim([-0.1, max(stage_results) + 1])
    plt.title('Connected Components by ' + title)
    plt.xlabel('time (' + title.lower() + 's)')
    plt.legend()

    plt.show()
