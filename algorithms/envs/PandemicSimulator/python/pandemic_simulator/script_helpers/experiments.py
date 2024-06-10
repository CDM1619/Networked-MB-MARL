# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from typing import List, Optional, Sequence, Union

import numpy as np
from tqdm import trange

from .covid_regulations import austin_regulations
from ..data.interfaces import ExperimentDataSaver, StageSchedule
from ..environment import PandemicSimOpts, PandemicSimConfig, NoPandemicDone, PandemicRegulation, init_globals, \
    PandemicGymEnv
from ..utils import shallow_asdict

__all__ = ['experiment_main', 'seeded_experiment_main']


def seeded_experiment_main(exp_id: int,
                           sim_config: PandemicSimConfig,
                           sim_opts: PandemicSimOpts,
                           data_saver: ExperimentDataSaver,
                           pandemic_regulations: Optional[List[PandemicRegulation]] = None,
                           stages_to_execute: Union[int, Sequence[StageSchedule]] = 0,
                           enable_warm_up: bool = False,
                           max_episode_length: int = 120,
                           random_seed: int = 0) -> bool:
    """A helper that runs an experiment with the given seed and records data"""
    init_globals(seed=random_seed)
    env = PandemicGymEnv.from_config(sim_config=sim_config,
                                     sim_opts=sim_opts,
                                     pandemic_regulations=pandemic_regulations or austin_regulations,
                                     done_fn=NoPandemicDone(30))
    env.reset()

    stages = ([StageSchedule(stage=stages_to_execute, end_day=None)]
              if isinstance(stages_to_execute, int) else stages_to_execute)

    stage_dict = {f'stage_{i}': (s.stage, s.end_day if s.end_day is not None else -1)
                  for i, s in enumerate(stages)}

    data_saver.begin(env.observation)

    stage_idx = 0
    warm_up_done = not enable_warm_up
    for i in trange(max_episode_length, desc='Simulating day'):

        if not env.observation.infection_above_threshold and not warm_up_done:
            stage = 0
        else:
            warm_up_done = True
            cur_stage = stages[stage_idx]
            stage = cur_stage.stage
            if cur_stage.end_day is not None and cur_stage.end_day <= i:
                stage_idx += 1

        obs, reward, done, aux = env.step(stage)
        data_saver.record(obs, reward)
        if done:
            print('done')
            break
    return data_saver.finalize(exp_id=exp_id,
                               seed=random_seed,
                               num_stages_to_execute=len(stages),
                               num_persons=sim_config.num_persons,
                               **stage_dict,
                               **shallow_asdict(sim_opts))


def experiment_main(exp_id: int,
                    sim_opts: PandemicSimOpts,
                    sim_config: PandemicSimConfig,
                    data_saver: ExperimentDataSaver,
                    pandemic_regulations: Optional[List[PandemicRegulation]] = None,
                    stages_to_execute: Union[int, Sequence[StageSchedule]] = 0,
                    enable_warm_up: bool = False,
                    max_episode_length: int = 120,
                    num_random_seeds: int = 5) -> None:
    """A helper that runs multi-seeded experiments and records data."""
    rng = np.random.RandomState(seed=0)
    num_evaluated_seeds = 0
    while num_evaluated_seeds < num_random_seeds:
        seed = rng.randint(0, 100000)
        print(f'Running experiment seed: {seed} - {num_evaluated_seeds + 1}/{num_random_seeds}')
        ret = seeded_experiment_main(exp_id=exp_id,
                                     sim_config=sim_config,
                                     sim_opts=sim_opts,
                                     data_saver=data_saver,
                                     pandemic_regulations=pandemic_regulations,
                                     stages_to_execute=stages_to_execute,
                                     enable_warm_up=enable_warm_up,
                                     max_episode_length=max_episode_length,
                                     random_seed=seed)
        if ret:
            num_evaluated_seeds += 1
        else:
            print(f'Experiment with seed {seed} did not succeed. Skipping...')
