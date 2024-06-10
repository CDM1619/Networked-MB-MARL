# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import dataclasses
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, List

from .experiments import experiment_main
from .sim_configs import small_town_config, medium_town_config, above_medium_town_config
from ..data import H5DataSaver, StageSchedule
from ..environment import PandemicSimOpts, PandemicSimConfig, PandemicRegulation, Risk

__all__ = ['EvaluationOpts', 'evaluate_strategies', 'evaluate_spread_rates', 'evaluate_location_contact_rates',
           'evaluate_population_sizes', 'evaluate_social_gatherings', 'evaluate_testing_rates',
           'population_size_to_config']

population_size_to_config: Dict[int, PandemicSimConfig] = {pp.num_persons: pp for pp in [
    small_town_config,
    medium_town_config,
    above_medium_town_config]}


@dataclasses.dataclass
class EvaluationOpts:
    num_seeds: int
    spread_rates: Optional[Sequence[float]] = None
    pandemic_test_rate_scales: Optional[Sequence[float]] = None
    avoid_gathering_sizes: Optional[Sequence[int]] = None
    social_distancing: Optional[Sequence[float]] = None
    population_sizes: Optional[Sequence[int]] = None
    strategies: Optional[Sequence[Union[int, Sequence[StageSchedule]]]] = None
    pandemic_regulations: Optional[List[PandemicRegulation]] = None
    default_sim_config: PandemicSimConfig = small_town_config
    sim_opts: Optional[List[PandemicSimOpts]] = None

    enable_warm_up: bool = False
    max_episode_length: int = 120
    data_saver_path: Path = Path('../results/')
    data_filename: str = dataclasses.field(init=False)
    render_runs: bool = False

    def __post_init__(self) -> None:
        os.makedirs(str(self.data_saver_path.absolute()), exist_ok=True)


def evaluate_strategies(exp_name: str, eval_opts: EvaluationOpts) -> None:
    assert eval_opts.strategies is not None
    data_saver = H5DataSaver(exp_name, path=eval_opts.data_saver_path)

    for i, strategy in enumerate(eval_opts.strategies):
        stage_schedule = [StageSchedule(stage=strategy, end_day=None)] if isinstance(strategy, int) else strategy

        txt_strategy = [f'(stage: {s.stage} end: {s.end_day})' for s in stage_schedule]
        sim_opts = eval_opts.sim_opts[i] if eval_opts.sim_opts is not None else PandemicSimOpts()

        print(f'Evaluating strategy - {", ".join(txt_strategy)}')
        experiment_main(sim_config=eval_opts.default_sim_config,
                        sim_opts=sim_opts,
                        data_saver=data_saver,
                        pandemic_regulations=eval_opts.pandemic_regulations,
                        stages_to_execute=strategy,
                        enable_warm_up=eval_opts.enable_warm_up,
                        num_random_seeds=eval_opts.num_seeds,
                        max_episode_length=eval_opts.max_episode_length,
                        exp_id=i)


def evaluate_spread_rates(exp_name: str, eval_opts: EvaluationOpts) -> None:
    assert eval_opts.spread_rates is not None
    data_saver = H5DataSaver(exp_name, path=eval_opts.data_saver_path)
    for i, spread_rate in enumerate(eval_opts.spread_rates):
        print(f'Evaluating spread_rate - {spread_rate}')
        sim_opts = PandemicSimOpts(infection_spread_rate_mean=spread_rate)
        experiment_main(sim_config=eval_opts.default_sim_config,
                        sim_opts=sim_opts,
                        data_saver=data_saver,
                        num_random_seeds=eval_opts.num_seeds,
                        max_episode_length=eval_opts.max_episode_length,
                        exp_id=i)


def evaluate_testing_rates(exp_name: str, eval_opts: EvaluationOpts) -> None:
    assert eval_opts.pandemic_test_rate_scales is not None
    data_saver = H5DataSaver(exp_name, path=eval_opts.data_saver_path)
    def_sim_opts = PandemicSimOpts()

    for i, parameter_scale in enumerate(eval_opts.pandemic_test_rate_scales):
        print(f'Evaluating testing_rate_scale - {parameter_scale}')
        sim_opts = PandemicSimOpts(spontaneous_testing_rate=def_sim_opts.spontaneous_testing_rate * parameter_scale,
                                   symp_testing_rate=def_sim_opts.symp_testing_rate * parameter_scale,
                                   retest_rate=def_sim_opts.retest_rate * parameter_scale)
        experiment_main(sim_config=eval_opts.default_sim_config,
                        sim_opts=sim_opts,
                        data_saver=data_saver,
                        pandemic_regulations=[PandemicRegulation(stay_home_if_sick=True, stage=0)],
                        stages_to_execute=0,
                        num_random_seeds=eval_opts.num_seeds,
                        max_episode_length=eval_opts.max_episode_length,
                        exp_id=i)


def evaluate_social_gatherings(exp_name: str, eval_opts: EvaluationOpts) -> None:
    assert eval_opts.avoid_gathering_sizes is not None
    data_saver = H5DataSaver(exp_name, path=eval_opts.data_saver_path)
    pandemic_regulations = [
        PandemicRegulation(risk_to_avoid_gathering_size={Risk.LOW: ags, Risk.HIGH: ags}, stage=stage)
        for stage, ags in enumerate(eval_opts.avoid_gathering_sizes)]

    for i, cr in enumerate(pandemic_regulations):
        print(f'Evaluating social_gathering_size_to_avoid - {cr.risk_to_avoid_gathering_size[Risk.LOW]}')
        experiment_main(sim_config=eval_opts.default_sim_config,
                        sim_opts=PandemicSimOpts(),
                        data_saver=data_saver,
                        pandemic_regulations=pandemic_regulations,
                        stages_to_execute=cr.stage,
                        num_random_seeds=eval_opts.num_seeds,
                        max_episode_length=eval_opts.max_episode_length,
                        exp_id=i)


def evaluate_location_contact_rates(exp_name: str, eval_opts: EvaluationOpts) -> None:
    assert eval_opts.social_distancing is not None
    data_saver = H5DataSaver(exp_name, path=eval_opts.data_saver_path)
    pandemic_regulations = [PandemicRegulation(social_distancing=sd, stage=stage)
                            for stage, sd in enumerate(eval_opts.social_distancing)]
    for i, cr in enumerate(pandemic_regulations):
        print(f'Evaluating social_distancing - {cr.social_distancing}')
        experiment_main(sim_config=eval_opts.default_sim_config,
                        sim_opts=PandemicSimOpts(),
                        data_saver=data_saver,
                        pandemic_regulations=pandemic_regulations,
                        stages_to_execute=cr.stage,
                        num_random_seeds=eval_opts.num_seeds,
                        max_episode_length=eval_opts.max_episode_length,
                        exp_id=i)


def evaluate_population_sizes(exp_name: str, eval_opts: EvaluationOpts) -> None:
    assert eval_opts.population_sizes is not None
    data_saver = H5DataSaver(exp_name, path=eval_opts.data_saver_path)

    for i, population_size in enumerate(eval_opts.population_sizes):
        print(f'Evaluating population_size - {population_size}')
        experiment_main(sim_config=population_size_to_config[population_size],
                        sim_opts=PandemicSimOpts(),
                        data_saver=data_saver,
                        num_random_seeds=eval_opts.num_seeds,
                        max_episode_length=eval_opts.max_episode_length,
                        exp_id=i)
