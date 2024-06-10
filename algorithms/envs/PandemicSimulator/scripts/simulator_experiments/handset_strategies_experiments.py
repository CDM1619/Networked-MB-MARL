# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from matplotlib import pyplot as plt

from pandemic_simulator.data import StageSchedule, H5DataLoader
from pandemic_simulator.script_helpers import EvaluationOpts, evaluate_strategies, make_evaluation_plots_from_data

if __name__ == '__main__':
    name_to_strategy = {
        'S0': 0,
        'S1': 1,
        'S2': 2,
        'S3': 3,
        'S4': 4,
        'S0-4-0': [StageSchedule(stage=4, end_day=30),
                   StageSchedule(stage=0, end_day=None)],
        'S0-4-0-FI': [StageSchedule(stage=4, end_day=30),
                      StageSchedule(stage=3, end_day=35),
                      StageSchedule(stage=2, end_day=40),
                      StageSchedule(stage=1, end_day=45),
                      StageSchedule(stage=0, end_day=None)],
        'S0-4-0-GI': [StageSchedule(stage=4, end_day=30),
                      StageSchedule(stage=3, end_day=50),
                      StageSchedule(stage=2, end_day=70),
                      StageSchedule(stage=1, end_day=90),
                      StageSchedule(stage=0, end_day=None)]
    }

    param_labels, strategies = zip(*[(k, v) for k, v in name_to_strategy.items()])

    opts = EvaluationOpts(
        num_seeds=30,
        strategies=strategies,
        max_episode_length=120,
        enable_warm_up=True
    )

    experiment_name = 'handset_strategies'
    try:
        evaluate_strategies(experiment_name, opts)
    except ValueError:
        # Expect a value error because we are reusing the same directory.
        pass

    loader = H5DataLoader('handset_strategies', path=opts.data_saver_path)
    data = list(loader.get_data())

    make_evaluation_plots_from_data(data=data[:5], exp_name='staged_regulations', param_labels=param_labels[:5],
                                    bar_plot_xlabel='Staged Regulations', show_cumulative_reward=False,
                                    show_time_to_peak=False, show_pandemic_duration=False,
                                    annotate_stages=True, figsize=(16, 8))

    plt.show()

    make_evaluation_plots_from_data(data=data[5:], exp_name='handset_strategies', param_labels=param_labels[5:],
                                    bar_plot_xlabel='Strategies', show_cumulative_reward=False,
                                    show_time_to_peak=False, show_pandemic_duration=False,
                                    annotate_stages=True)
    plt.show()
