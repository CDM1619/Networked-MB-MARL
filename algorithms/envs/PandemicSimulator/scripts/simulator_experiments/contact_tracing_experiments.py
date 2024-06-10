# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from matplotlib import pyplot as plt

from pandemic_simulator.environment import PandemicRegulation, PandemicSimOpts
from pandemic_simulator.script_helpers import EvaluationOpts, evaluate_strategies, make_evaluation_plots

if __name__ == '__main__':
    regulations = [
        PandemicRegulation(stay_home_if_sick=False, stage=0),
        PandemicRegulation(stay_home_if_sick=True, stage=1),
        PandemicRegulation(stay_home_if_sick=True, quarantine_if_contact_positive=True, stage=2)
    ]
    name_to_strategy_sim_opt = {
        'NONE': (0, PandemicSimOpts()),
        'SICK': (1, PandemicSimOpts()),
        'CON-2': (2, PandemicSimOpts(use_contact_tracer=True, contact_tracer_history_size=2)),
        'CON-5': (2, PandemicSimOpts(use_contact_tracer=True, contact_tracer_history_size=5)),
        'CON-10': (2, PandemicSimOpts(use_contact_tracer=True, contact_tracer_history_size=10)),
        'SICK+': (1, PandemicSimOpts(spontaneous_testing_rate=0.3)),
        'CON-2+': (2, PandemicSimOpts(use_contact_tracer=True, contact_tracer_history_size=2,
                                      spontaneous_testing_rate=0.3)),
        'CON-5+': (2, PandemicSimOpts(use_contact_tracer=True, contact_tracer_history_size=5,
                                      spontaneous_testing_rate=0.3)),
        'CON-10+': (2, PandemicSimOpts(use_contact_tracer=True, contact_tracer_history_size=10,
                                       spontaneous_testing_rate=0.3)),
        'SICK++': (1, PandemicSimOpts(spontaneous_testing_rate=1.)),
    }

    param_labels, strategies, sim_opts = zip(*[(k, v[0], v[1]) for k, v in name_to_strategy_sim_opt.items()])

    opts = EvaluationOpts(
        num_seeds=1,
        strategies=strategies,
        pandemic_regulations=regulations,
        sim_opts=sim_opts,  # type: ignore
        max_episode_length=10,
        enable_warm_up=False
    )

    experiment_name = 'contact_tracing'
    try:
        evaluate_strategies(experiment_name, opts)
    except ValueError:
        # Expect a value error because we are reusing the same directory.
        pass
    make_evaluation_plots(exp_name=experiment_name, data_saver_path=opts.data_saver_path, param_labels=param_labels,
                          bar_plot_xlabel='Contact Tracing', show_cumulative_reward=False, annotate_stages=False)
    plt.show()
