# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from matplotlib import pyplot as plt

from pandemic_simulator.data import H5DataLoader
from pandemic_simulator.script_helpers import EvaluationOpts, evaluate_spread_rates, \
    make_evaluation_plots, evaluate_location_contact_rates, evaluate_social_gatherings, \
    make_evaluation_plots_from_data

if __name__ == '__main__':
    opts = EvaluationOpts(
        num_seeds=30,
        spread_rates=(0.01, 0.02, 0.03),
        social_distancing=(0., 0.3, 0.6),
        avoid_gathering_sizes=(-1, 5, 0),
    )
    assert opts.spread_rates
    assert opts.social_distancing
    assert opts.avoid_gathering_sizes

    name_to_eval_fn_labels = {
        'spread_rate': (evaluate_spread_rates, [str(v) for v in opts.spread_rates], 'Spread Rate (mean)'),
        'contact_rate': (evaluate_location_contact_rates, [f'{1 - v:.1f}' for v in opts.social_distancing],
                         'Contact Rate Scale'),
        'social_gatherings': (evaluate_social_gatherings, [str(v if v != -1 else 'None')
                                                           for v in opts.avoid_gathering_sizes],
                              'Max Gathering Size To Avoid'),
    }
    for name, (eval_fn, param_lbls, xlabel) in name_to_eval_fn_labels.items():
        try:
            experiment_name = name + '_sensitivity'
            eval_fn(exp_name=experiment_name, eval_opts=opts)
        except ValueError:
            # Expect a value error because we are reusing the same directory.
            pass

    for name, (eval_fn, param_labels, xlabel) in name_to_eval_fn_labels.items():
        experiment_name = name + '_sensitivity'
        if name == 'social_gatherings':
            loader = H5DataLoader(experiment_name, path=opts.data_saver_path)
            data = list(loader.get_data())
            data = data[0:1] + data[-2:]
            param_labels = param_labels[0:1] + param_labels[-2:]
            make_evaluation_plots_from_data(data=data[:5], exp_name=experiment_name,
                                            param_labels=param_labels, bar_plot_xlabel=xlabel,
                                            show_summary_plots=False,
                                            show_cumulative_reward=False,
                                            show_pandemic_duration=False)
        else:
            make_evaluation_plots(exp_name=experiment_name, data_saver_path=opts.data_saver_path,
                                  param_labels=param_labels, bar_plot_xlabel=xlabel,
                                  show_summary_plots=False,
                                  show_cumulative_reward=False)

    plt.show()
