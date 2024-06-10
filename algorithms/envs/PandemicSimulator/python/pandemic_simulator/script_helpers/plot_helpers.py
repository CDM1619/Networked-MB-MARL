# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
import os
import string
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from .sim_configs import small_town_config
from ..data import H5DataLoader, ExperimentResult
from ..environment import sorted_infection_summary, PandemicSimConfig
from ..viz import plot_global_infection_summary, plot_critical_summary, plot_multi_params_summary

__all__ = ['make_evaluation_plots_from_data', 'make_evaluation_plots']


def make_evaluation_plots_from_data(data: Sequence[ExperimentResult],
                                    exp_name: str,
                                    param_labels: Sequence[str],
                                    bar_plot_xlabel: str,
                                    fig_save_path: Path = Path('../results/plots'),
                                    sim_config: Optional[PandemicSimConfig] = None,
                                    show_summary_plots: bool = True,
                                    show_cumulative_reward: bool = False,
                                    show_time_to_peak: bool = True,
                                    show_pandemic_duration: bool = True,
                                    show_stage_trials: bool = False,
                                    annotate_stages: Union[bool, Sequence[bool]] = False,
                                    figsize: Optional[Tuple[int, int]] = None) -> None:
    n_params = len(param_labels)
    os.makedirs(str(fig_save_path.absolute()), exist_ok=True)
    annotate_stages = [annotate_stages] * n_params if isinstance(annotate_stages, bool) else annotate_stages

    sim_config = sim_config or small_town_config
    gis_legend = [summ.value for summ in sorted_infection_summary]

    sup_title = f"{' '.join([s.capitalize() for s in exp_name.split(sep='_')])}"

    plot_ref_labels = string.ascii_lowercase + string.ascii_uppercase
    plot_ref_label_i = 0

    gs1: Optional[GridSpec] = None
    if show_summary_plots:
        figsize = figsize if figsize is not None else ((16, 6) if n_params <= 5 else (20, 12))
        fig = plt.figure(num=sup_title, figsize=figsize)
        gs1 = GridSpec(n_params, 3)
        axs = np.array([fig.add_subplot(sp) for sp in gs1]).reshape(n_params, 3)
        for i, (exp_result, param_label, ann_stages) in enumerate(zip(data, param_labels, annotate_stages)):
            plot_global_infection_summary(exp_result, ax=axs[i, 0], annotate_stages=ann_stages)
            if show_stage_trials:
                seed_indices = np.random.permutation(len(exp_result.obs_trajectories.stage.shape[1]))[:3]
                for seed_i in seed_indices:
                    axs[i, 2].plot(exp_result.obs_trajectories.stage[:, seed_i, 0])
                axs[i, 2].set_title(f'Stages over Time\n(shown for {len(seed_indices)} trials)')
                axs[i, 2].set_yticks([0, np.max(exp_result.obs_trajectories.stage)])
                axs[i, 2].set_yticklabels(['Open\n(Stage-0)',
                                           f'Lockdown\n(Stage-{int(np.max(exp_result.obs_trajectories.stage))})'])
                axs[i, 2].set_xlabel('time (days)')
            else:
                plot_global_infection_summary(exp_result, testing_summary=True, annotate_stages=ann_stages,
                                              ax=axs[i, 1])

            plot_critical_summary(exp_result,
                                  max_hospital_capacity=sim_config.max_hospital_capacity,
                                  annotate_stages=ann_stages,
                                  ax=axs[i, 1] if show_stage_trials else axs[i, 2])

            for j, ax in enumerate(axs[i]):
                ref_label_offset = 20
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(ax.get_ylabel(), rotation=-90, labelpad=10)
                if i < (axs.shape[0] - 1):
                    ax.set_xlabel('')
                    ref_label_offset = 0
                if j < (axs.shape[1] - 1):
                    ax.set_ylabel('')
                if i > 0:
                    ax.set_title('')

                axs[i, j].annotate(f'({plot_ref_labels[plot_ref_label_i]})', (0.5, 0.),
                                   xytext=(0, -25 - ref_label_offset),
                                   textcoords='offset points', xycoords='axes fraction',
                                   ha='center', va='center', size=14)
                plot_ref_label_i += 1

            axs[i, 0].annotate(f'{param_label}', (0, 0.5), xytext=(-15, 0),
                               textcoords='offset points', xycoords='axes fraction',
                               ha='center', va='center', size=14, rotation=90)

        handles = (axs[0, 0].get_legend_handles_labels()[0] + axs[0, 2].get_legend_handles_labels()[0][-1:])
        axs[0, 0].legend(handles, gis_legend + ['Max hospital capacity', 'cumulative_reward'],
                         fancybox=True, loc='best', fontsize=4 if n_params > 3 else 6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # This raises warnings since tight layout cannot
            # handle gridspec automatically. We are going to
            # do that manually so we can filter the warning.
            gs1.tight_layout(fig, rect=[0.01, None, 0.6, None])

        bar_plots_2d = (3 + int(show_time_to_peak) + int(show_pandemic_duration) + int(show_cumulative_reward))
        bar_plots_3d = 0
        total_bar_plots = bar_plots_2d + bar_plots_3d
        n_cols = int(np.round(np.sqrt(total_bar_plots)))
        n_rows = int(n_cols) if n_cols ** 2 >= total_bar_plots else n_cols + 1

    else:
        bar_plots_2d = (3 + int(show_time_to_peak) + int(show_pandemic_duration) + int(show_cumulative_reward))
        bar_plots_3d = 0
        total_bar_plots = bar_plots_2d + bar_plots_3d
        n_cols = 3 if total_bar_plots > 4 else 2
        n_rows = np.ceil(total_bar_plots / n_cols).astype('int')

        figsize = figsize if figsize is not None else (10, n_rows * 3)
        fig = plt.figure(num=sup_title, figsize=figsize)

    gs2 = gridspec.GridSpec(n_rows, n_cols)
    axs = []
    plot_i = 0
    for sp in gs2:
        if plot_i < bar_plots_2d:
            axs.append(fig.add_subplot(sp))
        elif plot_i < (bar_plots_2d + bar_plots_3d):
            axs.append(fig.add_subplot(sp, projection='3d'))
        plot_i += 1

    plot_multi_params_summary(data,
                              param_labels=param_labels,
                              xlabel=bar_plot_xlabel,
                              max_hospitals_capacities=[sim_config.max_hospital_capacity] * len(data),
                              show_cumulative_reward_plot=show_cumulative_reward,
                              show_time_to_peak=show_time_to_peak,
                              show_pandemic_duration=show_pandemic_duration,
                              axs=axs)

    for ax in axs:
        if ax.axison or isinstance(ax, Axes3D):
            offset = 20 if max([len(label) for label in param_labels]) < 5 else 30
            ax.annotate(f'({plot_ref_labels[plot_ref_label_i]})', (0.5, 0.), xytext=(0, -25 - offset),
                        textcoords='offset points', xycoords='axes fraction',
                        ha='center', va='center', size=14)
            plot_ref_label_i += 1

    if gs1 is not None:
        with warnings.catch_warnings():
            # This raises warnings since tight layout cannot
            # handle gridspec automatically. We are going to
            # do that manually so we can filter the warning.
            warnings.simplefilter("ignore", UserWarning)
            gs2.tight_layout(fig, rect=[0.6, None, None, None])

        top = min(gs1.top, gs2.top)
        bottom = max(gs1.bottom, gs2.bottom)

        gs1.update(top=top, bottom=bottom)
        gs2.update(top=top, bottom=bottom)
        plt.annotate(bar_plot_xlabel, (0.01, 0.5), xytext=(0, 0),
                     textcoords='offset points', xycoords='figure fraction',
                     ha='center', va='center', size=16, rotation=90)
    else:
        plt.tight_layout()

    plt.savefig(fig_save_path / (exp_name + '.pdf'))


def make_evaluation_plots(exp_name: str,
                          param_labels: Sequence[str],
                          bar_plot_xlabel: str,
                          data_saver_path: Path = Path('../results'),
                          sim_config: Optional[PandemicSimConfig] = None,
                          show_summary_plots: bool = True,
                          show_cumulative_reward: bool = False,
                          show_time_to_peak: bool = True,
                          show_pandemic_duration: bool = True,
                          show_stage_trials: bool = False,
                          annotate_stages: Union[bool, Sequence[bool]] = False,
                          figsize: Optional[Tuple[int, int]] = None) -> None:
    loader = H5DataLoader(exp_name, path=data_saver_path)
    data: Sequence[ExperimentResult] = loader.get_data()[:len(param_labels)]
    make_evaluation_plots_from_data(data=data,
                                    exp_name=exp_name,
                                    param_labels=param_labels,
                                    bar_plot_xlabel=bar_plot_xlabel,
                                    fig_save_path=data_saver_path / 'plots',
                                    sim_config=sim_config,
                                    show_summary_plots=show_summary_plots,
                                    show_cumulative_reward=show_cumulative_reward,
                                    show_time_to_peak=show_time_to_peak,
                                    show_pandemic_duration=show_pandemic_duration,
                                    show_stage_trials=show_stage_trials,
                                    annotate_stages=annotate_stages,
                                    figsize=figsize)
