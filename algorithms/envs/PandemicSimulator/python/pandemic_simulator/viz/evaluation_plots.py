# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import Optional, Sequence, List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt, cycler
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from ..data import ExperimentResult, StageSchedule
from ..environment import sorted_infection_summary, InfectionSummary

__all__ = ['plot_cumulative_reward', 'plot_critical_summary', 'plot_global_infection_summary',
           'plot_multi_params_summary', 'plot_deaths_per_day_summary', 'get_stage_strategy',
           'plot_annotate_stages', 'inf_colors']

critical_index = sorted_infection_summary.index(InfectionSummary.CRITICAL)
infection_index = sorted_infection_summary.index(InfectionSummary.INFECTED)
deaths_index = sorted_infection_summary.index(InfectionSummary.DEAD)

inf_to_color = {InfectionSummary.NONE: (0, 1, 0, 1),
                InfectionSummary.INFECTED: (1, 0.647, 0, 1),
                InfectionSummary.CRITICAL: (1, 0, 0, 1),
                InfectionSummary.DEAD: (0.392, 0.392, 0.392, 1),
                InfectionSummary.RECOVERED: (0, 0, 1, 1)
                }
inf_colors = [inf_to_color[summ] for summ in sorted_infection_summary]


def get_stage_strategy(stage_data: np.ndarray) -> Sequence[StageSchedule]:
    curr_stage = stage_data[0]
    ssched = []
    for day, stage in enumerate(stage_data):
        if stage != curr_stage:
            ssched.append(StageSchedule(int(curr_stage), end_day=day))
            curr_stage = stage
    ssched.append(StageSchedule(int(curr_stage), end_day=None))
    return ssched


def plot_annotate_stages(exp_result: ExperimentResult, ax: Axes) -> None:
    xmin = 0
    max_len = exp_result.obs_trajectories.global_infection_summary.shape[0]
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], 1.5 * ylim[1]])

    for strategy in get_stage_strategy(exp_result.obs_trajectories.stage[:, 0, 0]):
        xmax = max_len if strategy.end_day is None else min(strategy.end_day, max_len)
        y = 0.82 * ax.get_ylim()[1]
        ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
                    arrowprops={'arrowstyle': '<->', 'color': 'red', 'linewidth': 1})

        xcenter = xmin + (xmax - xmin) / 2

        ax.annotate(f'{strategy.stage}', xy=(xcenter, 0.92 * ax.get_ylim()[1]), ha='center', va='center', fontsize=8)
        xmin = xmax


def plot_global_infection_summary(exp_result: ExperimentResult,
                                  testing_summary: bool = False,
                                  show_variance: bool = True,
                                  annotate_stages: bool = False,
                                  ax: Optional[Axes] = None) -> None:
    """
    Plot global infection summary

    :param exp_result: ExperimentResult instance
    :param testing_summary: set to True to display testing summary
    :param show_variance: set to True to show variance along n-dim
    :param annotate_stages: set to True to show annotations regarding stages
    :param ax: figure axis handle
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    gis = (exp_result.obs_trajectories.global_infection_summary if not testing_summary
           else exp_result.obs_trajectories.global_testing_summary)
    num_seeds = gis.shape[1]

    title = 'Global Infection Summary' if not testing_summary else 'Global Testing Summary'
    ax.set_prop_cycle(cycler(color=inf_colors))
    if num_seeds > 1:
        gis_mean = gis.mean(1)
        ax.plot(gis_mean, label='gis')
        if show_variance:
            gis_std = gis.std(1)
            for i in range(gis.shape[-1]):
                ax.fill_between(np.arange(len(gis_mean)),
                                np.clip(gis_mean[..., i] - gis_std[..., i], 0, np.inf),
                                gis_mean[..., i] + gis_std[..., i],
                                alpha=0.2)
        title += f'\n ({num_seeds} trials)'
    else:
        ax.plot(gis[:, 0, :])

    if annotate_stages:
        plot_annotate_stages(exp_result, ax)

    ax.set_title(title)
    ax.set_xlabel('time (days)')
    ax.set_ylabel('persons')


def plot_critical_summary(exp_result: ExperimentResult,
                          max_hospital_capacity: int,
                          show_variance: bool = True,
                          annotate_stages: bool = False,
                          ax: Optional[Axes] = None) -> None:
    """
    Plot critical summary

    :param exp_result: ExperimentResult instance
    :param max_hospital_capacity: max hospital capacity
    :param show_variance: set to True to show variance along n-dim
    :param annotate_stages: set to True to show annotations regarding stages
    :param ax: figure axis handle
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    critical = exp_result.obs_trajectories.global_infection_summary[..., critical_index]

    num_seeds = critical.shape[1]
    title = 'Critical Summary'
    if num_seeds > 1:
        critical_mean = critical.mean(1)
        ax.plot(critical_mean, label='critical', c=inf_colors[critical_index])
        if show_variance:
            critical_std = critical.std(1)
            ax.fill_between(np.arange(len(critical_mean)),
                            np.clip(critical_mean - critical_std, 0, np.inf),
                            critical_mean + critical_std,
                            alpha=0.2)
        title += f'\n({num_seeds} trials)'
    else:
        ax.plot(critical[:, 0])

    ax.plot(np.arange(critical.shape[0]), np.ones(critical.shape[0]) * max_hospital_capacity, 'y', label='max_cap')
    ax.set_ylim([-0.1, max_hospital_capacity * 3])

    if annotate_stages:
        plot_annotate_stages(exp_result, ax)

    ax.set_title(title)
    ax.set_xlabel('time (days)')
    ax.set_ylabel('persons')


def plot_deaths_per_day_summary(exp_result: ExperimentResult,
                                show_variance: bool = True,
                                annotate_stages: bool = False,
                                ax: Optional[Axes] = None) -> None:
    """
    Plot deaths per day

    :param exp_result: ExperimentResult instance
    :param show_variance: set to True to show variance along n-dim
    :param annotate_stages: set to True to show annotations regarding stages
    :param ax: figure axis handle
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    deaths = exp_result.obs_trajectories.global_infection_summary[..., deaths_index]
    deaths_per_day = deaths[1:] - deaths[:-1]

    num_seeds = deaths.shape[1]
    title = 'Deaths Per Day'
    if num_seeds > 1:
        deaths_per_day_mean = np.insert(deaths_per_day.mean(1), 0, 0)
        ax.plot(deaths_per_day_mean, label='deaths', c=inf_colors[deaths_index])
        if show_variance:
            deaths_per_day_std = np.insert(deaths_per_day.std(1), 0, 0)
            ax.fill_between(np.arange(len(deaths_per_day_mean)),
                            np.clip(deaths_per_day_mean - deaths_per_day_std, 0, np.inf),
                            deaths_per_day_mean + deaths_per_day_std,
                            alpha=0.2)
        title += f'\n({num_seeds} trials)'
    else:
        ax.plot(np.insert(deaths_per_day[:, 0], 0, 0))

    if annotate_stages:
        plot_annotate_stages(exp_result, ax)

    ax.set_title(title)
    ax.set_xlabel('time (days)')
    ax.set_ylabel('persons')


def plot_cumulative_reward(exp_result: ExperimentResult,
                           show_variance: bool = True,
                           annotate_stages: bool = False,
                           ax: Optional[Axes] = None) -> None:
    """
    Plot cumulative reward

    :param exp_result: ExperimentResult instance
    :param show_variance: set to True to show variance along n-dim
    :param annotate_stages: set to True to add stage annotation
    :param ax: figure axis handle
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    crewards = np.cumsum(exp_result.reward_trajectories, axis=0)
    num_seeds = crewards.shape[1]
    title = 'Cumulative Reward'
    if num_seeds > 1:
        crewards_mean = crewards.mean(1).squeeze()
        ax.plot(crewards_mean, label='cum_reward')
        if show_variance:
            crewards_std = crewards.std(1).squeeze()
            ax.fill_between(np.arange(len(crewards)),
                            crewards_mean - crewards_std,
                            np.clip(crewards_mean + crewards_std, -np.inf, 0),
                            alpha=0.2)
        title += f'\n({num_seeds} trials)'
    else:
        ax.plot(crewards[:, 0])

    if annotate_stages:
        plot_annotate_stages(exp_result, ax)

    ax.set_title('Cumulative Reward')
    ax.set_xlabel('time (days)')


def _get_ylims(avgs: List[Sequence[float]],
               stds: List[Sequence[float]],
               ylim_max_offset: float = 0.) -> Tuple[float, float]:
    ylim_min = np.inf
    ylim_max = 0
    for avg, std in zip(avgs, stds):
        if ylim_min > np.min(np.array(avg) - np.array(std) * 1.3):
            ylim_min = np.min(np.array(avg) - np.array(std) * 1.3)
        if ylim_max < np.max(np.array(avg) + np.array(std) * 1.3):
            ylim_max = np.max(np.array(avg) + np.array(std) * 1.3)
    ylim_max += (ylim_max - ylim_min) * ylim_max_offset
    if ylim_max > 0:
        ylim_min = max(ylim_min, 0)
    return ylim_min, ylim_max


def _get_yerr(avg: Sequence[float], std: Sequence[float]) -> Union[Sequence[float],
                                                                   Tuple[Sequence[float], Sequence[float]]]:
    if np.min(avg) > 0:
        neg_offshoot = np.clip(np.array(avg) - np.array(std), -np.inf, 0)
        if np.min(neg_offshoot) >= 0:
            return std
        else:
            return np.array(std) + neg_offshoot, std
    else:
        pos_offshoot = np.clip(np.array(avg) + np.array(std), 0, np.inf)
        if np.max(pos_offshoot) >= 0:
            return std
        else:
            return np.array(std) - pos_offshoot, std


def _get_t_scores(avg: List[float], std: List[float]) -> List[float]:
    t_scores = []
    for i, (m1, s1) in enumerate(zip(avg, std)):
        for m2, s2 in zip(avg[i + 1:], std[i + 1:]):
            t_scores.append(np.sqrt(len(avg)) * np.abs(m2 - m1) / np.sqrt(s1 ** 2 + s2 ** 2))
    return t_scores


def plot_multi_params_summary(exp_results: Sequence[ExperimentResult],
                              param_labels: Sequence[str],
                              max_hospitals_capacities: Sequence[int],
                              show_testing_diff_plot: bool = False,
                              show_time_to_peak: bool = True,
                              show_cumulative_reward_plot: bool = False,
                              show_pandemic_duration: bool = True,
                              xlabel: str = '',
                              axs: Optional[Sequence[Axes]] = None) -> None:
    """
    Plot multi params summary as bars

    :param exp_results: A sequence of ExperimentResult instances, one for each param
    :param param_labels: A sequence of param labels
    :param max_hospitals_capacities: A sequence of max hospital capacities for each exp result
    :param show_testing_diff_plot: set to True to add a bar to denote the norm diff of testing summary from
        true infection summary
    :param show_time_to_peak: set to True to add a bar to show time to peak for infection, critical and deaths.
    :param show_cumulative_reward_plot: set to True to add a bar plot for the cumulative rewards
    :param show_pandemic_duration: set to True to add a bar plot showing pandemic duration
    :param xlabel: xlabel for all plots
    :param axs: A sequence of figure axis handles
    """
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(8, 6)) if not show_testing_diff_plot else plt.subplots(3, 2,
                                                                                                      figsize=(8, 6))
        axs = list(np.ravel(axs))

    ylabel_size = 8
    assert len(exp_results) == len(param_labels), f'{len(exp_results)}, {len(param_labels)}'
    assert len(exp_results) == len(max_hospitals_capacities)
    assert len(axs) >= 3, f'Expecting at least 4 plot handles, given {len(axs)}'

    infection_peaks_avg = []
    infection_peaks_std = []

    infection_peak_times_avg = []
    infection_peak_times_std = []
    critical_peak_times_avg = []
    critical_peak_times_std = []
    deaths_peak_times_avg = []
    deaths_peak_times_std = []

    total_deaths_avg = []
    total_deaths_std = []
    critical_over_capacity_avg = []
    critical_over_capacity_std = []
    gts_diff_avg = []
    gts_diff_std = []

    pandemic_duration_avg = []
    pandemic_duration_std = []

    cumulative_reward_avg = []
    cumulative_reward_std = []

    for i, (exp_result, max_hospitals_capacity) in enumerate(zip(exp_results, max_hospitals_capacities)):
        gis = exp_result.obs_trajectories.global_infection_summary
        gts = exp_result.obs_trajectories.global_testing_summary
        peaks = gis[..., infection_index].max(axis=0) / exp_result.num_persons

        infection_peak_times = gis[..., infection_index].argmax(axis=0)
        first_infection_day = (gis[..., infection_index] >= 1).argmin(axis=0)
        last_infection_day = (gis[..., infection_index] >= 1).argmax(axis=0)

        critical_peak_times = gis[..., critical_index].argmax(axis=0)
        first_critical_day = (gis[..., critical_index] >= 1).argmin(axis=0)
        last_critical_day = (gis[..., critical_index] >= 1).argmax(axis=0)

        deaths_per_day = gis[1:, :, deaths_index] - gis[:-1, :, deaths_index]
        deaths_peak_times = deaths_per_day.argmax(axis=0) + 1  # add 1 to account for diff on first day
        first_death_day = (gis[..., deaths_index] >= 1).argmin(axis=0)

        infection_peaks_avg.append(peaks.mean())
        infection_peaks_std.append(peaks.std())

        infection_peak_times_avg.append((infection_peak_times - first_infection_day).mean())
        infection_peak_times_std.append((infection_peak_times - first_infection_day).std())

        critical_peak_times_avg.append((critical_peak_times - first_critical_day).mean())
        critical_peak_times_std.append((critical_peak_times - first_critical_day).std())

        deaths_peak_times_avg.append((deaths_peak_times - first_death_day).mean())
        deaths_peak_times_std.append((deaths_peak_times - first_death_day).std())

        deaths = gis[..., deaths_index].max(axis=0) / exp_result.num_persons
        total_deaths_avg.append(deaths.mean())
        total_deaths_std.append(deaths.std())

        critical = exp_result.obs_trajectories.global_infection_summary[..., critical_index]
        critical_above_cap = np.clip((critical - max_hospitals_capacity) / max_hospitals_capacity,
                                     0, np.inf).sum(axis=0)
        critical_over_capacity_avg.append(critical_above_cap.mean())
        critical_over_capacity_std.append(critical_above_cap.std())

        gts_diff = np.mean(np.linalg.norm(gts - gis, axis=-1), axis=0)
        gts_diff_avg.append(gts_diff.mean())
        gts_diff_std.append(gts_diff.std())

        pandemic_duration = (np.max((last_infection_day, last_critical_day), axis=0) -
                             np.min((first_infection_day, first_critical_day), axis=0))
        pandemic_duration_avg.append(pandemic_duration.mean())
        pandemic_duration_std.append(pandemic_duration.std())

        cumulative_reward = np.cumsum(exp_result.reward_trajectories, axis=0)[-1]
        cumulative_reward_avg.append(cumulative_reward.mean())
        cumulative_reward_std.append(cumulative_reward.std())

    index = np.arange(len(exp_results))
    bar_width = 0.35
    opacity = 0.4
    xticks_fontsize = 8
    xticks_rotation = 20

    axs_2d = []
    axs_3d = []

    for ax in axs:
        if isinstance(ax, Axes3D):
            axs_3d.append(ax)
        else:
            axs_2d.append(ax)

    plot_i_2d = 0
    plot_i_3d = 0

    axs_2d[plot_i_2d].bar(index, infection_peaks_avg, bar_width, alpha=opacity, color='c',
                          yerr=_get_yerr(infection_peaks_avg, infection_peaks_std), error_kw={'elinewidth': 1.0})
    axs_2d[plot_i_2d].set_xlabel(xlabel)
    axs_2d[plot_i_2d].set_ylabel('persons / population size', fontsize=ylabel_size)
    axs_2d[plot_i_2d].set_title('Infection Peak (normalized)')
    axs_2d[plot_i_2d].set_xticks(index)
    axs_2d[plot_i_2d].set_xticklabels(param_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)
    axs_2d[plot_i_2d].set_ylim(_get_ylims([infection_peaks_avg], [infection_peaks_std]))

    plot_i_2d += 1
    axs_2d[plot_i_2d].bar(index, critical_over_capacity_avg, bar_width, alpha=opacity, color='y',
                          yerr=_get_yerr(critical_over_capacity_avg, critical_over_capacity_std),
                          error_kw={'elinewidth': 1.0})
    axs_2d[plot_i_2d].set_xlabel(xlabel)
    axs_2d[plot_i_2d].set_ylabel('persons x days / max capacity', fontsize=ylabel_size)
    axs_2d[plot_i_2d].set_title('Critical (> max capacity)')
    axs_2d[plot_i_2d].set_xticks(index)
    axs_2d[plot_i_2d].set_xticklabels(param_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)
    axs_2d[plot_i_2d].set_ylim(_get_ylims([critical_over_capacity_avg], [critical_over_capacity_std]))

    plot_i_2d += 1
    axs_2d[plot_i_2d].bar(index, total_deaths_avg, bar_width, alpha=opacity, color='k',
                          yerr=_get_yerr(total_deaths_avg, total_deaths_std), error_kw={'elinewidth': 1.0})
    axs_2d[plot_i_2d].set_xlabel(xlabel)
    axs_2d[plot_i_2d].set_ylabel('persons / population size', fontsize=ylabel_size)
    axs_2d[plot_i_2d].set_title('Deaths (normalized)')
    axs_2d[plot_i_2d].set_xticks(index)
    axs_2d[plot_i_2d].set_xticklabels(param_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)
    axs_2d[plot_i_2d].set_ylim(_get_ylims([total_deaths_avg], [total_deaths_std]))

    if show_time_to_peak:
        plot_i_2d += 1
        bar_width = 0.2
        axs_2d[plot_i_2d].bar(index - bar_width, infection_peak_times_avg, bar_width, alpha=opacity, color='r',
                              yerr=_get_yerr(infection_peak_times_avg, infection_peak_times_std),
                              error_kw={'elinewidth': 1.0})
        axs_2d[plot_i_2d].bar(index, critical_peak_times_avg, bar_width, alpha=opacity, color='g',
                              yerr=_get_yerr(critical_peak_times_avg, critical_peak_times_std),
                              error_kw={'elinewidth': 1.0})
        axs_2d[plot_i_2d].bar(index + bar_width, deaths_peak_times_avg, bar_width, alpha=opacity, color='b',
                              yerr=_get_yerr(deaths_peak_times_avg, deaths_peak_times_std),
                              error_kw={'elinewidth': 1.0})
        axs_2d[plot_i_2d].set_xlabel(xlabel)
        axs_2d[plot_i_2d].set_ylabel('days', fontsize=ylabel_size)
        axs_2d[plot_i_2d].set_title('Time to Peak')
        axs_2d[plot_i_2d].legend(['Infection', 'Critical', 'Death'], loc='upper right', ncol=3, fontsize=6)
        axs_2d[plot_i_2d].set_ylim(
            _get_ylims([infection_peak_times_avg, critical_peak_times_avg, deaths_peak_times_avg],
                       [infection_peak_times_std, critical_peak_times_std, deaths_peak_times_std],
                       ylim_max_offset=0.2))
        axs_2d[plot_i_2d].set_xticks(index)
        axs_2d[plot_i_2d].set_xticklabels(param_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)

    if show_pandemic_duration:
        plot_i_2d += 1
        axs_2d[plot_i_2d].bar(index, pandemic_duration_avg, bar_width, alpha=opacity, color='r',
                              yerr=_get_yerr(pandemic_duration_avg, pandemic_duration_std),
                              error_kw={'elinewidth': 1.0})
        axs_2d[plot_i_2d].set_xlabel(xlabel)
        axs_2d[plot_i_2d].set_ylabel('days', fontsize=ylabel_size)
        axs_2d[plot_i_2d].set_title('Pandemic Duration')
        axs_2d[plot_i_2d].set_xticks(index)
        axs_2d[plot_i_2d].set_xticklabels(param_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)
        axs_2d[plot_i_2d].set_ylim(_get_ylims([pandemic_duration_avg], [pandemic_duration_std]))

    if show_testing_diff_plot:
        plot_i_2d += 1
        axs_2d[plot_i_2d].bar(index, gts_diff_avg, bar_width, alpha=opacity, color='g',
                              yerr=_get_yerr(gts_diff_avg, gts_diff_std), error_kw={'elinewidth': 1.0})
        axs_2d[plot_i_2d].set_xlabel(xlabel)
        axs_2d[plot_i_2d].set_ylabel('', fontsize=ylabel_size)
        axs_2d[plot_i_2d].set_title('Testing/Infection\nSummary Difference')
        axs_2d[plot_i_2d].set_xticks(index)
        axs_2d[plot_i_2d].set_xticklabels(param_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)

    if show_cumulative_reward_plot:
        plot_i_2d += 1
        axs_2d[plot_i_2d].bar(index, cumulative_reward_avg, bar_width, alpha=opacity, color='r',
                              yerr=_get_yerr(cumulative_reward_avg, cumulative_reward_std),
                              error_kw={'elinewidth': 1.0})
        axs_2d[plot_i_2d].set_xlabel(xlabel)
        axs_2d[plot_i_2d].set_ylabel('days', fontsize=ylabel_size)
        axs_2d[plot_i_2d].set_title('Cumulative Reward')
        axs_2d[plot_i_2d].set_xticks(index)
        axs_2d[plot_i_2d].set_xticklabels(param_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)
        # axs_2d[plot_i_2d].set_ylim(_get_ylims([cumulative_reward_avg], [cumulative_reward_std]))

    for i in range(plot_i_2d + 1, len(axs_2d)):
        axs_2d[i].axis('off')

    for i in range(plot_i_3d + 1, len(axs_3d)):
        axs_3d[i].axis('off')
