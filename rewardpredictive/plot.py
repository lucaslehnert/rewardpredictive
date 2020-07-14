#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
import os.path as osp
from itertools import product

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import scipy
from scipy.stats import sem
import numpy as np
import rlutils as rl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon

from .cycle_mdp_dataset import load_cycle_mdp_dataset

get_avg_ep_len = lambda exp: np.mean(np.mean(exp.episode_length, axis=-1), axis=-1)


def transfer_episode_length(ep_len, color='C0', label=None):
    num_eps = np.shape(ep_len)[2]
    for task_idx in range(np.shape(ep_len)[1]):
        ep_idx = np.arange(num_eps) + task_idx * num_eps
        rl.plot.mean_with_sem(ep_idx, ep_len[:, task_idx, :], axis=0, color=color, label=label)
        label = None


def plot_lr_comparison_qlearning(experiment_list):  # pragma: no cover
    lr_avg_ep_len = [(e.learning_rate, get_avg_ep_len(e)) for e in experiment_list]
    lr_avg_ep_len = sorted(lr_avg_ep_len, key=lambda e: e[0])

    plt.violinplot([e[1] for e in lr_avg_ep_len])
    plt.xticks(range(1, len(lr_avg_ep_len) + 1), [e[0] for e in lr_avg_ep_len])
    plt.xlabel('Learning Rate')
    plt.ylabel('Avg. episode length')


def plot_lr_comparison_sflearning(experiment_list):  # pragma: no cover
    lr_ep_len = [(e.learning_rate_sf, e.learning_rate_reward, get_avg_ep_len(e)) for e in experiment_list]
    lr_ep_len = sorted(lr_ep_len, key=lambda e: (e[1], e[0]))

    plt.figure(figsize=(15, 6))
    plt.violinplot([e[2] for e in lr_ep_len])
    xticks = range(1, len(lr_ep_len) + 1)
    xticks_lab = [r'lr\_sf={}, lr\_r={}'.format(e[0], e[1]) for e in lr_ep_len]
    plt.xticks(xticks, xticks_lab)
    plt.xlabel('Learning Rate')
    plt.ylabel('Avg. episode length')


def plot_lr_comparison_dirichlet_process_model(experiment_list):  # pragma: no cover
    lr_a_b_avg_ep_len = [(e.learning_rate, e.alpha, e.beta, get_avg_ep_len(e)) for e in experiment_list]
    lr_a_b_avg_ep_len = sorted(lr_a_b_avg_ep_len, key=lambda e: (e[2], e[1], e[0]))

    plt.figure(figsize=(15, 6))
    for i, lr in enumerate([0.1, 0.5, 0.9]):
        ep_len = [e[3] for e in filter(lambda e: e[0] == 0.1, lr_a_b_avg_ep_len)]
        plt.violinplot(ep_len, positions=np.arange(len(ep_len)) * 3 + i)

    xticks_labels = [r'lr={}, $\alpha$={}, $\beta$={}'.format(e[0], e[1], e[2]) for e in lr_a_b_avg_ep_len]
    _ = plt.xticks(range(len(xticks_labels)), xticks_labels, rotation=90)
    _ = plt.ylabel('Avg. episode length')


def plot_lr_sf_lr_rew_comparison_dirichlet_process_model(experiment_list):  # pragma: no cover
    param_ep_len = []
    for e in experiment_list:
        param_ep_len.append((e.learning_rate_sf,
                             e.learning_rate_reward,
                             e.alpha,
                             e.beta,
                             get_avg_ep_len(e)))
    param_ep_len = sorted(param_ep_len, key=lambda e: (e[3], e[2], e[0], e[1]))

    lr_sf_list = [p[0] for p in param_ep_len]
    lr_rew_list = [p[1] for p in param_ep_len]

    plt.figure(figsize=(15, 6))
    for i, (lr_sf, lr_rew) in enumerate(product(lr_sf_list, lr_rew_list)):
        param_filtered = filter(lambda e: e[0] == lr_sf and e[1] == lr_rew, param_ep_len)
        ep_len = [e[4] for e in param_filtered]
        plt.violinplot(ep_len, positions=np.arange(len(ep_len)) * len(lr_sf_list) * len(lr_rew_list) + i)

    xticks_labels = []
    for e in param_ep_len:
        xticks_labels.append(
            r'lr_sf={}, lr_rew={}, $\alpha$={}, $\beta$={}'.format(e[0], e[1], e[2], e[3])
        )
    _ = plt.xticks(range(len(xticks_labels)), xticks_labels, rotation=90)
    _ = plt.ylabel('Avg. episode length')


def plot_alpha_vs_belief_space_size(experiment_set, hparam_alpha_beta,
                                    color_list=None,
                                    legend=False,
                                    figsize=(2.5, 2)):  # pragma: no cover
    plt.figure(figsize=figsize)
    alpha_values = experiment_set.get_hparam_values('alpha')
    beta_values = experiment_set.get_hparam_values('beta')
    if color_list is None:
        color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    posterior_size = lambda e: [np.shape(cnt)[-1] for cnt in e.results['count']]
    exp_getter = lambda a, b: experiment_set.get_experiment_list_by_hparam(hparam_alpha_beta(a, b))[0]

    for beta, c in zip(beta_values, color_list):
        exp_alpha = [exp_getter(alpha, beta) for alpha in alpha_values]
        belief_size_list = np.stack([posterior_size(exp) for exp in exp_alpha])
        if beta == np.inf:
            beta = r'$\infty$'
        elif beta == int(beta):
            beta = int(beta)
        rl.plot.mean_with_sem(alpha_values, belief_size_list, axis=1, color=c, label=r'$\beta$={}'.format(beta))

    plt.gca().set_xscale('log')
    plt.xlabel(r'$\alpha$ Value')
    plt.ylabel('Avg. belief space size')
    plt.ylim([0, 9.6])
    plt.yticks([1, 2, 3, 4, 5, 6], ['{:4d}'.format(i) for i in [1, 2, 3, 4, 5, 6]])
    plt.xticks(alpha_values)
    if legend:
        plt.legend(frameon=False)


def plot_cycle_mdp_belief_space_size(experiment_set, hparam_alpha_beta):
    plot_alpha_vs_belief_space_size(experiment_set, hparam_alpha_beta, legend=True, figsize=(1.8, 2))
    yticks = list(range(0, 21, 2))
    plt.yticks(yticks, ['{:2d}'.format(i) for i in yticks])
    plt.xticks([1e-3, 1e0, 1e3])
    plt.ylim([0, 21])


def plot_alpha_vs_episode_length(experiment_set, hparam_alpha_beta, color_list=None, legend=False):  # pragma: no cover
    plt.figure(figsize=(2.5, 2))
    alpha_values = experiment_set.get_hparam_values('alpha')
    beta_values = experiment_set.get_hparam_values('beta')
    if color_list is None:
        color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    avg_ep_len = lambda exp: np.mean(np.mean(exp.results['episode_length'], axis=-1), axis=-1)
    exp_getter = lambda a, b: experiment_set.get_experiment_list_by_hparam(hparam_alpha_beta(a, b))[0]

    for beta, c in zip(beta_values, color_list):
        exp_alpha = [exp_getter(alpha, beta) for alpha in alpha_values]
        ep_len_list = np.stack([avg_ep_len(exp) for exp in exp_alpha])
        if beta == np.inf:
            beta = r'$\infty$'
        elif beta == int(beta):
            beta = int(beta)
        rl.plot.mean_with_sem(alpha_values, ep_len_list, axis=1, color=c, label=r'$\beta$={}'.format(beta))

    plt.gca().set_xscale('log')
    plt.xlabel(r'$\alpha$ Value')
    plt.ylabel('Avg. Belief Space Size')
    plt.xticks(alpha_values)
    plt.ylim([100, 400])
    if legend:
        plt.legend(frameon=False)


def plot_alpha_vs_total_reward(experiment_set, hparam_alpha_beta, color_list=None, legend=False,
                               figsize=(1.8, 2)):  # pragma: no cover
    plt.figure(figsize=figsize)
    alpha_values = experiment_set.get_hparam_values('alpha')
    beta_values = experiment_set.get_hparam_values('beta')
    if color_list is None:
        color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    avg_ep_len = lambda exp: np.mean(exp.results['total_reward'], axis=-1)
    exp_getter = lambda a, b: experiment_set.get_experiment_list_by_hparam(hparam_alpha_beta(a, b))[0]

    for beta, c in zip(beta_values, color_list):
        exp_alpha = [exp_getter(alpha, beta) for alpha in alpha_values]
        ep_len_list = np.stack([avg_ep_len(exp) for exp in exp_alpha])
        if beta == np.inf:
            beta = r'$\infty$'
        elif beta == int(beta):
            beta = int(beta)
        rl.plot.mean_with_sem(alpha_values, ep_len_list, axis=1, color=c, label=r'$\beta$={}'.format(beta))

    plt.gca().set_xscale('log')
    plt.xlabel(r'$\alpha$ Value')
    plt.ylabel('Avg. Total Reward')
    plt.xticks([alpha_values[0], 1., alpha_values[-1]])
    plt.ylim([1.8, 3.4])
    plt.yticks([2, 3])
    if legend:
        plt.legend(frameon=False)



def plot_avg_highest_count(experiment_set, hparam_alpha_beta, figsize=(3, 2)):
    plt.figure(figsize=figsize)
    partition_idx_seq_list = load_cycle_mdp_dataset()[1]
    gt_abs_count = np.sum(partition_idx_seq_list == 0, axis=-1)
    gt_abs_count_m = np.mean(gt_abs_count, axis=-1)
    gt_abs_count_e = sem(gt_abs_count, axis=-1)
    plt.fill_between(
        [-1, len(experiment_set.experiment_list) + 1],
        y1=gt_abs_count_m+gt_abs_count_e,
        y2=gt_abs_count_m-gt_abs_count_e,
        color='k',
        alpha=0.2
    )
    plt.bar(
        [0],
        [gt_abs_count_m],
        yerr=[gt_abs_count_e],
        color='w',
        edgecolor='k',
        label='Ground Truth\n(G.T.)'
    )

    alpha_values = experiment_set.get_hparam_values('alpha')
    beta_values = experiment_set.get_hparam_values('beta')
    get_exp = lambda a, b: experiment_set.get_experiment_list_by_hparam(hparam_alpha_beta(a, b))[0]
    get_counts = lambda e: [np.max(c[-1]) for c in e.results['count']]
    for i, (beta, color) in enumerate(zip(beta_values, ['C0', 'C1', 'C2'])):
        counts = np.stack([get_counts(get_exp(alpha, beta)) for alpha in alpha_values])
        counts_m = np.mean(counts, axis=-1)
        counts_e = sem(counts, axis=-1)
        xvalues = np.arange(len(alpha_values))
        xvalues += i * len(alpha_values) + 1
        if beta == np.inf:
            beta_str = r'$\infty$'
        elif beta == int(beta):
            beta_str = '{}'.format(int(beta))
        else:
            beta_str = '{}'.format(beta)
        plt.bar(xvalues, counts_m, yerr=counts_e, color=color, label=r'$\beta$={}'.format(beta_str))
    alpha_val_str = ['{}'.format(a) for a in alpha_values]
    alpha_val_str = ['G. T.'] + alpha_val_str + alpha_val_str + alpha_val_str
    plt.xticks(range(len(alpha_val_str)), alpha_val_str, rotation='vertical')
    plt.xlabel(r'$\alpha$ Value')
    plt.xlim([-.8, len(experiment_set.experiment_list) + .8])
    plt.ylabel('Average Highest Count')
    plt.yticks([0, 5, 10, 15, 20])
    plt.ylim([0, 27])
    plt.legend(loc=9, ncol=4, frameon=False, handlelength=.8, handletextpad=.4, columnspacing=1.0)


def plot_convergence_rate_comparison(ep_len_list, color_list, label_list, figsize=(10, 5)):  # pragma: no cover
    f, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 5]}
    )

    plt.sca(ax1)
    for ep_len, c, l in zip(ep_len_list, color_list, label_list):
        transfer_episode_length(ep_len, color=c, label=l)
    plt.yticks([1500, 2000], ['1500', '2000\n(Timeout)'])
    plt.ylim([1400, 2100])
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])
    ax1.xaxis.tick_top()

    plt.sca(ax2)
    for ep_len, c, l in zip(ep_len_list, color_list, label_list):
        transfer_episode_length(ep_len, color=c, label=l)
    ax2.set_ylim([0, 310])
    ax2.spines['top'].set_visible(False)
    ax2.set_xticks([1, 200, 400, 600, 800, 1000])
    plt.xlabel('Episode')

    plt.ylabel('Avg. episode length')
    plt.legend()


def process_and_pad_posterior(ar, l):
    if len(ar) > 1:
        ar = list(ar[-1:]) + list(ar[:-1])
    return np.array(list(ar) + [np.nan] * (l - len(ar)), dtype=np.float32)[::-1]


# prefix_nan_pad = lambda ar, l: np.array([np.nan] * (l - len(ar)) + list(ar[:-1][::-1]) + list(ar[-1]))

def plot_belief_posterior(experiment, repeat=0, belief_size=None):  # pragma: no cover
    import matplotlib
    import matplotlib.colors as colors
    if belief_size is None:
        belief_size = np.max([len(p) for p in experiment.posterior_episode_log])
    posterior_padded = np.stack([process_and_pad_posterior(p, belief_size) for p in experiment.posterior_episode_log])
    posterior_padded = posterior_padded[repeat * 1000:(repeat + 1) * 1000].transpose()
    posterior_padded = posterior_padded[::-1]

    #     cmap = matplotlib.cm.winter
    cmap = colors.LinearSegmentedColormap.from_list(
        'custom_cmap',
        matplotlib.cm.winter(np.linspace(.5, 1, 5))
    )
    cmap.set_bad('white', 1.)

    numrows, numcols = np.shape(posterior_padded)
    plt.matshow(
        posterior_padded,
        cmap=cmap,
        aspect='auto',
        origin='upper',
        extent=(0.5, numcols + 0.5, numrows + 0.5, +0.5),
        fignum=1
    )
    cbar = plt.colorbar(aspect=8, pad=0.01)
    cbar.set_label('Probability')
    plt.clim([0, 1])

    _ = plt.yticks(range(1, belief_size + 1), ['Identity'] + ['Rep. {}'.format(i) for i in range(1, belief_size)])
    _ = plt.gca().xaxis.set_ticks_position("bottom")


def plot_maze_a_background():  # pragma: no cover
    plt.gca().add_collection(PatchCollection([Rectangle([-.5, -.5], 1., 1.)], facecolor='C0'))
    plt.gca().add_collection(PatchCollection([Rectangle([8.5, -.5], 1., 1.)], facecolor='C2'))

    for i in range(9):
        plt.plot([i + .5, i + .5], [-5., 9.5], ':k', alpha=0.5)
        plt.plot([-5., 9.5], [i + .5, i + .5], ':k', alpha=0.5)

    plt.plot([1.5, 1.5], [-.5, 7.5], 'k', linewidth=3)
    plt.plot([3.5, 3.5], [1.5, 9.5], 'k', linewidth=3)
    plt.plot([5.5, 5.5], [-.5, 7.5], 'k', linewidth=3)
    plt.plot([7.5, 7.5], [1.5, 9.5], 'k', linewidth=3)


def plot_maze_b_background():  # pragma: no cover
    plt.gca().add_collection(PatchCollection([Rectangle([-.5, 8.5], 1., 1.)], facecolor='C0'))
    plt.gca().add_collection(PatchCollection([Rectangle([8.5, 8.5], 1., 1.)], facecolor='C2'))

    for i in range(9):
        plt.plot([i + .5, i + .5], [-5., 9.5], ':k', alpha=0.5)
        plt.plot([-5., 9.5], [i + .5, i + .5], ':k', alpha=0.5)

    plt.plot([1.5, 1.5], [1.5, 9.5], 'k', linewidth=3)
    plt.plot([3.5, 3.5], [-.5, 7.5], 'k', linewidth=3)
    plt.plot([5.5, 5.5], [1.5, 9.5], 'k', linewidth=3)
    plt.plot([7.5, 7.5], [-.5, 7.5], 'k', linewidth=3)


def plot_maze_a_decoration():  # pragma: no cover
    plt.text(9.65, 0.05, 'Goal\nState', fontsize=24, horizontalalignment='left', verticalalignment='center')
    plt.text(-.65, -.05, 'Start\nState', fontsize=23, horizontalalignment='right', verticalalignment='center')

    plt.ylim([-.5, 9.5])
    plt.xlim([-.5, 9.5])
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')


def plot_maze_b_decoration():  # pragma: no cover
    # plt.text(9.05, 9.05, '+1', fontsize=9, horizontalalignment='center', verticalalignment='center')
    plt.text(9.65, 9.05, 'Goal\nState', fontsize=24, horizontalalignment='left', verticalalignment='center')
    plt.text(-.65, 9.05, 'Start\nState', fontsize=24, horizontalalignment='right', verticalalignment='center')

    plt.ylim([-.5, 9.5])
    plt.xlim([-.5, 9.5])
    _ = plt.xticks([])
    _ = plt.yticks([])
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')


def plot_double_state_space_abstraction():  # pragma: no cover
    plt.gca().add_collection(
        PatchCollection(
            [Polygon([[7, -.5], [9.5, -.5], [9.5, 9.5], [3, 9.5]])],
            facecolor='k',
            alpha=0.2
        )
    )


def plot_half_state_space_abstraction():  # pragma: no cover
    plt.gca().add_collection(
        PatchCollection(
            [Polygon([[4.5, -.5], [9.5, -.5], [9.5, 1], [4.5, 4]])],
            facecolor='r',
            alpha=0.2
        )
    )
    plt.gca().add_collection(
        PatchCollection(
            [Polygon([[4.5, 4], [9.5, 1], [9.5, 5], [4.5, 8]])],
            facecolor='g',
            alpha=0.2
        )
    )
    plt.gca().add_collection(
        PatchCollection(
            [Polygon([[4.5, 8], [9.5, 5], [9.5, 9.5], [4.5, 9.5]])],
            facecolor='b',
            alpha=0.2
        )
    )


def plot_maze_task_ep_len(experiment_list,
                          label_list,
                          color_list,
                          show_legend=True,
                          show_ylabel=True,
                          figsize=None):  # pragma: no cover
    if figsize is not None:
        plt.figure(figsize=figsize)

    step_list = [np.mean(exp.results['episode_length'], axis=-1) for exp in experiment_list]
    xticks = [1, 2, 3, 4, 5]
    for steps, label, color in zip(step_list, label_list, color_list):
        rl.plot.mean_with_sem(xticks, steps, axis=0, color=color, label=label)

    if show_legend:
        plt.legend(
            loc=1,
            ncol=2,
            frameon=False,
            columnspacing=1.0,
            handlelength=1.0,
            handletextpad=0.4
        )
    if show_ylabel:
        plt.ylabel('Avg. episode length per task')
    plt.xticks(
        xticks,
        [
            'Task 1\nMaze A\nLight Dark',
            'Task 2\nMaze B\nColoured',
            'Task 3\nMaze A\nLight Dark',
            'Task 4\nMaze B\nLight Dark',
            'Task 5\nMaze A\nColoured'
        ]
    )
    plt.ylim([80, 165])


def plot_maze_task_ep_len_broken_yaxis(experiment_list,
                                       label_list,
                                       color_list,
                                       figsize=None):  # pragma: no cover
    if figsize is None:
        figsize = (4, 3)
    f, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': [1, 1]}
    )

    plt.sca(ax1)
    plot_maze_task_ep_len(
        experiment_list=experiment_list,
        label_list=label_list,
        color_list=color_list,
        show_legend=True,
        show_ylabel=False
    )
    plt.ylim([85, 310])
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticks([])
    ax1.xaxis.tick_top()
    ax1.tick_params(axis='x', colors='w')

    plt.sca(ax2)
    plot_maze_task_ep_len(
        experiment_list=experiment_list,
        label_list=label_list,
        color_list=color_list,
        show_legend=False,
        show_ylabel=False
    )
    ax2.set_ylabel('Avg. episode length per task')
    ax2.yaxis.set_label_coords(-.11, 1.07)

    ax2.set_ylim([50, 85])
    ax2.spines['top'].set_visible(False)

    d = .015
    ax1.plot((-d, +d), (-d, +d), transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
    ax1.plot((1 - d, 1 + d), (-d, +d), transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
    ax2.plot((-d, +d), (1 - d, 1 + d), transform=ax2.transAxes, color='k', clip_on=False, linewidth=1)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2.transAxes, color='k', clip_on=False, linewidth=1)


def plot_maze_posterior(posterior):  # pragma: no cover
    plt.figure(figsize=(5, 1.2))

    num_tasks, num_eps, _ = np.shape(posterior)
    posterior = np.reshape(posterior, [num_tasks * num_eps, -1])
    assert (np.shape(posterior)[1] <= 5)
    cmap = colors.LinearSegmentedColormap.from_list(
        'custom_cmap',
        matplotlib.cm.winter(np.linspace(.5, 1, 5))
    )
    cmap.set_bad('white', 1.)
    numrows, numcols = np.shape(posterior.transpose())
    plt.matshow(
        posterior.transpose(),
        cmap=cmap,
        aspect='auto',
        origin='upper',
        extent=(0.5, numcols + 0.5, numrows + 0.5, +0.5),
        fignum=1
    )

    cbar = plt.colorbar(aspect=8, pad=0.01)
    cbar.set_label('Probability')
    plt.clim([0, 1])
    plt.gca().xaxis.set_ticks_position("bottom")

    y_label_list = ['Identity', 'Rep. 1', 'Rep. 2', 'Rep. 3', 'Rep. 4']
    for i in range(np.shape(posterior)[1], 5):
        y_label_list[i] = ''
    plt.yticks([1, 2, 3, 4, 5], y_label_list)
    plt.ylim([.5, 5.5])
    plt.gca().invert_yaxis()

    _ = plt.gca().xaxis.set_ticks([0, 200, 400, 600, 800, 1000], minor=False)
    _ = plt.gca().xaxis.set_ticks([100, 300, 500, 700, 900], minor=True)
    _ = plt.gca().xaxis.set_ticklabels(
        [
            'Task 1\nMaze A\nLight-Dark',
            'Task 2\nMaze B\nColoured',
            'Task 3\nMaze A\nLight-Dark',
            'Task 4\nMaze B\nLight-Dark',
            'Task 5\nMaze A\nColoured'
        ],
        minor=True
    )
    _ = plt.gca().xaxis.set_tick_params(length=0, which='minor', pad=35)
    plt.xlabel('Episode', labelpad=-50)


def plot_histogram_column_world(experiment, num_latent_states=None):  # pragma: no cover
    rl.set_seeds(12345)

    rew_max = get_total_reward_for_reward_maximizing(experiment, num_latent_states=num_latent_states)
    rew_prd = get_total_reward_for_reward_predictive(experiment, num_latent_states=num_latent_states)

    pval = scipy.stats.ttest_ind(rew_max, rew_prd, equal_var=True).pvalue
    if num_latent_states is None:
        print('Column World Welch\'s T-test pvalue: {}'.format(pval))
    else:
        print('Column World Welch\'s T-test pvalue ({} latent states): {}'.format(num_latent_states, pval))

    bins = np.linspace(4, 10, 21)
    plt.hist(rew_max, bins=bins, label='Reward Maximizing', alpha=0.5)
    plt.hist(rew_prd, bins=bins, label='Reward Predictive', alpha=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(0., 1.3), frameon=False)
    _ = plt.xlabel('Total Reward at Transfer')
    _ = plt.ylabel('Number of\nAbstractions')


def plot_and_save_histogram_column_world(plot_dir, experiment):  # pragma: no cover
    plt.figure(figsize=(2.6, 2.4))
    plot_histogram_column_world(experiment)
    plt.tight_layout()
    plt.savefig(osp.join(plot_dir, 'histogram_column_world.pdf'), bbox_inches='tight', pad_inches=.02)
    # plt.close('all')

    plt.figure(figsize=(2.6, 2.4))
    plot_histogram_column_world(experiment, 3)
    # plt.ylim([0, 60])
    plt.tight_layout()
    plt.savefig(osp.join(plot_dir, 'histogram_column_world_latent_3.pdf'), bbox_inches='tight', pad_inches=.02)
    # plt.close('all')


def plot_histogram_goal_and_wall_world(experiment, num_latent_states=None):  # pragma: no cover
    rl.set_seeds(12345)

    rew_max = get_total_reward_for_reward_maximizing(experiment, num_latent_states=num_latent_states)
    rew_prd = get_total_reward_for_reward_predictive(experiment, num_latent_states=num_latent_states)
    pval = scipy.stats.ttest_ind(rew_max, rew_prd, equal_var=True).pvalue
    if num_latent_states is None:
        print('Goal and Wall World Welch\'s T-test pvalue: {}'.format(pval))
    else:
        print('Goal and Wall World Welch\'s T-test pvalue ({} latent states): {}'.format(num_latent_states, pval))

    bins = np.linspace(4, 10, 21)
    plt.hist(rew_max, bins=bins, label='Reward Maximizing', alpha=0.5)
    plt.hist(rew_prd, bins=bins, label='Reward Predictive', alpha=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(0., 1.3), frameon=False)
    _ = plt.xlabel('Total Reward at Transfer')
    _ = plt.ylabel('Number of\nAbstractions')


def plot_and_save_histogram_goal_and_wall_world(plot_dir, experiment):  # pragma: no cover
    plt.figure(figsize=(2.6, 2.4))
    plot_histogram_goal_and_wall_world(experiment)
    plt.tight_layout()
    plt.savefig(osp.join(plot_dir, 'histogram_goal_and_wall_world.pdf'), bbox_inches='tight', pad_inches=.02)
    # plt.close('all')

    plt.figure(figsize=(2.6, 2.4))
    plot_histogram_goal_and_wall_world(experiment, 3)
    # plt.ylim([0, 60])
    plt.tight_layout()
    plt.savefig(osp.join(plot_dir, 'histogram_goal_and_wall_world_latent_3.pdf'), bbox_inches='tight', pad_inches=.02)
    # plt.close('all')


def plot_histogram_rand_mdp(experiment, num_latent_states=None):  # pragma: no cover
    rl.set_seeds(12345)

    rew_max = get_total_reward_for_reward_maximizing(experiment, num_latent_states=num_latent_states)
    rew_prd = get_total_reward_for_reward_predictive(experiment, num_latent_states=num_latent_states)
    pval = scipy.stats.ttest_ind(rew_max, rew_prd, equal_var=True).pvalue
    if num_latent_states is None:
        print('Rand. MDP Welch\'s T-test pvalue: {}'.format(pval))
    else:
        print('Rand. MDP Welch\'s T-test pvalue ({} latent states): {}'.format(num_latent_states, pval))

    bins = np.linspace(1.8, 2.4, 21)
    plt.hist(rew_max, bins=bins, label='Reward Maximizing', alpha=0.5)
    plt.hist(rew_prd, bins=bins, label='Reward Predictive', alpha=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(0., 1.3), frameon=False)
    _ = plt.xlabel('Total Reward at Transfer')
    _ = plt.ylabel('Number of\nAbstractions')


def plot_and_save_histogram_rand_mdp(plot_dir, experiment):  # pragma: no cover
    plt.figure(figsize=(2.6, 2.4))
    plot_histogram_rand_mdp(experiment)
    plt.tight_layout()
    plt.savefig(osp.join(plot_dir, 'histogram_rand_mdp.pdf'), bbox_inches='tight', pad_inches=.02)
    # plt.close('all')

    plt.figure(figsize=(2.6, 2.4))
    plot_histogram_rand_mdp(experiment, 3)
    # plt.ylim([0, 60])
    plt.tight_layout()
    plt.savefig(osp.join(plot_dir, 'histogram_rand_mdp_latent_3.pdf'), bbox_inches='tight', pad_inches=.02)
    # plt.close('all')


def mean_with_sem(xvals, yvals, axis=0, color='C0', linewidth=1, label=None):  # pragma: no cover
    import matplotlib.pyplot as plt  # Must be imported on use, otherwise matplotlib may open a UI window.
    from scipy.stats import sem

    yvals_m = np.mean(yvals, axis=axis)
    yvals_e = sem(yvals, axis=axis)
    plt.plot(xvals, yvals_m, c=color, label=label, linewidth=linewidth)
    plt.fill_between(xvals, y1=yvals_m + yvals_e, y2=yvals_m - yvals_e, color=color, alpha=0.2)


def get_total_reward_for_reward_predictive(experiment, frac=0.05, num_latent_states=None):
    total_rew = experiment.results['total_reward_list'][0]
    rew_pred_err = experiment.results['reward_prediction_error_list'][0]

    if num_latent_states is None:
        total_reward_list = total_rew
        reward_prediction_error_list = rew_pred_err
    else:
        num_c = np.max(experiment.results['partition_list'][0], axis=-1) + 1
        total_reward_list = []
        reward_prediction_error_list = []
        for n, t_rew, rew_err in zip(num_c, total_rew, rew_pred_err):
            if n == num_latent_states:
                total_reward_list.append(t_rew)
                reward_prediction_error_list.append(rew_err)
        total_reward_list = np.stack(total_reward_list)
        reward_prediction_error_list = np.stack(reward_prediction_error_list)
    total_reward_m = np.mean(total_reward_list, axis=-1)
    reward_err_m = np.mean(np.mean(reward_prediction_error_list, axis=-1), axis=-1)
    rew_pred = _get_best_total_reward_at_transfer(total_reward_m, reward_err_m, frac=frac)
    return rew_pred


def get_total_reward_for_reward_maximizing(experiment, frac=0.05, num_latent_states=None):
    if num_latent_states is None:
        total_reward_list = experiment.results['total_reward_list'][0]
    else:
        num_c = np.max(experiment.results['partition_list'][0], axis=-1) + 1
        total_reward_list = []
        for n, v in zip(num_c, experiment.results['total_reward_list'][0]):
            if n == num_latent_states:
                total_reward_list.append(v)
        total_reward_list = np.stack(total_reward_list)
    total_reward_m = np.mean(total_reward_list, axis=-1)
    rew_max = _get_best_total_reward_at_transfer(total_reward_m, -total_reward_m, frac=frac)
    return rew_max


def _get_best_total_reward_at_transfer(total_reward, scores, frac=0.05):
    num_abs, num_task = np.shape(total_reward)
    best_num = int(np.ceil(num_abs * frac))

    total_reward_transfer = []
    for task_idx in range(num_task):
        best_part = np.argsort(scores[:, task_idx])[:best_num]
        total_reward_all = np.array([total_reward[i] for i in best_part])
        bit_mask = np.reshape(1. - rl.one_hot(task_idx, num_task), [1, -1])
        bit_mask = bit_mask / np.sum(bit_mask, axis=-1, keepdims=True)
        total_reward_transfer_from_task = np.sum(total_reward_all * bit_mask, axis=-1)
        total_reward_transfer.append(total_reward_transfer_from_task)

    total_reward_transfer = np.stack(total_reward_transfer)
    rand_task_idx = np.random.randint(0, num_task, size=best_num)
    total_reward_transfer = np.stack([total_reward_transfer[j, i] for i, j in enumerate(rand_task_idx)])

    return total_reward_transfer