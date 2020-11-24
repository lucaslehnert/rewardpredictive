#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from . import enumerate_partitions
from . import evaluate
from . import experiment
from . import mdp
from . import plot
from . import utils

# from .significant_experiment import construct_experiment_set_by_name
from .experiment import SmallTaskSequenceName
from .experiment import ExperimentSetTaskSequenceRewardChangeQLearning
from .experiment import ExperimentSetTaskSequenceRewardChangeQTransfer
from .experiment import ExperimentSetTaskSequenceRewardChangeSFLearning
from .experiment import ExperimentSetTaskSequenceRewardChangeSFTransfer
from .experiment import ExperimentSetTaskSequenceRewardChangeSFTransferAll

from .experiment import ExperimentSetTaskSequenceRandomRewardChangeQLearning
from .experiment import ExperimentSetTaskSequenceRandomRewardChangeQTransfer
from .experiment import ExperimentSetTaskSequenceRandomRewardChangeSFLearning
from .experiment import ExperimentSetTaskSequenceRandomRewardChangeSFTransfer
from .experiment import ExperimentSetTaskSequenceRandomRewardChangeSFTransferAll

from .plot import plot_alpha_vs_belief_space_size
from .plot import get_total_reward_for_reward_predictive
from .plot import get_total_reward_for_reward_maximizing
from .plot import plot_alpha_vs_episode_length
from .plot import plot_belief_posterior
from .plot import plot_convergence_rate_comparison
from .plot import plot_double_state_space_abstraction
from .plot import plot_half_state_space_abstraction
from .plot import plot_lr_comparison_dirichlet_process_model
from .plot import plot_lr_comparison_qlearning
from .plot import plot_lr_comparison_sflearning
from .plot import plot_maze_a_background
from .plot import plot_maze_a_decoration
from .plot import plot_maze_b_background
from .plot import plot_maze_b_decoration
from .plot import plot_lr_sf_lr_rew_comparison_dirichlet_process_model
from .plot import plot_maze_task_ep_len
from .plot import plot_maze_task_ep_len_broken_yaxis
from .plot import plot_maze_posterior
from .plot import plot_histogram_column_world
from .plot import plot_and_save_histogram_column_world
from .plot import plot_histogram_goal_and_wall_world
from .plot import plot_and_save_histogram_goal_and_wall_world
from .plot import plot_histogram_rand_mdp
from .plot import plot_and_save_histogram_rand_mdp
from .plot import plot_cycle_mdp_belief_space_size
from .plot import plot_alpha_vs_total_reward
# from .plot import plot_avg_highest_count

from .utils import set_seeds

def construct_experiment_set_by_name(experiment_set_name, **kwargs):
    return globals()[experiment_set_name].construct(**kwargs)
