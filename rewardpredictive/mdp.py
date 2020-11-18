#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import numpy as np
import rlutils as rl
from rlutils.environment.gridworld import pt_to_idx, \
    generate_gridworld_transition_function_with_barrier, \
    generate_mdp_from_transition_and_reward_function


class TaskASlightRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 9), (10, 10))]
        goal_list_idx = [pt_to_idx((8, 0), (10, 10))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def __str__(self):
        return 'TaskASlightRewardChange'


class TaskBSlightRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 9), (10, 10))]
        goal_list_idx = [pt_to_idx((9, 1), (10, 10))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def __str__(self):
        return 'TaskBSlightRewardChange'


class TaskASignificantRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 9), (10, 10))]
        goal_list_idx = [pt_to_idx((9, 9), (10, 10))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def __str__(self):
        return 'TaskASignificantRewardChange'


class TaskBSignificantRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 9), (10, 10))]
        goal_list_idx = [pt_to_idx((0, 0), (10, 10))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def __str__(self):
        return 'TaskBSignificantRewardChange'

class RandomRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        """
        TODO
        :param slip_prob:
        """
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 0), (10, 10))]
        self.possible_indices = np.arange(0, 10)
        self.ignore_positions = [(0, 0)]

        goal_list_idx = [pt_to_idx(self.sample_position(), (10, 10))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def sample_position(self):
        """
        Samples a random position, excluding a given list of positions.
        :param exclude: list of 2-tuples, positions to exclude from sampling
        :return: 2-tuple denoting a position.
        """
        pos = tuple(np.random.choice(self.possible_indices, size=2, replace=True))
        if pos not in self.ignore_positions:
            return pos

        return self.sample_position()

    def __str__(self):
        return 'TaskBSignificantRewardChange'
