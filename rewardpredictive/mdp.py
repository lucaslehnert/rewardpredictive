#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from itertools import product, combinations

import numpy as np
import rlutils as rl
from rlutils.environment.gridworld import pt_to_idx, idx_to_pt, \
    generate_gridworld_transition_function, \
    generate_gridworld_transition_function_with_barrier, \
    generate_mdp_from_transition_and_reward_function

from .utils import cluster_idx_to_phi_mat
from .enumerate_partitions import enumerate_n_partitions


class InflatedTabularMDP(rl.environment.TabularMDP):
    def __init__(self, env, phi_mat, agg_weights=None, start_states=None, goal_states=None):
        '''

        :param env:
        :param phi_mat:
        :param agg_weights: If None, then uniform aggregation weights are used.
        '''
        num_a = env.num_actions()
        num_s = np.shape(phi_mat)[0]
        if agg_weights is None:
            uniform_weights = phi_mat.transpose()
            uniform_weights = uniform_weights / np.sum(uniform_weights, axis=-1, keepdims=True)
            uniform_weights = np.reshape(uniform_weights, [1, 1] + list(np.shape(uniform_weights)))
            agg_weights = np.concatenate([uniform_weights] * num_s, axis=1)
            agg_weights = np.concatenate([agg_weights] * num_a, axis=0)

        t_mat_lat, r_mat_lat = env.get_t_mat_r_mat()

        t_mat = np.zeros([num_a, num_s, num_s], dtype=np.float32)
        r_mat = np.zeros([num_a, num_s, num_s], dtype=np.float32)

        for s, a in product(range(num_s), range(num_a)):
            s_phi = np.where(phi_mat[s] == 1.)[0][0]
            t_mat[a, s] = np.matmul(t_mat_lat[a, s_phi], agg_weights[a, s])
            r_mat[a, s] = np.matmul(r_mat_lat[a, s_phi], agg_weights[a, s])

        if start_states is None:
            if len(env.start_state_list()) > 0:
                start_bits = [np.matmul(phi_mat, rl.one_hot(i, env.num_states())) for i in env.start_state_list()]
                start_states = np.concatenate([np.where(b == 1.)[0] for b in start_bits])
            else:
                start_states = []
        if goal_states is None:
            if len(env.goal_state_list()) > 0:
                goal_bits = [np.matmul(phi_mat, rl.one_hot(i, env.num_states())) for i in env.goal_state_list()]
                goal_states = np.concatenate([np.where(b == 1.)[0] for b in goal_bits])
            else:
                goal_states = []

        self.env_latent = env
        super().__init__(t_mat, r_mat, start_states, goal_states)

    def __str__(self):
        return 'InflatedTabularMDP({})'.format(str(self.env_latent))


class NaviA(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 5), (10, 10))]
        goal_list_idx = [pt_to_idx((9, 0), (10, 10))]

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
        return 'NaviA'


class NaviB(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 4), (10, 10))]
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
        return 'NaviB'


class MazeA(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        barrier_idx_list += [(pt_to_idx((1, i), (10, 10)), pt_to_idx((2, i), (10, 10))) for i in np.arange(0, 8)]
        barrier_idx_list += [(pt_to_idx((3, i), (10, 10)), pt_to_idx((4, i), (10, 10))) for i in np.arange(2, 10)]
        barrier_idx_list += [(pt_to_idx((5, i), (10, 10)), pt_to_idx((6, i), (10, 10))) for i in np.arange(0, 8)]
        barrier_idx_list += [(pt_to_idx((7, i), (10, 10)), pt_to_idx((8, i), (10, 10))) for i in np.arange(2, 10)]
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 0), (10, 10))]
        goal_list_idx = [pt_to_idx((9, 0), (10, 10))]

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
        return 'MazeA'


class MazeB(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        barrier_idx_list += [(pt_to_idx((1, i), (10, 10)), pt_to_idx((2, i), (10, 10))) for i in np.arange(2, 10)]
        barrier_idx_list += [(pt_to_idx((3, i), (10, 10)), pt_to_idx((4, i), (10, 10))) for i in np.arange(0, 8)]
        barrier_idx_list += [(pt_to_idx((5, i), (10, 10)), pt_to_idx((6, i), (10, 10))) for i in np.arange(2, 10)]
        barrier_idx_list += [(pt_to_idx((7, i), (10, 10)), pt_to_idx((8, i), (10, 10))) for i in np.arange(0, 8)]
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
        return 'MazeB'


def color_right_half_10_by_10_grid(task):
    s_idx_unique = list(filter(lambda i: idx_to_pt(i, (10, 10))[0] < 5, range(100)))
    s_idx_duplicate = list(filter(lambda i: idx_to_pt(i, (10, 10))[0] >= 5, range(100)))
    partition_map = s_idx_unique + s_idx_duplicate + s_idx_duplicate + s_idx_duplicate
    phi_mat = cluster_idx_to_phi_mat(partition_map)
    return InflatedTabularMDP(task, phi_mat), phi_mat


def double_state_space(task):
    cluster_idx = np.concatenate([np.arange(task.num_states()), np.arange(task.num_states())])
    phi_mat = cluster_idx_to_phi_mat(cluster_idx)
    return InflatedTabularMDP(task, phi_mat), phi_mat


def identity_inflation(task):
    agg_weights_pinv = np.concatenate((np.eye(task.num_states(), dtype=np.float32),
                                       np.zeros((task.num_states(), task.num_states()), dtype=np.float32)), axis=0)
    agg_weights = np.linalg.pinv(agg_weights_pinv)
    agg_weights = np.stack([agg_weights for _ in range(task.num_states() * 2)])
    agg_weights = np.stack([agg_weights for _ in range(task.num_actions())])
    phi_mat = np.concatenate((np.eye(task.num_states(), dtype=np.float32),
                              np.eye(task.num_states(), dtype=np.float32)), axis=0)

    mdp_infl = InflatedTabularMDP(task,
                                  phi_mat=phi_mat,
                                  agg_weights=agg_weights,
                                  start_states=task.start_state_list(),
                                  goal_states=task.goal_state_list())
    return mdp_infl, phi_mat


class TaskSequence(object):
    def __init__(self):
        self.task_sequence = []

    @classmethod
    def get_classname(cls):
        return cls.__name__


class NavigationTaskSequence(TaskSequence):
    def __init__(self):
        super().__init__()
        self.task_sequence = load_navigation_sequence()[0]


class MazeTaskSequence(TaskSequence):
    def __init__(self):
        super().__init__()
        self.task_sequence = load_maze_sequence()[0]


class ShortMazeTaskSequence(TaskSequence):
    def __init__(self):
        super().__init__()
        self.task_sequence = load_short_maze_sequence()[0]


def load_task_sequence_from_string(task_seq_name: str):
    return globals()[task_seq_name]()


def load_navigation_sequence():
    task_1, phi_mat_1 = double_state_space(NaviA())
    task_2, phi_mat_2 = color_right_half_10_by_10_grid(NaviB())
    task_3, phi_mat_3 = double_state_space(NaviA())
    task_4, phi_mat_4 = double_state_space(NaviB())
    task_5, phi_mat_5 = color_right_half_10_by_10_grid(NaviA())
    task_seq = [task_1, task_2, task_3, task_4, task_5]
    phi_mat_seq = [phi_mat_1, phi_mat_2, phi_mat_3, phi_mat_4, phi_mat_5]
    return task_seq, phi_mat_seq


def load_maze_sequence(slip_prob=0.05):
    task_1, phi_mat_1 = double_state_space(MazeA(slip_prob=slip_prob))
    task_2, phi_mat_2 = color_right_half_10_by_10_grid(MazeB(slip_prob=slip_prob))
    task_3, phi_mat_3 = double_state_space(MazeA(slip_prob=slip_prob))
    task_4, phi_mat_4 = double_state_space(MazeB(slip_prob=slip_prob))
    task_5, phi_mat_5 = color_right_half_10_by_10_grid(MazeA(slip_prob=slip_prob))

    task_seq = [task_1, task_2, task_3, task_4, task_5]
    phi_mat_seq = [phi_mat_1, phi_mat_2, phi_mat_3, phi_mat_4, phi_mat_5]
    return task_seq, phi_mat_seq


def load_short_maze_sequence():
    task_1, phi_mat_1 = double_state_space(MazeA())
    task_2, phi_mat_2 = color_right_half_10_by_10_grid(MazeB())
    task_3, phi_mat_3 = double_state_space(MazeB())
    task_seq = [task_1, task_2, task_3]
    phi_mat_seq = [phi_mat_1, phi_mat_2, phi_mat_3]
    return task_seq, phi_mat_seq


class ColumnWorld0(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05, dtype=np.float32):
        t_fn = generate_gridworld_transition_function(3, 3, slip_prob=slip_prob)

        def r_fn(s_1, a, s_2):
            x, y = idx_to_pt(s_2, (3, 3))
            if x == 0:
                return 1.
            else:
                return 0.

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(
            num_states=3 * 3,
            num_actions=4,
            transition_fn=t_fn,
            reward_fn=r_fn,
            reward_matrix=True,
            dtype=dtype
        )
        super().__init__(t_mat, r_mat, np.arange(9), [])


class ColumnWorld1(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05, dtype=np.float32):
        t_fn = generate_gridworld_transition_function(3, 3, slip_prob=slip_prob)

        def r_fn(s_1, a, s_2):
            x, y = idx_to_pt(s_2, (3, 3))
            if x == 1:
                return 1.
            else:
                return 0.

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(
            num_states=3 * 3,
            num_actions=4,
            transition_fn=t_fn,
            reward_fn=r_fn,
            reward_matrix=True,
            dtype=dtype
        )
        super().__init__(t_mat, r_mat, np.arange(9), [])


class ColumnWorld2(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05, dtype=np.float32):
        t_fn = generate_gridworld_transition_function(3, 3, slip_prob=slip_prob)

        def r_fn(s_1, a, s_2):
            x, y = idx_to_pt(s_2, (3, 3))
            if x == 2:
                return 1.
            else:
                return 0.

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(
            num_states=3 * 3,
            num_actions=4,
            transition_fn=t_fn,
            reward_fn=r_fn,
            reward_matrix=True,
            dtype=dtype
        )
        super().__init__(t_mat, r_mat, np.arange(9), [])


def load_column_world_sequence():
    return [
        ColumnWorld0(),
        ColumnWorld1(),
        ColumnWorld2()
    ]


class GridWord3x3WithGoals(rl.environment.TabularMDP):
    def __init__(self, goal_state_idx_list, slip_prob=0.05, dtype=np.float32):
        assert np.min(goal_state_idx_list) >= 0
        assert np.max(goal_state_idx_list) < 9

        t_fn = generate_gridworld_transition_function(3, 3, slip_prob=slip_prob)

        def r_fn(s_1, a, s_2):
            nonlocal goal_state_idx_list
            if s_2 in goal_state_idx_list:
                return 1.
            else:
                return 0.

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(
            num_states=3 * 3,
            num_actions=4,
            transition_fn=t_fn,
            reward_fn=r_fn,
            reward_matrix=True,
            dtype=dtype
        )
        super().__init__(t_mat, r_mat, np.arange(9), [])


class GridWord3x3WithGoalsAndWalls(rl.environment.TabularMDP):
    @staticmethod
    def _get_wall_position_0():
        """
        .|. .
        .|. .
        . . .
        """
        barrier_idx_list = [
            (pt_to_idx((0, 0), (3, 3)), pt_to_idx((1, 0), (3, 3))),
            (pt_to_idx((0, 1), (3, 3)), pt_to_idx((1, 1), (3, 3)))
        ]
        return barrier_idx_list

    @staticmethod
    def _get_wall_position_1():
        """
        . .|.
        . .|.
        . . .
        """
        barrier_idx_list = [
            (pt_to_idx((1, 0), (3, 3)), pt_to_idx((2, 0), (3, 3))),
            (pt_to_idx((1, 1), (3, 3)), pt_to_idx((2, 1), (3, 3)))
        ]
        return barrier_idx_list

    @staticmethod
    def _get_wall_position_2():
        """
        . . .
        .|. .
        .|. .
        """
        barrier_idx_list = [
            (pt_to_idx((0, 1), (3, 3)), pt_to_idx((1, 1), (3, 3))),
            (pt_to_idx((0, 2), (3, 3)), pt_to_idx((1, 2), (3, 3)))
        ]
        return barrier_idx_list

    @staticmethod
    def _get_wall_position_3():
        """
        . . .
        . .|.
        . .|.
        """
        barrier_idx_list = [
            (pt_to_idx((1, 1), (3, 3)), pt_to_idx((2, 1), (3, 3))),
            (pt_to_idx((1, 2), (3, 3)), pt_to_idx((2, 2), (3, 3)))
        ]
        return barrier_idx_list

    def __init__(self, goal_state_idx_list, wall_position_idx, slip_prob=0.05, dtype=np.float32):
        assert np.min(goal_state_idx_list) >= 0
        assert np.max(goal_state_idx_list) < 9

        if wall_position_idx == 0:
            barrier_idx_list = GridWord3x3WithGoalsAndWalls._get_wall_position_0()
        elif wall_position_idx == 1:
            barrier_idx_list = GridWord3x3WithGoalsAndWalls._get_wall_position_1()
        elif wall_position_idx == 2:
            barrier_idx_list = GridWord3x3WithGoalsAndWalls._get_wall_position_2()
        elif wall_position_idx == 3:
            barrier_idx_list = GridWord3x3WithGoalsAndWalls._get_wall_position_3()

        t_fn = generate_gridworld_transition_function_with_barrier(3, 3, slip_prob, barrier_idx_list)

        def r_fn(s_1, a, s_2):
            nonlocal goal_state_idx_list
            if s_2 in goal_state_idx_list:
                return 1.
            else:
                return 0.

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(
            num_states=3 * 3,
            num_actions=4,
            transition_fn=t_fn,
            reward_fn=r_fn,
            reward_matrix=True,
            dtype=dtype
        )
        super().__init__(t_mat, r_mat, np.arange(9), [])


def load_two_goal_gridworld_sequence():
    return [GridWord3x3WithGoals(g) for g in combinations(range(9), 2)]


def load_two_goal_with_wall_gridworld_sequence():
    return [GridWord3x3WithGoalsAndWalls(g, w) for g, w in product(combinations(range(9), 2), range(4))]


class RandomMDP(rl.environment.TabularMDP):
    def __init__(self, num_states, num_actions, scale=10.0, num_goal_cells=1):
        t_mat = np.random.uniform(size=[num_actions, num_states, num_states]).astype(dtype=np.float32)
        t_mat = np.exp(scale * t_mat)
        t_mat = t_mat / np.sum(t_mat, axis=-1, keepdims=True)
        assert not np.any(np.isnan(t_mat))

        r_idx_list = np.random.choice(np.arange(num_states), size=num_goal_cells, replace=False)
        r_state = np.zeros(num_states)
        for i in r_idx_list:
            r_state[i] = 1.

        def r_fn(s, a, s_next):
            return r_state[s_next]

        r_mat = np.zeros([num_actions, num_states, num_states])
        for a in range(num_actions):
            for s_1 in range(num_states):
                for s_2 in range(num_states):
                    r_mat[a, s_1, s_2] = r_fn(s_1, a, s_2)

        super().__init__(t_mat, r_mat, idx_start_list=np.arange(num_states), idx_goal_list=[], name='RandomMDP')


def load_random_mdp_sequence():
    partition_list = enumerate_n_partitions(9, 3)
    partition_idx = np.random.randint(0, np.shape(partition_list)[0])
    phi_mat = cluster_idx_to_phi_mat(partition_list[partition_idx])

    latent_mdp_list = [RandomMDP(3, 3) for _ in range(100)]
    mdp_list = [InflatedTabularMDP(env, phi_mat) for env in latent_mdp_list]
    return mdp_list


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
