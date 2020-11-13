#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from unittest import TestCase


class TestGridWord3x3WithGoalsAndWalls(TestCase):
    def test(self):
        import rewardpredictive as rp
        import numpy as np
        import rlutils as rl
        from rlutils.environment.gridworld import pt_to_idx

        mdp = rp.mdp.GridWord3x3WithGoalsAndWalls([0, 1], 0, slip_prob=0.)
        t_mat, r_mat = mdp.get_t_mat_r_mat()

        left = rl.environment.gridworld.GridWorldAction.LEFT
        right = rl.environment.gridworld.GridWorldAction.RIGHT
        up = rl.environment.gridworld.GridWorldAction.UP
        down = rl.environment.gridworld.GridWorldAction.DOWN
        pt_to_state = lambda x, y: pt_to_idx((x, y), (3, 3))
        next_state = lambda x, y, a: np.where(np.matmul(rl.one_hot(pt_to_state(x, y), 9), t_mat[a]))[0][0]

        self.assertEqual(next_state(0, 0, left), pt_to_state(0, 0))
        self.assertEqual(next_state(0, 1, left), pt_to_state(0, 1))
        self.assertEqual(next_state(0, 2, left), pt_to_state(0, 2))
        self.assertEqual(next_state(1, 0, left), pt_to_state(1, 0))
        self.assertEqual(next_state(1, 1, left), pt_to_state(1, 1))
        self.assertEqual(next_state(1, 2, left), pt_to_state(0, 2))
        self.assertEqual(next_state(2, 0, left), pt_to_state(1, 0))
        self.assertEqual(next_state(2, 1, left), pt_to_state(1, 1))
        self.assertEqual(next_state(2, 2, left), pt_to_state(1, 2))
        self.assertEqual(next_state(0, 0, right), pt_to_state(0, 0))
        self.assertEqual(next_state(0, 1, right), pt_to_state(0, 1))
        self.assertEqual(next_state(0, 2, right), pt_to_state(1, 2))
        self.assertEqual(next_state(1, 0, right), pt_to_state(2, 0))
        self.assertEqual(next_state(1, 1, right), pt_to_state(2, 1))
        self.assertEqual(next_state(1, 2, right), pt_to_state(2, 2))
        self.assertEqual(next_state(2, 0, right), pt_to_state(2, 0))
        self.assertEqual(next_state(2, 1, right), pt_to_state(2, 1))
        self.assertEqual(next_state(2, 2, right), pt_to_state(2, 2))


class TestRandomReward(TestCase):
    """
    TODO
    """
    def test(self):
        raise NotImplementedError()

