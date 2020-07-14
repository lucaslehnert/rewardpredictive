#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from unittest import TestCase


class TestEvalTotalReward(TestCase):
    def test_task_no_terminal_state(self):
        import rewardpredictive as rp
        import numpy as np

        for task in rp.mdp.load_two_goal_with_wall_gridworld_sequence():
            partition = np.arange(task.num_states())
            total_rew = rp.evaluate.eval_total_reward(task, partition, repeats=10, rollout_depth=100, gamma=0.9)
            self.assertTrue(all(total_rew >= 1.))

    def test_task_terminal_state(self):
        import rewardpredictive as rp
        import numpy as np

        task = rp.mdp.MazeA(slip_prob=0.)
        partition = np.arange(task.num_states())
        total_rew = rp.evaluate.eval_total_reward(task, partition, repeats=2, rollout_depth=1000, gamma=0.9)
        self.assertEqual(total_rew[0], 1.)
        self.assertEqual(total_rew[1], 1.)


class TestEvalRewardPredictive(TestCase):
    def test_task_identity(self):
        import rewardpredictive as rp
        import numpy as np

        task = rp.mdp.GridWord3x3WithGoalsAndWalls([0, 1], 3, slip_prob=0.0, dtype=np.float32)
        partition = np.arange(task.num_states())
        rew_err = rp.evaluate.eval_reward_predictive(task, partition, repeats=5, rollout_depth=10)
        self.assertTrue(np.all(rew_err == 0.))

    def test_task_column_world(self):
        import rewardpredictive as rp
        import numpy as np

        task = rp.mdp.ColumnWorld0(slip_prob=0.)
        partition = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int)
        rew_err = rp.evaluate.eval_reward_predictive(task, partition, repeats=5, rollout_depth=10)
        self.assertTrue(np.all(rew_err == 0.))
