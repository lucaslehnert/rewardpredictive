#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
import unittest


class TestRewardMaximizingScores(unittest.TestCase):
    def test_score_complete_model(self):
        import rewardpredictive as rp
        import numpy as np

        task_seq, phi_mat_seq = rp.mdp.load_maze_sequence()
        vc = np.ones([4, 200], dtype=np.int)
        for task, phi_mat in zip(task_seq, phi_mat_seq):
            t_mat, r_vec = task.get_t_mat_r_vec()
            score = rp.reward_maximizing.reward_maximizing_score(phi_mat, t_mat, r_vec, vc, gamma=0.9)
            self.assertLessEqual(-1e-5, score)

    def test_score_reward_maximising_0(self):
        import rewardpredictive as rp
        import numpy as np

        task_seq, _ = rp.mdp.load_maze_sequence()
        vc = np.ones([4, 200], dtype=np.int)
        for task in task_seq:
            t_mat, r_vec = task.get_t_mat_r_vec()
            phi_mat = rp.reward_maximizing.reward_maximizing_representation(t_mat, r_vec, 100, 0.9)
            score = rp.reward_maximizing.reward_maximizing_score(phi_mat, t_mat, r_vec, vc, gamma=0.9)
            self.assertLessEqual(-1e-5, score)

    def test_score_reward_maximising_1(self):
        import rewardpredictive as rp
        import numpy as np

        task_seq, _ = rp.mdp.load_maze_sequence()
        vc = np.ones([4, 200], dtype=np.int)

        t_mat, r_vec = task_seq[0].get_t_mat_r_vec()
        phi_mat = rp.reward_maximizing.reward_maximizing_representation(t_mat, r_vec, 100, 0.9)

        for i, task in enumerate(task_seq):
            t_mat, r_vec = task.get_t_mat_r_vec()
            score = rp.reward_maximizing.reward_maximizing_score(phi_mat, t_mat, r_vec, vc, gamma=0.9)
            if i == 0 or i == 2:
                self.assertLessEqual(-1e-5, score)
            else:
                self.assertLessEqual(score, -.01)

    def test_score_partial_model(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        task_seq, phi_mat_seq = rp.mdp.load_maze_sequence(slip_prob=.0)
        for task, phi_mat in zip(task_seq, phi_mat_seq):
            table_model = rp.utils.TableModel(200, 4, reward_sampler=lambda s: np.zeros(s, dtype=np.float32))
            agent = rp.utils.SFLearning(
                num_states=200,
                num_actions=4,
                learning_rate_sf=0.5,
                learning_rate_reward=0.9,
                gamma=0.9,
                init_sf_mat=np.eye(4 * 200, dtype=np.float32),
                init_w_vec=np.ones(4 * 200, dtype=np.float32)
            )
            transition_listener = rl.data.transition_listener(agent, table_model)
            rl.data.simulate_gracefully(task, rl.policy.GreedyPolicy(agent), transition_listener, max_steps=100)

            t_mat, r_vec = table_model.get_t_mat_r_vec()
            vc = table_model.visitation_counts()
            score = rp.reward_maximizing.reward_maximizing_score(phi_mat, t_mat, r_vec, vc, gamma=0.9)
            self.assertLessEqual(-1e-5, score)

    def _get_four_state_task(self):
        import rlutils as rl
        import numpy as np

        t_mat = np.array([
            [
                [0., 0., .5, .5],
                [0., 0., .5, .5],
                [0., 0., .5, .5],
                [0., 0., .5, .5]
            ],
            [
                [.5, .5, 0., 0.],
                [.5, .5, 0., 0.],
                [0., 0., .5, .5],
                [0., 0., .5, .5]
            ]
        ], dtype=np.float32)
        r_mat = np.array([
            [
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 1., 1.],
                [0., 0., 1., 1.]
            ],
            [
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 1., 1.],
                [0., 0., 1., 1.]
            ]
        ], dtype=np.float32)
        mdp = rl.environment.TabularMDP(t_mat, r_mat, idx_start_list=[0, 1], idx_goal_list=[], name='TestMDP')
        phi_mat = np.array([
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [0., 1.]
        ], dtype=np.float32)

        return mdp, phi_mat

    def test_score_partial_model_small_0(self):
        import rewardpredictive as rp
        import numpy as np

        mdp, phi_mat = self._get_four_state_task()
        t_mat, r_vec = mdp.get_t_mat_r_vec()
        score = rp.reward_maximizing.reward_maximizing_score(
            phi_mat, t_mat, r_vec, np.ones([2, 4], dtype=np.float32), gamma=0.9
        )
        self.assertLessEqual(-1e-5, score)

    def test_score_partial_model_small_1(self):
        import rewardpredictive as rp
        import rlutils as rl

        table_model = rp.utils.TableModel(num_states=4, num_actions=2)
        table_model.update_transition(rl.one_hot(0, 4), 0, 0, rl.one_hot(2, 4), False, {})
        table_model.update_transition(rl.one_hot(0, 4), 1, 0, rl.one_hot(0, 4), False, {})
        table_model.update_transition(rl.one_hot(2, 4), 0, 1, rl.one_hot(2, 4), False, {})
        table_model.update_transition(rl.one_hot(2, 4), 1, 1, rl.one_hot(2, 4), False, {})
        mdp, phi_mat = self._get_four_state_task()
        t_mat, r_vec = table_model.get_t_mat_r_vec()
        score = rp.reward_maximizing.reward_maximizing_score(
            phi_mat, t_mat, r_vec, table_model.visitation_counts(), gamma=0.9
        )
        self.assertLessEqual(-1e-5, score)

    def test_score_partial_model_small_2(self):
        import rewardpredictive as rp
        import rlutils as rl

        table_model = rp.utils.TableModel(num_states=4, num_actions=2)
        table_model.update_transition(rl.one_hot(0, 4), 0, 0, rl.one_hot(2, 4), False, {})
        table_model.update_transition(rl.one_hot(2, 4), 0, 1, rl.one_hot(2, 4), False, {})
        table_model.update_transition(rl.one_hot(0, 4), 0, 0, rl.one_hot(3, 4), False, {})

        mdp, phi_mat = self._get_four_state_task()
        t_mat, r_vec = table_model.get_t_mat_r_vec()
        score = rp.reward_maximizing.reward_maximizing_score(
            phi_mat, t_mat, r_vec, table_model.visitation_counts(), gamma=0.9
        )
        self.assertLessEqual(-1e-5, score)

    def test_score_partial_model_small_3(self):
        import rewardpredictive as rp
        import rlutils as rl

        table_model = rp.utils.TableModel(num_states=4, num_actions=2)

        table_model.update_transition(rl.one_hot(0, 4), 0, 0, rl.one_hot(2, 4), False, {})
        table_model.update_transition(rl.one_hot(0, 4), 1, 0, rl.one_hot(0, 4), False, {})
        table_model.update_transition(rl.one_hot(2, 4), 0, 1, rl.one_hot(2, 4), False, {})
        table_model.update_transition(rl.one_hot(2, 4), 1, 1, rl.one_hot(2, 4), False, {})
        table_model.update_transition(rl.one_hot(1, 4), 1, 0, rl.one_hot(1, 4), False, {})

        mdp, phi_mat = self._get_four_state_task()
        t_mat, r_vec = table_model.get_t_mat_r_vec()
        score = rp.reward_maximizing.reward_maximizing_score(
            phi_mat, t_mat, r_vec, table_model.visitation_counts(), gamma=0.9
        )
        self.assertLessEqual(-1e-5, score)
