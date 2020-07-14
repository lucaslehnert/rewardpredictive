#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
import unittest


class TestRewardPredictiveScores(unittest.TestCase):
    def test_score_complete_model(self):
        import rewardpredictive as rp
        import numpy as np

        task_seq, phi_mat_seq = rp.mdp.load_maze_sequence()
        vc = np.ones([4, 200], dtype=np.int)
        for task, phi_mat in zip(task_seq, phi_mat_seq):
            t_mat, r_vec = task.get_t_mat_r_vec()
            score = rp.reward_predictive.reward_predictive_score(phi_mat, t_mat, r_vec, vc, gamma=0.9, alpha=1.0)
            self.assertLessEqual(-1e-5, score)

    def test_score_reward_predictive(self):
        import rewardpredictive as rp
        import numpy as np

        task_seq, phi_mat_seq = rp.mdp.load_maze_sequence()
        vc = np.ones([4, 200], dtype=np.int)

        phi_mat_0 = phi_mat_seq[0]
        phi_mat_1 = phi_mat_seq[1]

        t_mat, r_vec = task_seq[0].get_t_mat_r_vec()
        score = rp.reward_predictive.reward_predictive_score(phi_mat_1, t_mat, r_vec, vc, gamma=0.9, alpha=1.0)
        self.assertLessEqual(score, -.1)

        t_mat, r_vec = task_seq[2].get_t_mat_r_vec()
        score = rp.reward_predictive.reward_predictive_score(phi_mat_1, t_mat, r_vec, vc, gamma=0.9, alpha=1.0)
        self.assertLessEqual(score, -.1)

        t_mat, r_vec = task_seq[3].get_t_mat_r_vec()
        score = rp.reward_predictive.reward_predictive_score(phi_mat_1, t_mat, r_vec, vc, gamma=0.9, alpha=1.0)
        self.assertLessEqual(score, -.1)

        t_mat, r_vec = task_seq[1].get_t_mat_r_vec()
        score = rp.reward_predictive.reward_predictive_score(phi_mat_0, t_mat, r_vec, vc, gamma=0.9, alpha=1.0)
        self.assertLessEqual(score, -.1)

        t_mat, r_vec = task_seq[4].get_t_mat_r_vec()
        score = rp.reward_predictive.reward_predictive_score(phi_mat_0, t_mat, r_vec, vc, gamma=0.9, alpha=1.0)
        self.assertLessEqual(score, -.1)

    def test_score_partial_model(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        task_seq, phi_mat_seq = rp.mdp.load_maze_sequence(slip_prob=.0)
        for task, phi_mat in zip(task_seq, phi_mat_seq):
            table_model = rp.utils.TableModel(200, 4)
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
            score = rp.reward_predictive.reward_predictive_score(phi_mat, t_mat, r_vec, vc, gamma=0.9, alpha=1.0)
            self.assertLessEqual(-1e-5, score)
