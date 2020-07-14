#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from unittest import TestCase

class TestGuitarMDP(TestCase):
    def test_zero_negative_reward(self):
        import rlutils as rl
        import rewardpredictive as rp
        import numpy as np

        gamma = 0.9
        note_seq_corr = ('C', 'D', 'E', 'F', 'G', 'A', 'B')
        mdp_0 = rp.guitar.guitar_melody_mdp(note_seq_corr, reward_max=0., reward_min=-1.)
        agent = rl.agent.VIAgent(*mdp_0.get_t_mat_r_vec(), gamma=gamma)
        policy = rl.policy.GreedyPolicy(agent)
        traj_logger = rl.logging.LoggerTrajectory()
        rl.data.simulate_gracefully(mdp_0, policy, traj_logger)
        action_seq = traj_logger.get_trajectory_list()[0].all()[1]
        note_seq = [rp.guitar.notes[a] for a in action_seq]
        self.assertTrue(all([n1 == n2 for n1, n2 in zip(note_seq_corr, note_seq)]))

        _, v_star = rl.algorithm.vi(*mdp_0.get_t_mat_r_vec(), gamma=gamma)
        self.assertEqual(0., np.max(v_star))

    def test_positive_negative_reward(self):
        import rlutils as rl
        import rewardpredictive as rp
        import numpy as np

        gamma = 0.9
        note_seq_corr = ('C', 'D', 'E', 'F', 'G', 'A', 'B')
        mdp_0 = rp.guitar.guitar_melody_mdp(note_seq_corr, reward_max=1., reward_min=-1.)
        agent = rl.agent.VIAgent(*mdp_0.get_t_mat_r_vec(), gamma=gamma)
        policy = rl.policy.GreedyPolicy(agent)
        traj_logger = rl.logging.LoggerTrajectory()
        rl.data.simulate_gracefully(mdp_0, policy, traj_logger)
        action_seq = traj_logger.get_trajectory_list()[0].all()[1]
        note_seq = [rp.guitar.notes[a] for a in action_seq]
        self.assertTrue(all([n1 == n2 for n1, n2 in zip(note_seq_corr, note_seq)]))

        _, v_star = rl.algorithm.vi(*mdp_0.get_t_mat_r_vec(), gamma=gamma)
        self.assertTrue(np.abs((gamma ** 7 - 1) / (gamma - 1) - np.max(v_star)) <= 1e-5)

