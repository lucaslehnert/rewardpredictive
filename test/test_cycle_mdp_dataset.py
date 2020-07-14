#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from unittest import TestCase


class TestSampleCycleMDP(TestCase):
    def _test_cycle_mdp_matrices(self, t_mat, r_mat):
        import numpy as np

        a, i, j = np.where(r_mat == 1.)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(i), 1)
        self.assertEqual(len(j), 1)
        a = a[0]
        i = i[0]
        j = j[0]
        self.assertEqual(t_mat[a, i, j], 1.)
        for i in range(3):
            j_next = np.where(t_mat[:, i, :] == 1.)[0]
            self.assertTrue(np.any(j_next != i))

    def test_sample_cycle_mdp(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np
        t_mat, r_mat = rp.cycle_mdp_dataset.sample_cycle_mdp(3, 3)
        self._test_cycle_mdp_matrices(t_mat, r_mat)

        mdp = rl.environment.TabularMDP(t_mat, r_mat, [0, 1, 2], [])
        agent = rl.agent.VIAgent(*mdp.get_t_mat_r_vec(), gamma=0.9)
        logger = rl.logging.LoggerTotalReward()
        policy = rl.policy.GreedyPolicy(agent)
        for _ in range(20):
            rl.data.simulate_gracefully(mdp, policy, logger, max_steps=10)
        self.assertLessEqual(3.0, np.mean(logger.get_total_reward_episodic()))

    def test_sample_cycle_mdp_sequence_dataset(self):
        from rewardpredictive.cycle_mdp_dataset import sample_cycle_mdp_sequence_dataset
        task_seq_list, part_idx, part_list = sample_cycle_mdp_sequence_dataset(
            num_sequences=2
        )
        self.assertEqual(len(task_seq_list), 2)
        self.assertEqual(len(part_idx), 2)
        self.assertEqual(len(part_list), 2)
        for task_seq, idx in zip(task_seq_list, part_idx):
            self._test_task_partition_seq(task_seq, part_list[idx])

    def _test_task_partition_seq(self, task_seq, partition_seq):
        import rewardpredictive as rp
        import numpy as np
        for task, partition in zip(task_seq, partition_seq):
            t_mat, r_vec = task.get_t_mat_r_vec()
            phi_mat = rp.utils.cluster_idx_to_phi_mat(partition)
            vc = np.ones([task.num_actions(), task.num_states()], dtype=np.float32)
            score = rp.reward_maximizing.reward_maximizing_score(phi_mat, t_mat, r_vec, vc, gamma=0.9)
            self.assertLessEqual(abs(score), 1e-5)


class TestLoadCycleMDPDataset(TestCase):
    def test_load_cycle_mdp_dataset(self):
        import rewardpredictive as rp
        import numpy as np
        from functools import reduce

        mdp_dataset, partition_idx_seq_list, partition_list = rp.cycle_mdp_dataset.load_cycle_mdp_dataset()
        phi_mat_list = [rp.utils.cluster_idx_to_phi_mat(p) for p in partition_list]

        self.assertEqual(len(mdp_dataset), 100)
        self.assertEqual(len(partition_idx_seq_list), 100)
        for mdp_seq in mdp_dataset:
            self.assertEqual(len(mdp_seq), 20)
        for part_idx_seq in partition_idx_seq_list:
            self.assertEqual(len(part_idx_seq), 20)

        num_s = mdp_dataset[0][0].num_states()
        num_a = mdp_dataset[0][0].num_actions()
        vc = np.ones([num_a, num_s], dtype=np.float32)

        mdp_list = reduce(lambda a, b: a + b, mdp_dataset)
        part_idx_list = np.reshape(partition_idx_seq_list, -1)
        for mdp, part_idx in zip(mdp_list, part_idx_list):
            phi_mat = phi_mat_list[part_idx]
            score = rp.reward_predictive.reward_predictive_score(phi_mat, *mdp.get_t_mat_r_vec(), vc, 0.9, alpha=1.0)
            self.assertLessEqual(-1e-5, score)

