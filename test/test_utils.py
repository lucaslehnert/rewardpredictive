#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
import unittest
from unittest import TestCase


class TestLAM(unittest.TestCase):
    def test_lam_complete_model(self):
        import rewardpredictive as rp
        import numpy as np

        task_seq, phi_mat_seq = rp.mdp.load_maze_sequence()
        task = task_seq[0]
        t_mat, r_vec = task.get_t_mat_r_vec()
        phi_mat = phi_mat_seq[0]

        m_mat, w_vec = rp.utils.lam_from_mat(t_mat, r_vec, phi_mat)

        self.assertTrue(np.all(np.matmul(np.linalg.pinv(phi_mat), r_vec.transpose()) == w_vec.transpose()))
        for a in range(task.num_actions()):
            self.assertTrue(np.all(np.matmul(np.matmul(np.linalg.pinv(phi_mat), t_mat[a]), phi_mat) == m_mat[a]))

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

    def test_lam_partial_model(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        table_model = rp.utils.TableModel(num_states=4, num_actions=2)
        table_model.update_transition(rl.one_hot(0, 4), 0, 0, rl.one_hot(2, 4), False, {})
        table_model.update_transition(rl.one_hot(0, 4), 1, 0, rl.one_hot(0, 4), False, {})
        table_model.update_transition(rl.one_hot(2, 4), 0, 1, rl.one_hot(2, 4), False, {})
        table_model.update_transition(rl.one_hot(2, 4), 1, 1, rl.one_hot(2, 4), False, {})
        task, phi_mat = self._get_four_state_task()
        t_mat, r_vec = table_model.get_t_mat_r_vec()

        m_mat, w_vec = rp.utils.lam_from_mat_visitation_counts(t_mat, r_vec, phi_mat, table_model.visitation_counts())
        m_mat_test = np.array([
            [
                [0., 1.],
                [0., 1.]
            ],
            [
                [1., 0.],
                [0., 1.]
            ]
        ], dtype=np.float32)
        w_vec_test = np.array([
            [0., 1.],
            [0., 1.]
        ], dtype=np.float32)
        self.assertTrue(np.all(m_mat == m_mat_test))
        self.assertTrue(np.all(w_vec == w_vec_test))


class TestNDArrayPadding(unittest.TestCase):
    def test(self):
        import rewardpredictive as rp
        import numpy as np
        ar = [
            [0., 1.],
            [0.]
        ]
        ar_pad = rp.utils.pad_list_of_list_to_ndarray(ar, pad_value=5, dtype=np.float32)
        ar_test = np.array([
            [0., 1.],
            [0., 5.]
        ], dtype=np.float32)
        self.assertTrue(np.all(ar_test == ar_pad))


class TestTableModel(TestCase):
    def test_getters(self):
        import rewardpredictive as rp

        table_model = rp.utils.TableModel(123, 4, max_reward=5)
        self.assertEqual(table_model.num_states(), 123)
        self.assertEqual(table_model.num_actions(), 4)
        self.assertEqual(table_model.get_max_reward(), 5)

    def test_reset(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        table_model = rp.utils.TableModel(3, 2, max_reward=1)
        table_model.update_transition(rl.one_hot(0, 3), 0, .1, rl.one_hot(0, 3), False, {})
        table_model.update_transition(rl.one_hot(0, 3), 0, 0, rl.one_hot(1, 3), False, {})
        table_model.on_simulation_timeout()
        table_model.update_transition(rl.one_hot(1, 3), 1, .5, rl.one_hot(2, 3), True, {})
        table_model.reset()
        t_mat, r_vec = table_model.get_t_mat_r_vec()
        self.assertTrue(np.all(t_mat[0] == np.eye(3, dtype=np.float32)))
        self.assertTrue(np.all(t_mat[1] == np.eye(3, dtype=np.float32)))
        self.assertTrue(np.all(r_vec[0] == np.ones([2, 3], dtype=np.float32)))
        self.assertTrue(np.all(r_vec[1] == np.ones([2, 3], dtype=np.float32)))
        self.assertTrue(np.all(table_model.visitation_counts() == 0))

    def test_reward_sampler(self):
        import rewardpredictive as rp
        import numpy as np

        table_model = rp.utils.TableModel(
            num_states=3,
            num_actions=2,
            max_reward=1,
            reward_sampler=lambda s: np.stack([np.arange(s[1]) for i in range(s[0])])
        )
        t_mat, r_vec = table_model.get_t_mat_r_vec()
        self.assertTrue(np.all(t_mat[0] == np.eye(3, dtype=np.float32)))
        self.assertTrue(np.all(t_mat[1] == np.eye(3, dtype=np.float32)))
        r_vec_corr = np.array([
            [0, 1, 2],
            [0, 1, 2]
        ], dtype=np.float32)
        self.assertTrue(np.all(r_vec == r_vec_corr))
        self.assertTrue(np.all(table_model.visitation_counts() == 0))

    def test_update_transition(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        table_model = rp.utils.TableModel(3, 2, max_reward=1)
        t_mat, r_vec = table_model.get_t_mat_r_vec()
        self.assertTrue(np.all(t_mat[0] == np.eye(3, dtype=np.float32)))
        self.assertTrue(np.all(t_mat[1] == np.eye(3, dtype=np.float32)))
        self.assertTrue(np.all(r_vec[0] == np.ones([2, 3], dtype=np.float32)))
        self.assertTrue(np.all(r_vec[1] == np.ones([2, 3], dtype=np.float32)))
        self.assertTrue(np.all(table_model.visitation_counts() == 0))

        table_model.update_transition(rl.one_hot(0, 3), 0, .1, rl.one_hot(0, 3), False, {})
        table_model.update_transition(rl.one_hot(0, 3), 0, 0, rl.one_hot(1, 3), False, {})
        table_model.on_simulation_timeout()
        table_model.update_transition(rl.one_hot(1, 3), 1, .5, rl.one_hot(2, 3), True, {})
        t_mat, r_vec = table_model.get_t_mat_r_vec()
        t_mat_corr = np.array([
            [[.5, .5, 0.],
             [0., 1., 0.],
             [0., 0., 1.]],
            [[1., 0., 0.],
             [0., 0., 1.],
             [0., 0., 1.]]
        ], dtype=np.float32)
        self.assertTrue(np.all(t_mat == t_mat_corr))
        r_vec_corr = np.array([
            [.05, 1., 1.],
            [1., 0.5, 1.]
        ], dtype=np.float32)
        self.assertTrue(np.all(r_vec == r_vec_corr))
        visitation_counts_corr = np.array([
            [2, 0, 0],
            [0, 1, 0]
        ], dtype=np.int)
        self.assertTrue(np.all(table_model.visitation_counts() == visitation_counts_corr))


class TestSFLearning(TestCase):
    def test_getters(self):
        import rewardpredictive as rp
        import numpy as np

        agent = rp.utils.SFLearning(
            num_states=14,
            num_actions=3,
            learning_rate_sf=0.1,
            learning_rate_reward=0.2,
            gamma=0.9
        )
        self.assertEqual(agent.get_gamma(), 0.9)
        self.assertEqual(agent.get_learning_rate_sf(), 0.1)
        self.assertEqual(agent.get_learning_rate_reward(), 0.2)
        self.assertTrue(np.all(agent.get_sf_matrix() == 0.))
        self.assertTrue(np.all(agent.get_w_vector() == 0.))
        self.assertTrue(np.all(agent.get_q_vector() == 0.))
        self.assertEqual(agent.get_error_avg(), 0.)

    def test_reset_0(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        agent = rp.utils.SFLearning(
            num_states=14,
            num_actions=3,
            learning_rate_sf=0.1,
            learning_rate_reward=0.2,
            gamma=0.9,
            init_sf_mat=np.eye(3 * 14, dtype=np.float32),
            init_w_vec=np.zeros(3 * 14, dtype=np.float32)
        )
        self.assertEqual(agent.get_error_avg(), 0.)

        for _ in range(10):
            agent.update_transition(rl.one_hot(0, 14), 0, 0, rl.one_hot(0, 14), False, {})
        sf_mat = agent.get_sf_matrix()
        w_vec = agent.get_w_vector()

        agent.reset(reset_sf=False, reset_w=False)
        self.assertTrue(np.all(agent.get_sf_matrix() == sf_mat))
        self.assertTrue(np.all(agent.get_w_vector() == w_vec))

    def test_reset_1(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        agent = rp.utils.SFLearning(
            num_states=14,
            num_actions=3,
            learning_rate_sf=0.1,
            learning_rate_reward=0.2,
            gamma=0.9,
            init_sf_mat=np.eye(3 * 14, dtype=np.float32),
            init_w_vec=np.zeros(3 * 14, dtype=np.float32)
        )
        self.assertEqual(agent.get_error_avg(), 0.)

        for _ in range(10):
            agent.update_transition(rl.one_hot(0, 14), 0, 0, rl.one_hot(0, 14), False, {})
        sf_mat = agent.get_sf_matrix()

        agent.reset(reset_sf=False, reset_w=True)
        self.assertTrue(np.all(agent.get_sf_matrix() == sf_mat))
        self.assertTrue(np.all(agent.get_w_vector() == np.zeros(3 * 14, dtype=np.float32)))

    def test_reset_2(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        agent = rp.utils.SFLearning(
            num_states=14,
            num_actions=3,
            learning_rate_sf=0.1,
            learning_rate_reward=0.2,
            gamma=0.9,
            init_sf_mat=np.eye(3 * 14, dtype=np.float32),
            init_w_vec=np.zeros(3 * 14, dtype=np.float32)
        )
        self.assertEqual(agent.get_error_avg(), 0.)

        for _ in range(10):
            agent.update_transition(rl.one_hot(0, 14), 0, 0, rl.one_hot(0, 14), False, {})
        w_vec = agent.get_w_vector()

        agent.reset(reset_sf=True, reset_w=False)
        self.assertTrue(np.all(agent.get_sf_matrix() == np.eye(3 * 14, dtype=np.float32)))
        self.assertTrue(np.all(agent.get_w_vector() == w_vec))

    def test_reset_3(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        agent = rp.utils.SFLearning(
            num_states=14,
            num_actions=3,
            learning_rate_sf=0.1,
            learning_rate_reward=0.2,
            gamma=0.9,
            init_sf_mat=np.eye(3 * 14, dtype=np.float32),
            init_w_vec=np.zeros(3 * 14, dtype=np.float32)
        )
        self.assertEqual(agent.get_error_avg(), 0.)

        for _ in range(10):
            agent.update_transition(rl.one_hot(0, 14), 0, 0, rl.one_hot(0, 14), False, {})

        agent.reset(reset_sf=True, reset_w=True)
        self.assertTrue(np.all(agent.get_sf_matrix() == np.eye(3 * 14, dtype=np.float32)))
        self.assertTrue(np.all(agent.get_w_vector() == np.zeros(3 * 14, dtype=np.float32)))

    def test_convergence(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np
        from itertools import product

        mdp = rp.mdp.ColumnWorld2(slip_prob=0.)
        t_mat, r_vec = mdp.get_t_mat_r_vec()
        q_star, _ = rl.algorithm.vi(t_mat, r_vec, gamma=0.9)
        pi_star = np.argmax(q_star, axis=0)

        num_act = mdp.num_actions()
        num_states = mdp.num_states()

        p_mat = np.zeros([num_act * num_states, num_act * num_states], dtype=np.float32)
        for a, s in product(range(num_act), range(num_states)):
            sn = np.where(t_mat[a, s] == 1)[0][0]
            an = pi_star[sn]
            p_mat[num_states * a + s] = rl.one_hot(num_states * an + sn, num_states * num_act)
        psi_mat = np.linalg.pinv(np.eye(num_act * num_states) - 0.9 * p_mat)
        w_vec = np.reshape(r_vec, -1)
        q_flat = np.matmul(psi_mat, w_vec)
        self.assertLessEqual(np.linalg.norm(q_flat - np.reshape(q_star, -1), ord=np.inf), 1e-4)

        agent = rp.utils.SFLearning(
            num_states=num_states,
            num_actions=num_act,
            learning_rate_sf=0.1,
            learning_rate_reward=0.2,
            gamma=0.9,
            init_sf_mat=psi_mat,
            init_w_vec=w_vec
        )
        self.assertLessEqual(np.max(np.abs(agent.get_q_vector() - q_star)), 1e-4)
        s = rl.one_hot(0, num_states)
        sn = np.matmul(s, t_mat[0])
        for _ in range(10):
            agent.update_transition(s, 0, r_vec[0, 0], sn, False, {})
        agent.on_simulation_timeout()
        self.assertLessEqual(np.max(np.abs(agent.get_q_vector() - q_star)), 1e-4)

    def test_get_error_avg(self):
        import rewardpredictive as rp
        import rlutils as rl
        import numpy as np

        agent = rp.utils.SFLearning(
            num_states=14,
            num_actions=3,
            learning_rate_sf=0.1,
            learning_rate_reward=0.2,
            gamma=0.9,
            init_sf_mat=np.eye(3 * 14, dtype=np.float32),
            init_w_vec=np.zeros(3 * 14, dtype=np.float32)
        )
        self.assertEqual(agent.get_error_avg(), 0.)

        s = rl.one_hot(0, 14)
        err_dict_list = [agent.update_transition(s, 0, 0, s, False, {}) for _ in range(10)]
        err_avg = np.mean([np.linalg.norm(d['sf_error']) + abs(d['r_error']) for d in err_dict_list])
        self.assertEqual(agent.get_error_avg(), err_avg)
