#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from itertools import product

import numpy as np
import rlutils as rl
import tensorflow as tf


def set_seeds(seed):
    rl.set_seeds(seed)
    tf.set_random_seed(seed)


def simulate_episodes(task, policy, transition_listener, num_episodes, max_steps=2000):
    for i in range(num_episodes):
        rl.data.simulate_gracefully(task, policy, transition_listener, max_steps=max_steps)


def pad_list_of_list_to_ndarray(ar, pad_value=np.nan, dtype=np.float32, row_length=0):
    inner_len = max(np.max([len(a) for a in ar]), row_length)
    nd_ar = np.ones([len(ar), inner_len], dtype=dtype) * pad_value
    for i, a in enumerate(ar):
        nd_ar[i, :len(a)] = a
    return nd_ar


def lam_from_mat_visitation_counts(t_mat, r_vec, phi_mat, visitation_counts):
    num_actions = np.shape(t_mat)[0]
    visit_mask = np.array(visitation_counts > 0, dtype=np.float32)
    phi_mat_pinv = np.linalg.pinv(phi_mat)

    m_mat = []
    w_vec = []
    for a in range(num_actions):
        visit_mask_a = np.reshape(visit_mask[a], [-1, 1])
        phi_mat_masked = phi_mat * visit_mask_a
        phi_mat_masked_pinv = np.linalg.pinv(phi_mat_masked)
        visit_mask_latent_a = np.array(np.matmul(visit_mask[a], phi_mat) > 0., dtype=np.float32)
        visit_mask_latent_a_1 = np.reshape(visit_mask_latent_a, [-1, 1])

        m_a = np.matmul(phi_mat_masked_pinv, np.matmul(t_mat[a], phi_mat))
        m_a_reg = np.matmul(phi_mat_pinv, np.matmul(t_mat[a], phi_mat))
        m_mat.append(visit_mask_latent_a_1 * m_a + (1. - visit_mask_latent_a_1) * m_a_reg)

        w_a = np.matmul(phi_mat_masked_pinv, r_vec[a])
        w_a_reg = np.matmul(phi_mat_pinv, r_vec[a])
        w_vec.append(visit_mask_latent_a * w_a + (1. - visit_mask_latent_a) * w_a_reg)

    m_mat = np.stack(m_mat)
    w_vec = np.stack(w_vec)

    return m_mat, w_vec


def lam_from_mat(t_mat, r_vec, phi_mat):
    """
    Compute LAM from matrix representations of the transition table, expected reward vectors, and state representation
    matrix.

    :param t_mat:
    :param r_vec:
    :param phi_mat:
    :return: m_mat, w_vec
    """
    num_actions = np.shape(t_mat)[0]
    phi_mat_pinv = np.linalg.pinv(phi_mat)

    w_vec = np.stack([np.matmul(phi_mat_pinv, r_vec[a]) for a in range(num_actions)])
    m_mat = np.stack([np.matmul(phi_mat_pinv, np.matmul(t_mat[a], phi_mat)) for a in range(num_actions)])

    return m_mat, w_vec


class TableModel(rl.data.TransitionListener):
    def __init__(self, num_states, num_actions, reward_sampler=None, max_reward=1.0):
        self._reward_sampler = reward_sampler

        self._t_cnt = np.zeros([num_actions, num_states, num_states]).astype(np.int)
        self._r_sum = np.zeros([num_actions, num_states]).astype(np.float32)
        self._v_cnt = np.zeros([num_actions, num_states]).astype(np.int)
        self._q_vec = np.zeros([num_actions, num_states]).astype(np.float32)

        if reward_sampler is None:
            self._r_default = np.ones([num_actions, num_states], dtype=np.float32) * max_reward
        else:
            self._r_default = reward_sampler([num_actions, num_states]).astype(np.float32)
        self._update_cnt = 0
        self._max_reward = max_reward

    def reset(self):
        self._t_cnt = np.zeros(np.shape(self._t_cnt)).astype(np.int)
        self._r_sum = np.zeros(np.shape(self._r_sum)).astype(np.float32)
        self._v_cnt = np.zeros(np.shape(self._v_cnt)).astype(np.int)
        self._q_vec = np.zeros(np.shape(self._q_vec)).astype(np.float32)

    def update_transition(self, s, a, r, s_next, t, info):
        si = np.where(s == 1)[0][0]
        sni = np.where(s_next == 1)[0][0]

        self._t_cnt[a, si, sni] += 1
        self._r_sum[a, si] += r
        self._v_cnt[a, si] += 1

    def on_simulation_timeout(self):
        pass

    def get_max_reward(self):
        return self._max_reward

    def num_states(self):
        return np.shape(self._q_vec)[1]

    def num_actions(self):
        return np.shape(self._q_vec)[0]

    def visitation_counts(self):
        return self._v_cnt

    def get_t_mat_r_vec(self):
        num_actions, num_states = np.shape(self._v_cnt)
        t_mat = np.zeros(np.shape(self._t_cnt), dtype=np.float32)
        r_vec = np.zeros(np.shape(self._r_sum), dtype=np.float32)

        for a, i in product(range(num_actions), range(num_states)):
            if self._v_cnt[a, i] == 0:
                t_mat[a, i] = rl.one_hot(i, num_states)
                r_vec[a, i] = self._r_default[a, i]
            else:
                t_mat[a, i] = self._t_cnt[a, i] / self._v_cnt[a, i]
                r_vec[a, i] = self._r_sum[a, i] / self._v_cnt[a, i]
        return t_mat, r_vec


class SFLearning(rl.agent.Agent):
    def __init__(self,
                 num_states,
                 num_actions,
                 learning_rate_sf,
                 learning_rate_reward,
                 gamma=0.9,
                 init_sf_mat=None,
                 init_w_vec=None):
        self._num_actions = num_actions
        self._basis_fn = rl.basisfunction.ActionTiledBasisFunction(num_states, num_actions, dtype=np.float32)
        self._lr_sf = learning_rate_sf
        self._lr_r = learning_rate_reward
        self._gamma = gamma
        if init_sf_mat is None:
            init_sf_mat = np.zeros([num_states * num_actions, num_states * num_actions], dtype=np.float32)
        self._init_sf_mat = init_sf_mat
        if init_w_vec is None:
            init_w_vec = np.zeros(num_states * num_actions, dtype=np.float32)
        self._init_w_vec = init_w_vec

        self._sf = np.array(self._init_sf_mat, copy=True)
        self._w = np.array(self._init_w_vec, copy=True)

        self._error_avg = 0.
        self._step_cnt = 0

    def reset(self, reset_sf=True, reset_w=True):
        if reset_sf:
            self._sf = np.array(self._init_sf_mat, copy=True)
        if reset_w:
            self._w = np.array(self._init_w_vec, copy=True)
        self._error_avg = 0.
        self._step_cnt = 0

    def q_values(self, state):
        q_vec = np.matmul(self._sf, self._w)
        return np.array([np.dot(self._basis_fn(state, a), q_vec) for a in range(self._num_actions)], dtype=np.float32)

    def update_transition(self, state, action, reward, next_state, term, info):
        a_star_next = np.argmax(self.q_values(next_state))

        phi = self._basis_fn(state, action)
        phi_next = self._basis_fn(next_state, a_star_next)

        sf_target = phi + (1. - term) * self._gamma * np.matmul(phi_next, self._sf)
        sf_error = sf_target - np.matmul(phi, self._sf)
        self._sf = self._sf + self._lr_sf * np.outer(phi, sf_error)

        r_error = reward - np.dot(self._w, phi)
        self._w = self._w + self._lr_r * r_error * phi

        err = np.linalg.norm(sf_error) + abs(r_error)
        self._error_avg = self._error_avg * (self._step_cnt / (self._step_cnt + 1.)) + err / (self._step_cnt + 1.)
        self._step_cnt += 1

        return {'sf_error': sf_error, 'r_error': r_error}

    def on_simulation_timeout(self):
        pass

    def get_q_vector(self):
        return np.reshape(np.matmul(self._sf, self._w), [self._num_actions, -1])

    def get_sf_matrix(self):
        return self._sf

    def get_w_vector(self):
        return self._w

    def get_learning_rate_sf(self):
        return self._lr_sf

    def get_learning_rate_reward(self):
        return self._lr_r

    def get_gamma(self):
        return self._gamma

    def get_error_avg(self):
        return self._error_avg


def cluster_idx_to_phi_mat(cluster_idx):
    num_states = len(cluster_idx)
    num_cluster = len(np.unique(cluster_idx))
    phi_mat = np.zeros([num_states, num_cluster])
    for i, c in enumerate(cluster_idx):
        phi_mat[i, c] = 1.
    return phi_mat


def reward_rollout(t_mat, r_vec, s, action_seq):
    '''
    Computes an expected reward rollout on a tabulated representation of an MDP.

    :param t_mat:
    :param r_vec:
    :param s:
    :param action_seq:
    :return:
    '''
    rew_list = []
    for a in action_seq:
        rew_list.append(np.dot(s, r_vec[a]))
        s = np.matmul(s, t_mat[a])
    return np.array(rew_list)


def lam_m_mat_from_sf_mat(f_mat, gamma=0.9):
    '''
    Compute LSFM for a uniform random action selection policy from a LAM.

    :param m_mat:
    :param gamma:
    :return:
    '''
    num_actions, num_features, _ = np.shape(f_mat)
    id_mat = np.eye(num_features).astype(f_mat.dtype)
    f_mat_pinv = np.linalg.pinv(np.mean(f_mat, axis=0))
    m_mat = np.stack([np.matmul(f_mat[a] - id_mat, f_mat_pinv) / gamma for a in range(num_actions)])
    return m_mat.astype(f_mat.dtype)


def sf_mat_from_lam(m_mat, gamma=0.9):
    '''
    Compute LSFM for a uniform random action selection policy from a LAM.

    :param m_mat:
    :param gamma:
    :return:
    '''
    num_actions, num_features, _ = np.shape(m_mat)
    id_mat = np.eye(num_features).astype(m_mat.dtype)
    f_mat_mean = np.linalg.pinv(id_mat - gamma * np.mean(m_mat, axis=0))
    f_mat = np.stack([id_mat + gamma * np.matmul(m_mat[a], f_mat_mean) for a in range(num_actions)])
    return f_mat


def get_trajectory_list(experiment, repeat):
    s_buffer = experiment.results['s_buffer'][repeat]
    a_buffer = experiment.results['a_buffer'][repeat]
    r_buffer = experiment.results['r_buffer'][repeat]
    sn_buffer = experiment.results['sn_buffer'][repeat]
    t_buffer = experiment.results['t_buffer'][repeat]

    traj_list = [rl.data.TransitionBuffer()]
    for s, a, r, sn, t in zip(s_buffer, a_buffer, r_buffer, sn_buffer, t_buffer):
        traj_list[-1].update_transition(s, a, r, sn, t, {})
        if t:
            traj_list.append(rl.data.TransitionBuffer())
    traj_list = traj_list[:-1]
    return traj_list


class EGreedyScheduleUpdate(rl.data.TransitionListener):

    def __init__(self, policy: rl.policy.EGreedyPolicy, schedule: rl.schedule.VariableSchedule):
        self._policy = policy
        self._schedule = schedule
        self._ep_idx = 0

    def update_transition(self, s, a, r, s_next, t, info):
        if t:
            self.on_simulation_timeout()

    def on_simulation_timeout(self):
        self._ep_idx += 1
        self._policy.set_epsilon(self._schedule(self._ep_idx))


class TransitionListenerAgentDecorator(rl.agent.Agent):
    def __init__(self, agent: rl.agent.Agent, transition_listener_list):
        self.agent = agent
        self.transition_listener_list = transition_listener_list

    def q_values(self, state):
        return self.agent.q_values(state)

    def reset(self, *params, **kwargs):
        self.agent.reset(*params, **kwargs)

    def update_transition(self, s, a, r, s_next, t, info):
        for listener in [self.agent] + self.transition_listener_list:
            listener.update_transition(s, a, r, s_next, t, info)

    def on_simulation_timeout(self):
        for listener in [self.agent] + self.transition_listener_list:
            listener.on_simulation_timeout()
