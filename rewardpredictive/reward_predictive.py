#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
import multiprocessing as mp
from abc import abstractmethod

import numpy as np
import rlutils as rl
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering

from .utils import cluster_idx_to_phi_mat, lam_from_mat
from .belief_set import BeliefSetHypothesisLearningAgent, BeliefSetOracleLearningAgent
from .utils import TableModel, SFLearning, lam_from_mat_visitation_counts, lam_m_mat_from_sf_mat, sf_mat_from_lam


class LSFMatrixModel(object):
    def __init__(self, session, num_actions, num_states, num_features,
                 gamma=0.9,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                 alpha_r=1.,
                 alpha_sf=.001,
                 alpha_reg=0.,
                 init_phi_mat=lambda s: tf.random_uniform(s, minval=0., maxval=1., dtype=tf.float32),
                 init_f_mat=lambda s: tf.zeros(s, dtype=tf.float32),
                 init_w_vec=lambda s: tf.zeros(s, dtype=tf.float32),
                 fixed_target=False):
        '''
        LSFM Matrix implementation.

        :param session:
        :param num_actions:
        :param num_states:
        :param num_features:
        :param gamma:
        :param optimizer:
        :param alpha_r:
        :param alpha_sf:
        :param init_phi_mat:
        :param init_f_mat:
        :param init_w_vec:
        '''
        self._num_actions = num_actions
        self._num_states = num_states
        self._num_features = num_features

        self._session = session

        self._t_mat_ph = tf.placeholder(tf.float32, [num_actions, num_states, num_states], name='t_mat_ph')
        self._r_vec_ph = tf.placeholder(tf.float32, [num_actions, num_states], name='r_vec_ph')

        self._phi_mat = tf.Variable(init_phi_mat([num_states, num_features]), name='phi_mat')
        self._w_vec = tf.Variable(init_w_vec([num_actions, num_features]), name='w_vec')
        self._f_mat = tf.Variable(init_f_mat([num_actions, num_features, num_features]), name='f_mat')

        psi_mat = tf.stack([tf.matmul(self._phi_mat, self._f_mat[a]) for a in range(num_actions)])
        psi_bar_mat = tf.reduce_mean(psi_mat, axis=0)

        sf_target = [self._phi_mat + gamma * tf.matmul(self._t_mat_ph[a], psi_bar_mat) for a in range(num_actions)]
        sf_target = tf.stack(sf_target)
        if fixed_target:
            sf_target = tf.stop_gradient(sf_target)
        error_sf = sf_target - psi_mat

        self._loss_sf = tf.reduce_sum(tf.square(error_sf), axis=-1)

        r_pred = tf.transpose(tf.matmul(self._phi_mat, tf.transpose(self._w_vec)))
        self._loss_r = tf.square(r_pred - self._r_vec_ph)

        self._loss_reg = tf.square(tf.reduce_sum(self._phi_mat * self._phi_mat, axis=-1) - tf.reshape(1., [-1, 1]))

        self._loss = alpha_r * self._loss_r + alpha_sf * self._loss_sf + alpha_reg * self._loss_reg
        self._loss_m = tf.reduce_mean(self._loss)
        self._train_op = optimizer.minimize(self._loss_m)

        self._phi_mat_ph = tf.placeholder(tf.float32, [num_states, num_features], name='phi_mat_ph')
        self._phi_mat_update = tf.assign(self._phi_mat, self._phi_mat_ph)

    def _get_feed_dict(self, t_mat, r_vec):
        return {self._t_mat_ph: t_mat, self._r_vec_ph: r_vec}

    def get_num_actions(self):
        return self._num_actions

    def get_num_states(self):
        return self._num_states

    def get_num_features(self):
        return self._num_features

    def set_phi_mat(self, phi_mat):
        self._session.run(self._phi_mat_update, feed_dict={self._phi_mat_ph: phi_mat})

    def train(self, t_mat, r_vec):
        return self._session.run([self._train_op, self._loss_m], feed_dict=self._get_feed_dict(t_mat, r_vec))[1]

    def get_loss_vec(self, t_mat, r_vec):
        return self._session.run(self._loss, feed_dict=self._get_feed_dict(t_mat, r_vec))

    def loss_vec_r(self, t_mat, r_vec):
        return self._session.run(self._loss_r, feed_dict=self._get_feed_dict(t_mat, r_vec))

    def loss_vec_sf(self, t_mat, r_vec):
        return self._session.run(self._loss_sf, feed_dict=self._get_feed_dict(t_mat, r_vec))

    def loss_vec_reg(self, t_mat, r_vec):
        return self._session.run(self._loss_reg, feed_dict=self._get_feed_dict(t_mat, r_vec))

    def get_phi_mat(self):
        return self._session.run(self._phi_mat)

    def get_f_mat(self):
        return self._session.run(self._f_mat)

    def get_w_vec(self):
        return self._session.run(self._w_vec)

    def exp_reward_error_bound(self, t_mat, r_vec, t_steps=200):
        eps_r = np.max(self.loss_vec_r(t_mat, r_vec))
        eps_sf = np.max(self.loss_vec_sf(t_mat, r_vec))

        f_mat = self.get_f_mat()
        m_mat = lam_m_mat_from_sf_mat(f_mat, self.gamma)
        w_vec = self.get_w_vec()

        m_norm = np.max([np.linalg.norm(m) for m in m_mat])
        w_norm = np.max([np.linalg.norm(w) for w in w_vec])

        m_sum = np.sum(np.power(m_norm, np.arange(1, t_steps)))
        return eps_sf * w_norm * (1 + self.gamma * m_norm) * m_sum / self.gamma + eps_r


class LSFMRepresentationLearner(object):
    def __init__(self,
                 num_states,
                 num_actions,
                 num_latent_states,
                 gamma,
                 num_training_iterations=1000,
                 log_interval=100,
                 learning_rate=1e-2,
                 alpha_r=1.,
                 alpha_sf=.01,
                 alpha_reg=0.):
        self._session = tf.Session(
            config=tf.ConfigProto(
                device_count={"CPU": mp.cpu_count()},
                inter_op_parallelism_threads=mp.cpu_count(),
                intra_op_parallelism_threads=mp.cpu_count()
            )
        )
        self._model = LSFMatrixModel(
            self._session,
            num_actions,
            num_states,
            num_latent_states,
            gamma=gamma,
            optimizer=tf.train.AdamOptimizer(learning_rate),
            alpha_r=alpha_r,
            alpha_sf=alpha_sf,
            alpha_reg=alpha_reg,
            init_phi_mat=lambda s: tf.random_uniform(s, minval=0., maxval=1., dtype=tf.float32),
            init_f_mat=lambda s: tf.zeros(s, dtype=tf.float32),
            init_w_vec=lambda s: tf.zeros(s, dtype=tf.float32),
            fixed_target=True
        )
        self._num_training_iterations = num_training_iterations
        self._log_interval = log_interval
        self._loss_list = []

    def learn_representation(self, t_mat, r_vec):
        self._loss_list = []
        self._session.run(tf.global_variables_initializer())
        for i in range(self._num_training_iterations):
            self._loss_list.append(self._model.train(t_mat, r_vec))
            if (i + 1) % self._log_interval == 0:
                print('It {:5d}: loss={:1.5e} '.format(i + 1, self._loss_list[-1]))
        phi_mat_learned = self._model.get_phi_mat()
        num_latent_states = np.shape(phi_mat_learned)[1]
        cluster_idx = AgglomerativeClustering(n_clusters=num_latent_states).fit(phi_mat_learned).labels_
        return cluster_idx_to_phi_mat(cluster_idx)

    def get_loss_list(self):
        return np.array(self._loss_list)

    def get_num_training_iterations(self):
        return self._num_training_iterations

    def get_num_latent_states(self):
        return self._model.get_num_features()


def _score_model(phi_mat, table_model, gamma, alpha):
    t_mat, r_vec = table_model.get_t_mat_r_vec()
    visitation_counts = table_model.visitation_counts()
    return reward_predictive_score(phi_mat, t_mat, r_vec, visitation_counts, gamma, alpha)


def reward_predictive_score(phi_mat, t_mat, r_vec, visitation_counts, gamma, alpha):
    m_mat, w_vec = lam_from_mat_visitation_counts(t_mat, r_vec, phi_mat, visitation_counts)
    num_actions, _, _ = np.shape(m_mat)
    f_mat = sf_mat_from_lam(m_mat, gamma)

    err_r = np.abs(np.matmul(phi_mat, w_vec.transpose()).transpose() - r_vec)

    psi_mat = np.stack([np.matmul(phi_mat, f_mat[a]) for a in range(num_actions)])
    psi_bar = np.mean(psi_mat, axis=0)
    psi_target = np.stack([phi_mat + gamma * np.matmul(t_mat[a], psi_bar) for a in range(num_actions)])
    err_sf = np.linalg.norm(psi_mat - psi_target, axis=-1)

    cnt_mask = np.array(visitation_counts >= 1.0, dtype=np.float32)
    if np.sum(cnt_mask > 0):
        cnt_mask = cnt_mask / np.sum(cnt_mask)
        score = - np.sum(cnt_mask * (err_sf * alpha + err_r))
    else:
        score = 0
    return score


def reward_predictive_score_tabular(phi_mat, t_mat, r_vec, gamma, alpha=0.1):
    m_mat, w_vec = lam_from_mat(t_mat, r_vec, phi_mat)
    num_actions, _, _ = np.shape(m_mat)
    f_mat = sf_mat_from_lam(m_mat, gamma)

    err_r = np.abs(np.matmul(phi_mat, w_vec.transpose()).transpose() - r_vec)

    psi_mat = np.stack([np.matmul(phi_mat, f_mat[a]) for a in range(num_actions)])
    psi_bar = np.mean(psi_mat, axis=0)
    psi_target = np.stack([phi_mat + gamma * np.matmul(t_mat[a], psi_bar) for a in range(num_actions)])
    err_sf = np.linalg.norm(psi_mat - psi_target, ord=2, axis=-1)

    return - (np.max(err_r) + alpha * np.max(err_sf))


class RewardPredictiveLearningAgent(BeliefSetHypothesisLearningAgent):

    def __init__(self,
                 phi_mat,
                 num_actions,
                 agent,
                 max_reward=1.,
                 gamma=0.9,
                 score_alpha=1.0):
        super().__init__()
        self._phi_mat = np.array(phi_mat, copy=True, dtype=np.float32)
        self._phi_mat.flags.writeable = False
        self._score_alpha = score_alpha
        self._score = 0
        self._gamma = gamma

        self._phi = lambda s: np.matmul(s, self._phi_mat)
        num_states, num_latent_states = np.shape(phi_mat)
        self._agent = agent
        self._table = TableModel(
            num_states=num_states,
            num_actions=num_actions,
            reward_sampler=None,
            max_reward=max_reward
        )

    def get_score(self) -> float:
        assert (self._score <= 0.)
        return self._score

    def to_ndarray(self) -> np.ndarray:
        return np.array(self._phi_mat, copy=True, dtype=np.float32)

    def q_values(self, state):
        return self._agent.q_values(state)

    def get_table_model(self):
        return self._table

    def reset(self, *params, **kwargs):
        self._agent.reset()
        self._table.reset()
        self._score = 0.

    def _update_score(self):
        self._score = _score_model(
            phi_mat=self._phi_mat,
            table_model=self._table,
            gamma=self._gamma,
            alpha=self._score_alpha
        )

    def update_transition(self, s, a, r, s_next, t, info):
        self._agent.update_transition(s, a, r, s_next, t, info)
        self._table.update_transition(s, a, r, s_next, t, info)
        if t:
            self._update_score()

    def on_simulation_timeout(self):
        self._update_score()


class RewardPredictiveQLearningAgent(RewardPredictiveLearningAgent):
    def __init__(self,
                 phi_mat,
                 num_actions,
                 learning_rate=0.9,
                 init_v=1.,
                 max_reward=1.,
                 gamma=0.9,
                 score_alpha=1.0):
        _, num_latent_states = np.shape(phi_mat)
        latent_agent = rl.agent.QLearning(
            num_states=num_latent_states,
            num_actions=num_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            init_Q=init_v
        )
        agent = rl.agent.StateRepresentationWrapperAgent(
            agent=latent_agent,
            phi=lambda s: np.matmul(s, phi_mat)
        )
        super().__init__(
            phi_mat=phi_mat,
            num_actions=num_actions,
            agent=agent,
            max_reward=max_reward,
            gamma=gamma,
            score_alpha=score_alpha
        )


class RewardPredictiveSFLearningAgent(RewardPredictiveLearningAgent):
    def __init__(self,
                 phi_mat,
                 num_actions,
                 learning_rate_sf=0.9,
                 learning_rate_reward=0.9,
                 init_v=1.,
                 max_reward=1.,
                 gamma=0.9,
                 score_alpha=1.0):
        _, num_latent_states = np.shape(phi_mat)
        init_sf_mat = np.eye(num_latent_states * num_actions, dtype=np.float32)
        init_w_vec = np.ones(num_latent_states * num_actions, dtype=np.float32) * init_v
        self._latent_agent = SFLearning(
            num_states=num_latent_states,
            num_actions=num_actions,
            learning_rate_sf=learning_rate_sf,
            learning_rate_reward=learning_rate_reward,
            gamma=gamma,
            init_sf_mat=init_sf_mat,
            init_w_vec=init_w_vec
        )
        agent = rl.agent.StateRepresentationWrapperAgent(
            agent=self._latent_agent,
            phi=lambda s: np.matmul(s, phi_mat)
        )
        super().__init__(
            phi_mat=phi_mat,
            num_actions=num_actions,
            agent=agent,
            max_reward=max_reward,
            gamma=gamma,
            score_alpha=score_alpha
        )

    def reset(self, *params, **kwargs):
        self._agent.reset(reset_sf=True, reset_w=True)
        self._table.reset()
        self._score = 0


class RewardPredictiveLearningOracle(BeliefSetOracleLearningAgent):

    def __init__(self,
                 num_states,
                 num_actions,
                 agent,
                 lsfm_model: LSFMRepresentationLearner,
                 init_v=1.,
                 gamma=0.9,
                 max_reward=1.0,
                 score_alpha=1.0):
        super().__init__()
        self._init_v = init_v
        self._agent = agent

        self._table = TableModel(
            num_states=num_states,
            num_actions=num_actions,
            reward_sampler=None,
            max_reward=max_reward
        )
        self._lsfm_model = lsfm_model
        self._init_v = init_v
        self._gamma = gamma
        self._score_alpha = score_alpha
        self._score = 0

    def get_table_model(self):
        return self._table

    @abstractmethod
    def _create_learning_agent(self, phi_mat):
        pass

    def generate_entry(self) -> (BeliefSetHypothesisLearningAgent, float):
        phi_mat = self._lsfm_model.learn_representation(*self._table.get_t_mat_r_vec())
        hyp = self._create_learning_agent(phi_mat)
        self._score = _score_model(phi_mat, self._table, self._gamma, self._score_alpha)
        assert (self._score <= 0.)
        return hyp, self._score

    def get_score(self) -> float:
        assert (self._score <= 0.)
        return self._score

    def q_values(self, state):
        return self._agent.q_values(state)

    def reset(self, *params, **kwargs):
        self._agent.reset()
        self._table.reset()
        self._score = 0

    def update_transition(self, s, a, r, s_next, t, info):
        self._agent.update_transition(s, a, r, s_next, t, info)
        self._table.update_transition(s, a, r, s_next, t, info)

    def on_simulation_timeout(self):
        pass

    def get_init_v(self):
        return self._init_v

    def get_agent(self):
        return self._agent

    def get_score_alpha(self):
        return self._score_alpha

    def get_lsfm_model(self):
        return self._lsfm_model


class RewardPredictiveQLearningOracle(RewardPredictiveLearningOracle):

    def __init__(self,
                 num_states,
                 num_actions,
                 lsfm_model: LSFMRepresentationLearner,
                 learning_rate=0.9,
                 init_v=1.,
                 gamma=0.9,
                 max_reward=1.0,
                 score_alpha=1.0):
        self._agent = rl.agent.QLearning(num_states, num_actions, learning_rate, gamma=gamma, init_Q=init_v)
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            agent=self._agent,
            lsfm_model=lsfm_model,
            init_v=init_v,
            gamma=gamma,
            max_reward=max_reward,
            score_alpha=score_alpha
        )

    def _create_learning_agent(self, phi_mat):
        return RewardPredictiveQLearningAgent(
            phi_mat=phi_mat,
            num_actions=self._table.num_actions(),
            learning_rate=self._agent.get_learning_rate(),
            init_v=self._init_v,
            gamma=self._agent.get_gamma()
        )


class RewardPredictiveSFLearningOracle(RewardPredictiveLearningOracle):

    def __init__(self,
                 num_states,
                 num_actions,
                 lsfm_model: LSFMRepresentationLearner,
                 learning_rate_sf=0.9,
                 learning_rate_reward=0.9,
                 init_v=1.,
                 gamma=0.9,
                 max_reward=1.0,
                 score_alpha=1.0):
        init_sf_mat = np.eye(num_states * num_actions, dtype=np.float32)
        init_w_vec = np.ones(num_states * num_actions, dtype=np.float32) * init_v
        self._agent = SFLearning(
            num_states=num_states,
            num_actions=num_actions,
            learning_rate_sf=learning_rate_sf,
            learning_rate_reward=learning_rate_reward,
            gamma=gamma,
            init_sf_mat=init_sf_mat,
            init_w_vec=init_w_vec
        )
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            agent=self._agent,
            lsfm_model=lsfm_model,
            init_v=init_v,
            gamma=gamma,
            max_reward=max_reward,
            score_alpha=score_alpha
        )

        self._lr_sf = learning_rate_sf
        self._lr_r = learning_rate_reward
        self._gamma = gamma
        self._init_v = init_v

    def _create_learning_agent(self, phi_mat):
        return RewardPredictiveSFLearningAgent(
            phi_mat=phi_mat,
            num_actions=self._table.num_actions(),
            learning_rate_sf=self._lr_sf,
            learning_rate_reward=self._lr_r,
            init_v=self._init_v,
            gamma=self._gamma
        )

    def reset(self, *params, **kwargs):
        # super().reset(*params, **kwargs)
        self._agent.reset(reset_sf=True, reset_w=True)
        self._table.reset()
        self._score = 0
        self._episode_cnt = 0

    def get_learning_rate_sf(self):
        return self._lr_sf

    def get_learning_rate_reward(self):
        return self._lr_r

    def get_gamma(self):
        return self._gamma

    def get_init_v(self):
        return self._init_v
