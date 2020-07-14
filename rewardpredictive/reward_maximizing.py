#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from abc import abstractmethod

import numpy as np
import rlutils as rl
from sklearn.cluster import AgglomerativeClustering

from .utils import cluster_idx_to_phi_mat
from .belief_set import BeliefSetHypothesisLearningAgent, BeliefSetOracleLearningAgent
from .utils import TableModel
from .utils import lam_from_mat_visitation_counts


def _score_model(phi_mat, table_model, gamma):
    t_mat, r_vec = table_model.get_t_mat_r_vec()
    visitation_counts = table_model.visitation_counts()

    return reward_maximizing_score(phi_mat, t_mat, r_vec, visitation_counts, gamma)


def reward_maximizing_score(phi_mat, t_mat, r_vec, visitation_counts, gamma):
    m_mat, w_vec = lam_from_mat_visitation_counts(t_mat, r_vec, phi_mat, visitation_counts)

    q_phi, _ = rl.algorithm.vi(m_mat, w_vec, gamma)
    q_hat = np.matmul(phi_mat, q_phi.transpose()).transpose()
    pi_hat = np.argmax(q_hat, axis=0)
    target_op = lambda q: np.array([q[a, i] for i, a in enumerate(pi_hat)], dtype=np.float32)
    _, v_hat = rl.algorithm.vi(t_mat, r_vec, gamma, target_op=target_op)

    _, v_star = rl.algorithm.vi(t_mat, r_vec, gamma)

    return - np.max(np.abs(v_star - v_hat))


def reward_maximizing_score_tabular(phi_mat, t_mat, r_vec, gamma):
    num_a, num_s = np.shape(r_vec)
    vc = np.ones([num_a, num_s], dtype=np.float32)
    return reward_maximizing_score(phi_mat, t_mat, r_vec, vc, gamma)



def reward_maximizing_representation(t_mat, r_vec, num_latent_states, gamma):
    q_hat, _ = rl.algorithm.vi(t_mat, r_vec, gamma=gamma)
    cluster_idx = AgglomerativeClustering(n_clusters=num_latent_states).fit(q_hat.transpose()).labels_
    return cluster_idx_to_phi_mat(cluster_idx)


class RewardMaximizingLearningAgent(BeliefSetHypothesisLearningAgent):

    def __init__(self,
                 phi_mat,
                 num_actions,
                 agent,
                 reward_range=[0., 1.],
                 gamma=0.9):
        super().__init__()
        self._phi_mat = np.array(phi_mat, copy=True, dtype=np.float32)
        self._phi_mat.flags.writeable = False
        self._score = 0
        self._gamma = gamma

        self._phi = lambda s: np.matmul(s, self._phi_mat)
        num_states, num_latent_states = np.shape(phi_mat)
        self._agent = agent
        self._table = TableModel(
            num_states=num_states,
            num_actions=num_actions,
            reward_sampler=lambda s: np.ones(s, dtype=np.float32) * reward_range[0],
            max_reward=reward_range[1]
        )

    def get_score(self) -> float:
        assert (self._score <= 0.)
        return self._score

    def to_ndarray(self) -> np.ndarray:
        return np.array(self._phi_mat, copy=True, dtype=np.float32)

    def q_values(self, state):
        return self._agent.q_values(state)

    def reset(self, *params, **kwargs):
        self._agent.reset()
        self._table.reset()
        self._score = 0.

    def _update_score(self):
        self._score = _score_model(
            phi_mat=self._phi_mat,
            table_model=self._table,
            gamma=self._gamma,
        )

    def update_transition(self, s, a, r, s_next, t, info):
        self._agent.update_transition(s, a, r, s_next, t, info)
        self._table.update_transition(s, a, r, s_next, t, info)
        if t:
            self._update_score()

    def on_simulation_timeout(self):
        self._update_score()


class RewardMaximizingQLearningAgent(RewardMaximizingLearningAgent):
    def __init__(self,
                 phi_mat,
                 num_actions,
                 learning_rate=0.9,
                 init_v=1.,
                 reward_range=[0., 1.],
                 gamma=0.9):
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
            reward_range=reward_range,
            gamma=gamma
        )


class RewardMaximizingLearningOracle(BeliefSetOracleLearningAgent):

    def __init__(self,
                 num_states,
                 num_actions,
                 num_latent_states,
                 agent,
                 init_v=1.,
                 gamma=0.9,
                 reward_range=[0., 1.]):
        super().__init__()
        self._init_v = init_v
        self._agent = agent
        self._num_latent_states = num_latent_states

        self._table = TableModel(
            num_states=num_states,
            num_actions=num_actions,
            reward_sampler=lambda s: np.ones(s, dtype=np.float32) * reward_range[0],
            max_reward=reward_range[1]
        )
        self._init_v = init_v
        self._gamma = gamma
        self._score = 0

    def get_num_latent_states(self):
        return self._num_latent_states

    @abstractmethod
    def _create_learning_agent(self, phi_mat):
        pass

    def generate_entry(self) -> (BeliefSetHypothesisLearningAgent, float):
        phi_mat = reward_maximizing_representation(*self._table.get_t_mat_r_vec(), self._num_latent_states, self._gamma)
        hyp = self._create_learning_agent(phi_mat)
        self._score = _score_model(phi_mat, self._table, self._gamma)
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


class RewardMaximizingQLearningOracle(RewardMaximizingLearningOracle):

    def __init__(self,
                 num_states,
                 num_actions,
                 num_latent_states,
                 learning_rate=0.9,
                 init_v=1.,
                 gamma=0.9,
                 reward_range=[0., 1.]):
        self._agent = rl.agent.QLearning(num_states, num_actions, learning_rate, gamma=gamma, init_Q=init_v)
        self._reward_range = reward_range
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            num_latent_states=num_latent_states,
            agent=self._agent,
            init_v=init_v,
            gamma=gamma,
            reward_range=reward_range
        )

    def _create_learning_agent(self, phi_mat):
        return RewardMaximizingQLearningAgent(
            phi_mat=phi_mat,
            num_actions=self._table.num_actions(),
            learning_rate=self._agent.get_learning_rate(),
            init_v=self._init_v,
            reward_range=self._reward_range,
            gamma=self._gamma
        )
