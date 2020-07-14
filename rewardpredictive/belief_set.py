#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from abc import abstractmethod, ABC

import numpy as np
import rlutils as rl

from .enumerate_partitions import enumerate_n_partitions
from .utils import lam_from_mat, cluster_idx_to_phi_mat


class BeliefSetEntry(rl.agent.Agent):
    @abstractmethod
    def get_score(self) -> float:  # pragma: no cover
        pass

    def get_action_probabilities(self, state) -> np.ndarray:
        q_values = self.q_values(state)
        action_prob = np.array(q_values == np.max(q_values), dtype=np.float32)
        action_prob /= np.sum(action_prob)
        return action_prob


class BeliefSetHypothesis(BeliefSetEntry, ABC):
    def __init__(self):
        self._count = 0

    def increase_count(self):
        self._count += 1

    def get_count(self) -> int:
        return self._count

    @abstractmethod
    def to_ndarray(self) -> np.ndarray:  # pragma: no cover
        pass


class BeliefSetHypothesisLearningAgent(BeliefSetHypothesis, ABC):
    pass


class BeliefSetOracle(BeliefSetEntry, ABC):
    @abstractmethod
    def generate_entry(self) -> BeliefSetEntry:  # pragma: no cover
        pass

    @abstractmethod
    def reset_oracle(self):  # pragma: no cover
        pass


class BeliefSetOracleLearningAgent(BeliefSetEntry, ABC):
    @abstractmethod
    def generate_entry(self) -> (BeliefSetHypothesisLearningAgent, float):  # pragma: no cover
        pass


class MetaAgent(rl.agent.Agent, rl.policy.Policy):
    def __init__(self, oracle: BeliefSetOracleLearningAgent, alpha: float,
                 beta: float):
        self._oracle = oracle
        self._alpha = alpha
        self._beta = beta
        self._belief_set = []
        self._iteration = 0

    def get_alpha(self):
        return self._alpha

    def get_beta(self):
        return self._beta

    def get_belief_set(self):
        return list(self._belief_set)

    def update_transition(self, s, a, r, s_next, t, info):
        for e in self._belief_set + [self._oracle]:
            e.update_transition(s, a, r, s_next, t, info)

    def on_simulation_timeout(self):
        for e in self._belief_set + [self._oracle]:
            e.on_simulation_timeout()

    def get_prior_prob(self) -> np.ndarray:
        counts = [e.get_count() for e in self._belief_set]
        return chinese_restaurant_process_probabilities(counts, self._alpha,
                                                        self._iteration)

    def get_posterior_prob(self, oracle_score=None) -> np.ndarray:
        if oracle_score is None:
            oracle_score = self._oracle.get_score()
        score_values = [e.get_score() for e in self._belief_set]
        score_values.append(oracle_score)
        posterior_prob = softmax_posterior(score_values,
                                           self._beta) * self.get_prior_prob()
        posterior_prob = posterior_prob / np.sum(posterior_prob)
        return posterior_prob

    def update_belief_set(self):
        new_entry, score = self._oracle.generate_entry()
        posterior = self.get_posterior_prob(score)
        i = np.random.choice(np.arange(len(posterior)), p=posterior)
        if i == len(self._belief_set):
            self._belief_set.append(new_entry)
        self._belief_set[i].increase_count()
        self._iteration += 1

        for a in self._belief_set + [self._oracle]:
            a.reset()

        return posterior

    def q_values(self, state):
        q_vals_list = np.stack([b.q_values(state) for b in self._belief_set])
        prob_mat = np.reshape(self.get_posterior_prob(), [-1, 1])
        q_vals = np.sum(prob_mat * q_vals_list, axis=0)
        return q_vals

    def reset(self, *params, **kwargs):
        self._oracle.reset()
        self._belief_set = []
        self._iteration = 0

    def __call__(self, state):
        action_prob_list = np.stack([b.get_action_probabilities(state) for b in
                                     self._belief_set + [self._oracle]])
        posterior = np.reshape(self.get_posterior_prob(), [-1, 1])
        action_prob = np.sum(posterior * action_prob_list, axis=0)
        action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        return action

    def get_oracle(self):
        return self._oracle


def chinese_restaurant_process_probabilities(counts, alpha, iteration):
    assert (np.sum(counts) == iteration)
    if alpha == np.inf:
        prior = np.array([0.] * len(counts) + [1.], dtype=np.float32)
    else:
        prior = np.array([c / (iteration + alpha) for c in counts] + [alpha / (iteration + alpha)], dtype=np.float32)
    return prior


def softmax_posterior(score_list, beta):
    assert not np.any(np.isnan(score_list))
    assert not np.any(np.isinf(score_list))
    assert not np.any(np.isneginf(score_list))

    if beta == np.inf:
        opt_idx = np.where(score_list == np.max(score_list))[0]
        prob_vec = np.zeros(len(score_list), dtype=np.float32)
        for i in opt_idx:
            prob_vec[i] = 1.
    else:
        x = beta * np.array(score_list, dtype=np.float32)
        prob_vec = np.exp(x - np.max(x))
    prob_vec = prob_vec / np.sum(prob_vec)
    return prob_vec


class LoggerPriorEpisodic(rl.data.TransitionListener):

    def __init__(self, agent: MetaAgent):
        self._agent = agent
        self._prior_log = []

    def _log_prior(self):
        prob = self._agent.get_prior_prob()
        if len(prob) > 1:
            prob_sorted = np.zeros(len(prob), dtype=np.float32)
            prob_sorted[0] = prob[-1]
            prob_sorted[1:] = prob[:-1]
            prob = prob_sorted
        self._prior_log.append(prob)

    def get_prior_log(self):
        return np.array(self._prior_log)

    def update_transition(self, s, a, r, s_next, t, info):
        if t:
            self._log_prior()

    def on_simulation_timeout(self):
        self._log_prior()


class LoggerPosteriorEpisodic(rl.data.TransitionListener):
    def __init__(self, agent: MetaAgent):
        self._agent = agent
        self._posterior_log = []

    def _log_posterior(self):
        prob = self._agent.get_posterior_prob()
        if len(prob) > 1:
            prob_sorted = np.zeros(len(prob), dtype=np.float32)
            prob_sorted[0] = prob[-1]
            prob_sorted[1:] = prob[:-1]
            prob = prob_sorted
        self._posterior_log.append(prob)

    def get_posterior_log(self):
        return np.array(self._posterior_log)

    def update_transition(self, s, a, r, s_next, t, info):
        if t:
            self._log_posterior()

    def on_simulation_timeout(self):
        self._log_posterior()


class LoggerCountEpisodic(rl.data.TransitionListener):
    def __init__(self, agent: MetaAgent):
        self._agent = agent
        self._count_log = []

    def _log_count(self):
        cnts = [b.get_count() for b in self._agent.get_belief_set()]
        if len(cnts) > 1:
            cnts_sorted = np.zeros(len(cnts), dtype=np.float32)
            cnts_sorted[0] = cnts[-1]
            cnts_sorted[1:] = cnts[:-1]
            cnts = cnts_sorted
        self._count_log.append(cnts)

    def get_count_log(self):
        return np.array(self._count_log)

    def update_transition(self, s, a, r, s_next, t, info):
        if t:
            self._log_count()

    def on_simulation_timeout(self):
        self._log_count()


class LoggerScoreEpisodic(rl.data.TransitionListener):
    def __init__(self, agent: MetaAgent):
        self._agent = agent
        self._count_log = []

    def _log_score(self):
        scores = [b.get_score() for b in self._agent.get_belief_set()]
        if len(scores) > 1:
            scores_sorted = np.zeros(len(scores), dtype=np.float32)
            scores_sorted[0] = scores[-1]
            scores_sorted[1:] = scores[:-1]
            scores = scores_sorted
        self._count_log.append(scores)

    def get_score_log(self):
        return np.array(self._count_log)

    def update_transition(self, s, a, r, s_next, t, info):
        if t:
            self._log_score()

    def on_simulation_timeout(self):
        self._log_score()


class BeliefSetTabularVIAgent(BeliefSetHypothesis, rl.agent.Agent):
    def __init__(self, phi_mat, gamma):
        super().__init__()
        self._phi_mat = phi_mat
        self._phi_mat_pinv = np.linalg.pinv(self._phi_mat)
        self._gamma = gamma

    def to_ndarray(self) -> np.ndarray:
        return self._phi_mat

    @abstractmethod
    def get_score(self) -> float:  # pragma: no cover
        pass

    def update_transition(self, s, a, r, s_next, t, info):
        pass

    def on_simulation_timeout(self):
        pass

    def _construct_vi_agent(self, m_mat, w_vec):
        latent_agent = rl.agent.VIAgent(m_mat, w_vec, self._gamma)
        self._agent = rl.agent.StateRepresentationWrapperAgent(
            agent=latent_agent,
            phi=lambda s: np.matmul(s, self._phi_mat)
        )

    def reset(self, t_mat, r_vec):
        m_mat, w_vec = lam_from_mat(t_mat, r_vec, self._phi_mat)
        self._construct_vi_agent(m_mat, w_vec)

    def q_values(self, state):
        return self._agent.q_values(state)


class BeliefSetTabularScoring(BeliefSetTabularVIAgent):
    def __init__(self, phi_mat, gamma, score_fn):
        super().__init__(phi_mat, gamma)
        self._score = 0.
        self._score_fn = score_fn

    def reset(self, t_mat, r_vec):
        self._score = self._score_fn(self._phi_mat, t_mat, r_vec, self._gamma)
        super().reset(t_mat, r_vec)

    def get_score(self) -> float:
        return self._score


class BeliefSetTabularOracle(BeliefSetOracle):
    def __init__(self, num_states, num_latent_states, belief_set_constructor):
        partition_list = enumerate_n_partitions(num_states, num_latent_states)
        phi_mat_list = [cluster_idx_to_phi_mat(p) for p in partition_list]
        self._agent_list = [belief_set_constructor(p) for p in phi_mat_list]
        self._agent_best_idx = 0

    def _update_best_agent(self):
        scores = [a.get_score() for a in self._agent_list]
        self._agent_best_idx = np.argmax(scores)

    def generate_entry(self) -> BeliefSetEntry:
        agent = self._agent_list.pop(self._agent_best_idx)
        self._update_best_agent()
        return agent

    def reset_oracle(self):
        pass

    def get_score(self) -> float:
        return self._agent_list[self._agent_best_idx].get_score()

    def q_values(self, state):
        return self._agent_list[self._agent_best_idx].q_values(state)

    def reset(self, *params, **kwargs):
        for agent in self._agent_list:
            agent.reset(*params, **kwargs)
        self._update_best_agent()

    def update_transition(self, s, a, r, s_next, t, info):
        pass

    def on_simulation_timeout(self):
        pass


class MetaAgentTabular(MetaAgent):
    def __init__(self, oracle: BeliefSetTabularOracle, alpha: float,
                 beta: float):
        self._oracle = oracle
        self._alpha = alpha
        self._beta = beta
        self._belief_set = []
        self._iteration = 0

    def update_belief_set(self):
        new_entry = self._oracle.generate_entry()
        posterior = self.get_posterior_prob(new_entry.get_score())
        i = np.random.choice(np.arange(len(posterior)), p=posterior)
        if i == len(self._belief_set):
            self._belief_set.append(new_entry)
        self._belief_set[i].increase_count()
        self._iteration += 1

        return posterior

    def reset(self, *params, **kwargs):
        self._oracle.reset(*params, **kwargs)
        for agent in self._belief_set:
            agent.reset(*params, **kwargs)
