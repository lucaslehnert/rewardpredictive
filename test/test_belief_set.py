#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from unittest import TestCase


class TestBeliefAgentTabular(TestCase):
    def _get_data(self, num_mpds=5):
        import rewardpredictive as rp
        from rewardpredictive.utils import cluster_idx_to_phi_mat
        import numpy as np
        from functools import reduce

        mdp_list, partition_idx_seq_list, partition_list = rp.cycle_mdp_dataset.load_cycle_mdp_dataset()

        mdp_list = list(reduce(lambda a, b: a + b, mdp_list))[:num_mpds]
        part_idx_list = np.reshape(partition_idx_seq_list, -1)[:num_mpds]
        phi_mat_list = [cluster_idx_to_phi_mat(p) for p in partition_list]

        return mdp_list, part_idx_list, phi_mat_list

    def _test_agent(self, agent, mdp):
        import numpy as np
        import rlutils as rl

        s_list = np.eye(mdp.num_states(), dtype=np.float32)
        q_hat = np.stack([agent.q_values(s) for s in s_list]).transpose()
        q_star, _ = rl.algorithm.vi(*mdp.get_t_mat_r_vec(), gamma=0.9)
        self.assertLessEqual(np.max(np.abs(q_hat - q_star)), 1e-5)

    def _test_score_fn(self, score_fn):
        import rewardpredictive as rp
        import numpy as np

        mdp_list, part_idx_list, phi_mat_list = self._get_data()
        agent_list = []
        for phi_mat in phi_mat_list:
            agent_list.append(rp.belief_set.BeliefSetTabularScoring(
                phi_mat=phi_mat, gamma=0.9, score_fn=score_fn
            ))

        for mdp, part_idx in zip(mdp_list, part_idx_list):
            agent = agent_list[part_idx]
            phi_mat = phi_mat_list[part_idx]
            self.assertTrue(np.all(agent.to_ndarray() == phi_mat))

            agent.reset(*mdp.get_t_mat_r_vec())
            agent.update_transition(
                s=None, a=None, r=None, s_next=None, t=None, info=None
            )
            agent.on_simulation_timeout()

            self.assertLessEqual(-1e-5, agent.get_score())
            self._test_agent(agent, mdp)

    def test_reward_predictive(self):
        import rewardpredictive as rp

        self._test_score_fn(rp.reward_predictive.reward_predictive_score_tabular)

    def test_reward_maximizing(self):
        import rewardpredictive as rp

        self._test_score_fn(rp.reward_maximizing.reward_maximizing_score_tabular)

    def _test_oracle(self, constructor):
        import rewardpredictive as rp
        import numpy as np

        mdp_list, part_idx_list, phi_mat_list = self._get_data()

        num_s, num_latent_s = np.shape(phi_mat_list[0])
        oracle = rp.belief_set.BeliefSetTabularOracle(
            num_states=num_s,
            num_latent_states=num_latent_s,
            belief_set_constructor=constructor
        )
        oracle.reset_oracle()

        for i, (mdp, part_idx) in enumerate(zip(mdp_list, part_idx_list)):
            oracle.reset(*mdp.get_t_mat_r_vec())
            oracle.update_transition(
                s=None, a=None, r=None, s_next=None, t=None, info=None
            )
            oracle.on_simulation_timeout()

            self.assertLessEqual(-1e-5, oracle.get_score())
            self._test_agent(oracle, mdp)
            if i == len(mdp_list) - 1:
                agent = oracle.generate_entry()
                self.assertLessEqual(-1e-5, agent.get_score())
                self._test_agent(agent, mdp)

    def test_oracle_reward_predictive(self):
        import rewardpredictive as rp
        from rewardpredictive.belief_set import BeliefSetTabularScoring
        score = rp.reward_predictive.reward_predictive_score_tabular
        self._test_oracle(lambda p: BeliefSetTabularScoring(p, 0.9, score))

    def test_meta_agent_tabular(self):
        import rewardpredictive as rp
        import numpy as np
        from rewardpredictive.belief_set import BeliefSetTabularScoring

        score = rp.reward_predictive.reward_predictive_score_tabular
        constructor = lambda p: BeliefSetTabularScoring(p, 0.9, score)

        mdp_list, part_idx_list, phi_mat_list = self._get_data()
        oracle = rp.belief_set.BeliefSetTabularOracle(
            num_states=9,
            num_latent_states=3,
            belief_set_constructor=constructor
        )
        agent = rp.belief_set.MetaAgentTabular(oracle, alpha=1., beta=np.inf)

        for mdp in mdp_list:
            agent.reset(*mdp.get_t_mat_r_vec())
            agent.update_belief_set()
            self._test_agent(agent, mdp)
