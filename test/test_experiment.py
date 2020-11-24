#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from unittest import TestCase


class ExperimentCycleMDPDatasetPredictive(TestCase):
    def test(self):
        import rewardpredictive as rp
        import numpy as np
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentCycleMDPDatasetPredictive({
            rp.experiment.ExperimentCycleMDPDatasetPredictive.HP_REPEATS: 2
        })
        exp.run()
        self.assertLessEqual(3., np.mean(exp.results['total_reward']))
        count_list = [np.shape(c)[1] for c in exp.results['count']]
        self.assertEqual(2., np.mean(count_list))
        count_sum = [np.sum(c[-1]) for c in exp.results['count']]
        self.assertTrue(np.all(np.array(count_sum) == 20))

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentCycleMDPDatasetPredictive({
            rp.experiment.ExperimentCycleMDPDatasetPredictive.HP_REPEATS: 2
        })
        exp_reprod.run()

        total_rew_exp = np.array(exp.results['total_reward'])
        total_rew_exp_reprod = np.array(exp_reprod.results['total_reward'])
        self.assertTrue(np.all(total_rew_exp == total_rew_exp_reprod))
        count_exp = np.array(exp.results['count'])
        count_exp_reprod = np.array(exp.results['count'])
        self.assertTrue(np.all(count_exp == count_exp_reprod))
        score_exp = np.array(exp.results['score'])
        score_exp_reprod = np.array(exp.results['score'])
        self.assertTrue(np.all(score_exp == score_exp_reprod))


class ExperimentCycleMDPDatasetMaximizing(TestCase):
    def test(self):
        import rewardpredictive as rp
        import numpy as np
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentCycleMDPDatasetMaximizing({
            rp.experiment.ExperimentCycleMDPDatasetMaximizing.HP_REPEATS: 2
        })
        exp.run()
        self.assertLessEqual(3., np.mean(exp.results['total_reward']))
        count_list = [np.shape(c)[1] for c in exp.results['count']]
        self.assertLessEqual(2., np.mean(count_list))
        count_sum = [np.sum(c[-1]) for c in exp.results['count']]
        self.assertTrue(np.all(np.array(count_sum) == 20))

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentCycleMDPDatasetMaximizing({
            rp.experiment.ExperimentCycleMDPDatasetMaximizing.HP_REPEATS: 2
        })
        exp_reprod.run()

        for i in [0, 1]:
            total_rew_exp = np.array(exp.results['total_reward'][i])
            total_rew_exp_reprod = np.array(exp_reprod.results['total_reward'][i])
            self.assertTrue(np.all(total_rew_exp == total_rew_exp_reprod))
            count_exp = np.array(exp.results['count'][i])
            count_exp_reprod = np.array(exp.results['count'][i])
            self.assertTrue(np.all(count_exp == count_exp_reprod))
            score_exp = np.array(exp.results['score'][i])
            score_exp_reprod = np.array(exp.results['score'][i])
            self.assertTrue(np.all(score_exp == score_exp_reprod))


class TestExperimentRepresentationEvaluation(TestCase):
    def test(self):
        import rewardpredictive as rp
        import numpy as np
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentRepresentationEvaluation()
        exp.run()

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentRepresentationEvaluation()
        exp_reprod.run()

        self.assertTrue(np.all(exp.results['partition_list'][0] == exp_reprod.results['partition_list'][0]))
        self.assertTrue(np.all(exp.results['total_reward_list'][0] == exp_reprod.results['total_reward_list'][0]))
        exp_rew_err = exp.results['reward_prediction_error_list'][0]
        exp_reprod_rew_err = exp_reprod.results['reward_prediction_error_list'][0]
        self.assertTrue(np.all(exp_rew_err == exp_reprod_rew_err))


class TestExperimentMaze(TestCase):
    def _test_episode_length(self, exp, exp_reprod):
        import numpy as np
        for res, res_reprod in zip(exp.results['episode_length'], exp_reprod.results['episode_length']):
            self.assertTrue(np.all(res == res_reprod))

    def test_qlearning(self):
        import rewardpredictive as rp
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentMazeQLearning({
            rp.experiment.ExperimentMazeQLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeQLearning.HP_REPEATS: 2
        })
        exp.run()

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentMazeQLearning({
            rp.experiment.ExperimentMazeQLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeQLearning.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_episode_length(exp, exp_reprod)

    def test_qtransfer(self):
        import rewardpredictive as rp
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentMazeQTransfer({
            rp.experiment.ExperimentMazeQTransfer.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeQTransfer.HP_REPEATS: 2
        })
        exp.run()

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentMazeQTransfer({
            rp.experiment.ExperimentMazeQTransfer.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeQTransfer.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_episode_length(exp, exp_reprod)

    def test_sflearning(self):
        import rewardpredictive as rp
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentMazeSFLearning({
            rp.experiment.ExperimentMazeSFLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeSFLearning.HP_REPEATS: 2
        })
        exp.run()

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentMazeSFLearning({
            rp.experiment.ExperimentMazeSFLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeSFLearning.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_episode_length(exp, exp_reprod)

    def test_sftransfer(self):
        import rewardpredictive as rp
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentMazeSFTransfer({
            rp.experiment.ExperimentMazeSFTransfer.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeSFTransfer.HP_REPEATS: 2
        })
        exp.run()

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentMazeSFTransfer({
            rp.experiment.ExperimentMazeSFTransfer.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeSFTransfer.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_episode_length(exp, exp_reprod)

    def _test_mixture_agent_results(self, exp, exp_reprod):
        import numpy as np
        for res, res_reprod in zip(exp.results['prior'], exp_reprod.results['prior']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['posterior'], exp_reprod.results['posterior']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['count'], exp_reprod.results['count']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['score'], exp_reprod.results['score']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['phi_mat_list'], exp_reprod.results['phi_mat_list']):
            self.assertTrue(np.all(res == res_reprod))

    def test_maximizingq(self):
        import rewardpredictive as rp
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentMazeMaximizingQLearning({
            rp.experiment.ExperimentMazeMaximizingQLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeMaximizingQLearning.HP_REPEATS: 2
        })
        exp.run()

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentMazeMaximizingQLearning({
            rp.experiment.ExperimentMazeMaximizingQLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazeMaximizingQLearning.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_episode_length(exp, exp_reprod)
        self._test_mixture_agent_results(exp, exp_reprod)

    def test_predictiveq(self):
        import rewardpredictive as rp
        import tensorflow as tf
        tf.reset_default_graph()
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentMazePredictiveQLearning({
            rp.experiment.ExperimentMazePredictiveQLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazePredictiveQLearning.HP_REPEATS: 2
        })
        exp.run()

        tf.reset_default_graph()
        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentMazePredictiveQLearning({
            rp.experiment.ExperimentMazePredictiveQLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazePredictiveQLearning.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_episode_length(exp, exp_reprod)
        self._test_mixture_agent_results(exp, exp_reprod)

    def test_predictivesf(self):
        import rewardpredictive as rp
        import tensorflow as tf
        tf.reset_default_graph()
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentMazePredictiveSFLearning({
            rp.experiment.ExperimentMazePredictiveSFLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazePredictiveSFLearning.HP_REPEATS: 2
        })
        exp.run()

        tf.reset_default_graph()
        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentMazePredictiveSFLearning({
            rp.experiment.ExperimentMazePredictiveSFLearning.HP_NUM_EPISODES: 2,
            rp.experiment.ExperimentMazePredictiveSFLearning.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_episode_length(exp, exp_reprod)
        self._test_mixture_agent_results(exp, exp_reprod)


class TestExperimentGuitar(TestCase):
    def _test_equality(self, exp, exp_reprod):
        import numpy as np
        for res, res_reprod in zip(exp.results['episode_length'], exp_reprod.results['episode_length']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['total_reward'], exp_reprod.results['total_reward']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['s_buffer'], exp_reprod.results['s_buffer']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['a_buffer'], exp_reprod.results['a_buffer']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['r_buffer'], exp_reprod.results['r_buffer']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['sn_buffer'], exp_reprod.results['sn_buffer']):
            self.assertTrue(np.all(res == res_reprod))
        for res, res_reprod in zip(exp.results['t_buffer'], exp_reprod.results['t_buffer']):
            self.assertTrue(np.all(res == res_reprod))

    def test_sflearning(self):
        import rewardpredictive as rp
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentGuitarSFLearning({
            rp.experiment.ExperimentGuitarSFLearning.HP_REPEATS: 2
        })
        exp.run()

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentGuitarSFLearning({
            rp.experiment.ExperimentGuitarSFLearning.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_equality(exp, exp_reprod)

    def test_sftransfer(self):
        import rewardpredictive as rp
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentGuitarSFTransfer({
            rp.experiment.ExperimentGuitarSFTransfer.HP_REPEATS: 2
        })
        exp.run()

        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentGuitarSFTransfer({
            rp.experiment.ExperimentGuitarSFTransfer.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_equality(exp, exp_reprod)

    def test_predictive(self):
        import rewardpredictive as rp
        import tensorflow as tf
        tf.reset_default_graph()
        rp.set_seeds(12345)
        exp = rp.experiment.ExperimentGuitarRewardPredictive({
            rp.experiment.ExperimentGuitarRewardPredictive.HP_REPEATS: 2
        })
        exp.run()

        tf.reset_default_graph()
        rp.set_seeds(12345)
        exp_reprod = rp.experiment.ExperimentGuitarRewardPredictive({
            rp.experiment.ExperimentGuitarRewardPredictive.HP_REPEATS: 2
        })
        exp_reprod.run()

        self._test_equality(exp, exp_reprod)


class TestExperimentTaskSequenceRewardChange(TestCase):
    def _test_ep_len(self, exp, exp_reprod):
        import numpy as np
        self.assertTrue(np.shape(exp.results['episode_length']) == (2, 4, 3))
        eq_ep = [e1 == e2 for e1, e2 in zip(exp.results['episode_length'], exp_reprod.results['episode_length'])]
        self.assertTrue(np.all(eq_ep))

    def _test_exp_class(self, cls):
        from itertools import product
        import rewardpredictive as rp
        for task_seq, exp in product(['slight', 'significant'], ['egreedy']):
            hparam = {
                cls.HP_NUM_EPISODES: 3,
                cls.HP_REPEATS: 2,
                cls.HP_TASK_SEQUENCE: task_seq,
                cls.HP_EXPLORATION: exp
            }
            rp.set_seeds(12345)
            exp = cls(hparam)
            exp.run()

            rp.set_seeds(12345)
            exp_reprod = cls(hparam)
            exp_reprod.run()

            self._test_ep_len(exp, exp_reprod)

    def test_qlearning(self):
        import rewardpredictive as rp
        self._test_exp_class(rp.experiment.ExperimentTaskSequenceRewardChangeQLearning)

    def test_qtransfer(self):
        import rewardpredictive as rp
        self._test_exp_class(rp.experiment.ExperimentTaskSequenceRewardChangeQTransfer)

    def test_sflearning(self):
        import rewardpredictive as rp
        self._test_exp_class(rp.experiment.ExperimentTaskSequenceRewardChangeSFLearning)

    def test_sftransfer(self):
        import rewardpredictive as rp
        self._test_exp_class(rp.experiment.ExperimentTaskSequenceRewardChangeSFTransfer)

    def test_sftransfer_all(self):
        import rewardpredictive as rp
        self._test_exp_class(rp.experiment.ExperimentTaskSequenceRewardChangeSFTransferAll)
