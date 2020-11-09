#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
import multiprocessing as mp
import os
from abc import abstractmethod, ABC
from glob import glob
from itertools import product
from os import path as osp

import numpy as np
import rlutils as rl
import yaml

from . import guitar
from .belief_set import LoggerCountEpisodic
from .belief_set import LoggerPosteriorEpisodic
from .belief_set import LoggerPriorEpisodic
from .belief_set import LoggerScoreEpisodic
from .belief_set import MetaAgent, BeliefSetTabularOracle, MetaAgentTabular, \
    BeliefSetTabularScoring
from .cycle_mdp_dataset import load_cycle_mdp_dataset
from .enumerate_partitions import enumerate_all_partitions
from .evaluate import eval_reward_predictive
from .evaluate import eval_total_reward
from .mdp import MazeTaskSequence, TaskASlightRewardChange, TaskBSlightRewardChange, TaskASignificantRewardChange, \
    TaskBSignificantRewardChange
from .mdp import load_column_world_sequence
from .mdp import load_random_mdp_sequence
from .mdp import load_two_goal_with_wall_gridworld_sequence
from .reward_maximizing import RewardMaximizingQLearningOracle
from .reward_maximizing import reward_maximizing_score_tabular
from .reward_predictive import LSFMRepresentationLearner
from .reward_predictive import RewardPredictiveQLearningOracle
from .reward_predictive import RewardPredictiveSFLearningOracle
from .reward_predictive import reward_predictive_score_tabular
from .utils import SFLearning, EGreedyScheduleUpdate, TransitionListenerAgentDecorator
from .utils import TableModel
from .utils import pad_list_of_list_to_ndarray
from .utils import set_seeds
from .utils import simulate_episodes


def _load_experiment_from_save_dir(save_dir):
    with open(osp.join(save_dir, 'index.yaml'), 'r') as f:
        exp_dict = yaml.load(f, Loader=yaml.Loader)
    experiment_constructor = globals()[exp_dict['class_name']]
    return experiment_constructor.load(save_dir)


def _load_experiment_list(base_dir='./data'):
    index_list = glob(osp.join(base_dir, '**', 'index.yaml'), recursive=True)
    save_dir_list = [osp.split(p)[0] for p in index_list]
    # with mp.Pool() as p:
    #     experiment_list = p.map(_load_experiment_from_save_dir, save_dir_list)
    experiment_list = [_load_experiment_from_save_dir(d) for d in save_dir_list]
    return experiment_list


LEARNING_RATE_LIST = [0.1, 0.5, 0.9]


class ExperimentHParam(rl.Experiment):
    HP_REPEATS = 'repeats'

    def __init__(self, hparam=None, base_dir='./data'):
        if hparam is None:
            hparam = {}
        self.hparam = self._add_defaults_to_hparam(hparam)
        self.results = {}
        self.save_dir = self._get_save_dir(self.hparam, base_dir)

    def _add_defaults_to_hparam(self, hparam: dict) -> dict:
        hparam_complete = self.get_default_hparam()
        for k in hparam_complete.keys():
            if k in hparam.keys():
                hparam_complete[k] = hparam[k]
        return hparam_complete

    def get_default_hparam(self) -> dict:
        return {
            ExperimentHParam.HP_REPEATS: 10
        }

    def _get_save_dir(self, hparams, save_dir_base, hparam_keys=None) -> str:
        if hparam_keys is None:
            hparam_keys = list(hparams.keys())
        param_str_list = ['{}_{}'.format(k, hparams[k]) for k in hparam_keys]
        return osp.join(save_dir_base, self.get_class_name(), *param_str_list)

    def _run_experiment(self):
        res_list = [self.run_repeat(i) for i in range(self.hparam[ExperimentHParam.HP_REPEATS])]
        for k in res_list[0].keys():
            self.results[k] = [r[k] for r in res_list]

    @abstractmethod
    def run_repeat(self, rep_idx: int) -> dict:
        pass

    def _save_ndarray_rep(self, rep_list, name):
        fn_list = ['{:03d}_{}.npy'.format(i, name) for i in
                   range(len(rep_list))]
        for rep, fn in zip(rep_list, fn_list):
            np.save(osp.join(self.save_dir, fn), rep, allow_pickle=True)
        return fn_list

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        results_fn = {}
        for k in self.results.keys():
            results_fn[k] = self._save_ndarray_rep(self.results[k], k)

        exp_dict = {
            'class_name': self.get_class_name(),
            'hparam': self.hparam,
            'results': results_fn
        }
        with open(osp.join(self.save_dir, 'index.yaml'), 'w') as f:
            yaml.dump(exp_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, save_dir: str):
        with open(osp.join(save_dir, 'index.yaml'), 'r') as f:
            exp_dict = yaml.load(f, Loader=yaml.Loader)
        if not issubclass(globals()[exp_dict['class_name']], cls):
            exp_msg = 'Cannot load experiment because class {} is not a sub-class of {}.'
            exp_msg = exp_msg.format(exp_dict['class_name'],
                                     cls.get_class_name())
            raise rl.ExperimentException(exp_msg)
        exp = globals()[exp_dict['class_name']](exp_dict['hparam'])
        exp.results = exp_dict['results']
        for k in exp.results:
            exp.results[k] = [
                np.load(osp.join(exp.save_dir, p), allow_pickle=True) for p in
                exp.results[k]]
        return exp


def _run_repeat(exp, rep_idx):
    return exp.run_repeat(rep_idx)


class ExperimentHParamParallel(ExperimentHParam, ABC):
    def _run_experiment(self):
        num_repeats = self.hparam[ExperimentHParam.HP_REPEATS]
        param_list = [(self, i) for i in range(num_repeats)]
        with mp.Pool() as p:
            res_list = p.starmap(_run_repeat, param_list)
        for k in res_list[0].keys():
            self.results[k] = [r[k] for r in res_list]


class ExperimentSet(object):
    def __init__(self, experiment_list):
        self.experiment_list = experiment_list

    @abstractmethod
    def get_best_experiment(self):
        pass

    def get_experiment_list_by_hparam(self, hparam):
        # print('Retrieving experiment(s) with hyper-parameter(s):')
        # for k, v in hparam.items():
        #     print('\t{}: {}'.format(k, v))

        exp_list = []
        for exp in self.experiment_list:
            if all([exp.hparam[k] == hparam[k] for k in hparam.keys()]):
                exp_list.append(exp)
        return exp_list

    def get_hparam_values(self, hparam_key):
        return np.sort(
            np.unique([exp.hparam[hparam_key] for exp in self.experiment_list]))

    def run(self, seed=12345):
        for i, exp in enumerate(self.experiment_list):
            print('Running experiment {:2d}'.format(i))
            set_seeds(seed)
            exp.run()
            exp.save()

    def run_best(self, seed=12345):
        exp = self.get_best_experiment()
        set_seeds(seed)
        exp.run()
        exp.save()

    @classmethod
    def load(cls, base_dir='./data'):
        return ExperimentSet(_load_experiment_list(base_dir=base_dir))


class SmallTaskSequenceName:
    COLUMN_WORLD = 'ColumnWorld'
    # GOAL_GRID = 'GoalGrid'
    GOAL_WALL_GRID = 'GoalWallGrid'
    RAND_MDP = 'RandMDP'


class ExperimentRepresentationEvaluation(ExperimentHParam):
    HP_TASK_SEQ_NAME = 'task_seq_name'

    def __init__(self, *params, **kwargs):
        super().__init__(*params, **kwargs)

    def get_default_hparam(self) -> dict:
        return {
            ExperimentRepresentationEvaluation.HP_REPEATS: 1,
            ExperimentRepresentationEvaluation.HP_TASK_SEQ_NAME: SmallTaskSequenceName.COLUMN_WORLD
        }

    def run_repeat(self, rep_idx: int) -> dict:
        task_seq_name = self.hparam[ExperimentRepresentationEvaluation.HP_TASK_SEQ_NAME]
        if task_seq_name == SmallTaskSequenceName.COLUMN_WORLD:
            task_seq = load_column_world_sequence()
        elif task_seq_name == SmallTaskSequenceName.GOAL_WALL_GRID:
            task_seq = load_two_goal_with_wall_gridworld_sequence()
        elif task_seq_name == SmallTaskSequenceName.RAND_MDP:
            task_seq = load_random_mdp_sequence()

        num_states = task_seq[0].num_states()
        partition_list = enumerate_all_partitions(num_states)
        param_list = [(task_seq, p, 20, 10, 0.9) for p in partition_list]

        with mp.Pool() as p:
            res_list = p.starmap(eval_partition, param_list)

        total_reward_list = np.stack([r[0] for r in res_list]).astype(dtype=np.float32)
        reward_prediction_error_list = np.stack([r[1] for r in res_list]).astype(dtype=np.float32)
        result_dict = {
            'partition_list': partition_list,
            'total_reward_list': total_reward_list,
            'reward_prediction_error_list': reward_prediction_error_list
        }
        return result_dict


class ExperimentSetRepresentationEvaluation(ExperimentSet):

    def get_best_experiment(self):
        return None

    def get_experiment_by_task_sequence_name(self, task_seq_name):
        exp = list(filter(lambda e: e.hparam[ExperimentRepresentationEvaluation.HP_TASK_SEQ_NAME] == task_seq_name,
                          self.experiment_list))[0]
        return exp

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(osp.join(base_dir, ExperimentRepresentationEvaluation.get_class_name()))
        return ExperimentSetRepresentationEvaluation(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        task_seq_name_list = [SmallTaskSequenceName.COLUMN_WORLD,
                              SmallTaskSequenceName.GOAL_WALL_GRID,
                              SmallTaskSequenceName.RAND_MDP]
        exp_list = []
        for task_seq_name in task_seq_name_list:
            exp_list.append(ExperimentRepresentationEvaluation({
                ExperimentRepresentationEvaluation.HP_TASK_SEQ_NAME: task_seq_name
            }, base_dir=base_dir))
        return ExperimentSetRepresentationEvaluation(exp_list)


def eval_partition(task_seq, partition, repeats=5, rollout_depth=10, gamma=0.9, seed=12345):
    set_seeds(12345)
    total_rew = []
    rew_err = []

    for task in task_seq:
        total_rew.append(
            eval_total_reward(
                task=task,
                partition=partition,
                repeats=repeats,
                rollout_depth=rollout_depth,
                gamma=gamma
            )
        )
        rew_err.append(
            eval_reward_predictive(
                task=task,
                partition=partition,
                repeats=repeats,
                rollout_depth=rollout_depth
            )
        )

    total_rew = np.stack(total_rew).astype(dtype=np.float32)
    rew_err = np.stack(rew_err).astype(dtype=np.float32)

    return total_rew, rew_err


class ExperimentCycleMDPDataset(ExperimentHParamParallel):
    HP_ALPHA = 'alpha'
    HP_BETA = 'beta'
    HP_SCORE_FN = 'score_fn'

    def __init__(self, *params, **kwargs):
        super().__init__(*params, **kwargs)
        self.mdp_seq_list, _, _ = load_cycle_mdp_dataset()

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentCycleMDPDataset.HP_REPEATS] = 100
        defaults[ExperimentCycleMDPDataset.HP_BETA] = np.inf
        defaults[ExperimentCycleMDPDataset.HP_ALPHA] = 1.
        defaults[ExperimentCycleMDPDataset.HP_SCORE_FN] = 'reward_predictive'
        return defaults

    @abstractmethod
    def _get_belief_set_entry_constructor(self):
        pass

    def run_repeat(self, rep_idx: int) -> dict:
        print('Running repeat {}.'.format(rep_idx))
        set_seeds(12345)
        mdp_seq = self.mdp_seq_list[rep_idx]

        oracle = BeliefSetTabularOracle(
            num_states=9,
            num_latent_states=3,
            belief_set_constructor=self._get_belief_set_entry_constructor()
        )
        agent = MetaAgentTabular(
            oracle=oracle,
            alpha=self.hparam[ExperimentCycleMDPDataset.HP_ALPHA],
            beta=self.hparam[ExperimentCycleMDPDataset.HP_BETA]
        )
        reward_logger = rl.logging.LoggerTotalReward()
        score_logger = LoggerScoreEpisodic(agent)
        count_logger = LoggerCountEpisodic(agent)
        listener = rl.data.transition_listener(
            reward_logger, score_logger, count_logger
        )
        for i, mdp in enumerate(mdp_seq):
            agent.reset(*mdp.get_t_mat_r_vec())
            agent.update_belief_set()
            rl.data.simulate_gracefully(mdp, agent, listener, max_steps=10)

        count = count_logger.get_count_log()
        count = pad_list_of_list_to_ndarray(count, -1, dtype=np.int8)
        score = score_logger.get_score_log()
        score = pad_list_of_list_to_ndarray(score, -np.inf, dtype=np.float32)

        res_dict = {
            'total_reward': reward_logger.get_total_reward_episodic(),
            'count': count,
            'score': score,
        }
        return res_dict


class ExperimentCycleMDPDatasetPredictive(ExperimentCycleMDPDataset):
    def _get_belief_set_entry_constructor(self):
        score_fn = reward_predictive_score_tabular
        return lambda p: BeliefSetTabularScoring(p, 0.9, score_fn)


class ExperimentCycleMDPDatasetMaximizing(ExperimentCycleMDPDataset):
    def _get_belief_set_entry_constructor(self):
        score_fn = reward_maximizing_score_tabular
        return lambda p: BeliefSetTabularScoring(p, 0.9, score_fn)


class ExperimentSetCycleMDPDatasetPredictive(ExperimentSet):

    def get_best_experiment(self):
        return self.get_experiment_list_by_hparam({
            ExperimentCycleMDPDatasetPredictive.HP_ALPHA: 1.,
            ExperimentCycleMDPDatasetPredictive.HP_BETA: np.inf
        })[0]

    @classmethod
    def construct(cls, base_dir='./data', alpha=None, beta=None):
        if alpha is None:
            alpha_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        else:
            alpha_list = [alpha]
        if beta is None:
            beta_list = [0., 1., np.inf]
        else:
            beta_list = [beta]
        exp_list = []
        for alpha, beta in product(alpha_list, beta_list):
            exp_list.append(ExperimentCycleMDPDatasetPredictive({
                ExperimentCycleMDPDatasetPredictive.HP_ALPHA: alpha,
                ExperimentCycleMDPDatasetPredictive.HP_BETA: beta
            }, base_dir=base_dir))
        return ExperimentSetCycleMDPDatasetPredictive(exp_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentCycleMDPDatasetPredictive.get_class_name()))
        return ExperimentSetCycleMDPDatasetPredictive(exp_list)


class ExperimentSetCycleMDPDatasetMaximizing(ExperimentSet):

    def get_best_experiment(self):
        return self.get_experiment_list_by_hparam({
            ExperimentCycleMDPDatasetMaximizing.HP_ALPHA: 1.,
            ExperimentCycleMDPDatasetMaximizing.HP_BETA: np.inf
        })[0]

    @classmethod
    def construct(cls, base_dir='./data', alpha=None, beta=None):
        if alpha is None:
            alpha_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        else:
            alpha_list = [alpha]
        if beta is None:
            beta_list = [0., 1., np.inf]
        else:
            beta_list = [beta]
        exp_list = []
        for alpha, beta in product(alpha_list, beta_list):
            exp_list.append(ExperimentCycleMDPDatasetMaximizing({
                ExperimentCycleMDPDatasetMaximizing.HP_ALPHA: alpha,
                ExperimentCycleMDPDatasetMaximizing.HP_BETA: beta
            }, base_dir=base_dir))
        return ExperimentSetCycleMDPDatasetMaximizing(exp_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentCycleMDPDatasetMaximizing.get_class_name()))
        return ExperimentSetCycleMDPDatasetMaximizing(exp_list)


class ExperimentMaze(ExperimentHParam):
    HP_NUM_EPISODES = 'num_episodes'

    def __init__(self, *params, **kwargs):
        super().__init__(*params, **kwargs)
        self._task_seq = MazeTaskSequence()

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentMaze.HP_NUM_EPISODES] = 200
        return defaults

    def run_repeat(self, rep_idx: int) -> dict:
        print('Running repeat: {:2d}'.format(rep_idx))
        agent = self._construct_agent(self.hparam)
        ep_len = []
        for i, task in enumerate(self._task_seq.task_sequence):
            print('Running on task {}'.format(i))
            ep_len_logger = rl.logging.LoggerEpisodeLength()
            simulate_episodes(
                task=task,
                policy=rl.policy.GreedyPolicy(agent),
                transition_listener=rl.data.transition_listener(agent, ep_len_logger),
                num_episodes=self.hparam[ExperimentMazeSFLearning.HP_NUM_EPISODES],
                max_steps=5000
            )
            ep_len.append(ep_len_logger.get_episode_length())
            self._reset_agent(agent, self.hparam)
        return {'episode_length': np.array(ep_len, dtype=np.uint16)}

    @abstractmethod
    def _construct_agent(self, hparams):
        pass

    @abstractmethod
    def _reset_agent(self, agent, hparams):
        pass


class ExperimentMazeMixtureAgent(ExperimentMaze, ABC):
    def run_repeat(self, rep_idx: int) -> dict:
        print('Running repeat: {:2d}'.format(rep_idx))
        num_task = len(self._task_seq.task_sequence)
        num_episodes = self.hparam[ExperimentMazeSFLearning.HP_NUM_EPISODES]

        agent = self._construct_agent(self.hparam)
        ep_len_logger = rl.logging.LoggerEpisodeLength()
        prior_logger = LoggerPriorEpisodic(agent)
        posterior_logger = LoggerPosteriorEpisodic(agent)
        count_logger = LoggerCountEpisodic(agent)
        score_logger = LoggerScoreEpisodic(agent)
        # vc_logger = LoggerVisitationCountsEpisodic(agent)

        transition_listener = rl.data.transition_listener(
            agent,
            ep_len_logger,
            prior_logger,
            posterior_logger,
            count_logger,
            score_logger,
            # vc_logger
        )

        for i, task in enumerate(self._task_seq.task_sequence):
            print('Running on task {}'.format(i))
            simulate_episodes(task, agent, transition_listener, num_episodes)
            self._reset_agent(agent, self.hparam)

        ep_len = np.array(ep_len_logger.get_episode_length(), dtype=np.uint16)
        prior = pad_list_of_list_to_ndarray(prior_logger.get_prior_log(), -np.inf, dtype=np.float32)
        posterior = pad_list_of_list_to_ndarray(posterior_logger.get_posterior_log(), -np.inf, dtype=np.float32)
        count = pad_list_of_list_to_ndarray(count_logger.get_count_log(), -1, dtype=np.int8)
        score = pad_list_of_list_to_ndarray(score_logger.get_score_log(), -np.inf, dtype=np.float32)
        result_dict = {
            'episode_length': np.reshape(ep_len, [num_task, -1]),
            'prior': np.reshape(prior, [num_task, num_episodes, -1]),
            'posterior': np.reshape(posterior, [num_task, num_episodes, -1]),
            'count': np.reshape(count, [num_task, num_episodes, -1]),
            'score': np.reshape(score, [num_task, num_episodes, -1]),
            'phi_mat_list': np.stack(
                [b.to_ndarray() for b in agent.get_belief_set()]),
            # 'visitation_count': np.reshape(vc_logger.get_visitation_count_log(), [num_task, num_episodes, 4, -1])
        }
        return result_dict


class ExperimentMazeMaximizingQLearning(ExperimentMazeMixtureAgent):
    HP_LEARNING_RATE = 'learing_rate'
    HP_ALPHA = 'alpha'
    HP_BETA = 'beta'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentMazeMaximizingQLearning.HP_LEARNING_RATE] = 0.9
        defaults[ExperimentMazeMaximizingQLearning.HP_ALPHA] = 1e-3
        defaults[ExperimentMazeMaximizingQLearning.HP_BETA] = 100.0
        return defaults

    def _construct_agent(self, hparam):
        oracle = RewardMaximizingQLearningOracle(
            num_states=200,
            num_actions=4,
            num_latent_states=100,
            learning_rate=hparam[ExperimentMazeMaximizingQLearning.HP_LEARNING_RATE],
            init_v=1.0,
            gamma=0.9,
            reward_range=[0., 1.]
        )
        meta_agent = MetaAgent(
            oracle,
            alpha=hparam[ExperimentMazeMaximizingQLearning.HP_ALPHA],
            beta=hparam[ExperimentMazeMaximizingQLearning.HP_BETA]
        )
        return meta_agent

    def _reset_agent(self, agent, hparams):
        agent.update_belief_set()


class ExperimentMazePredictiveQLearning(ExperimentMazeMixtureAgent):
    HP_LEARNING_RATE = 'learing_rate'
    HP_ALPHA = 'alpha'
    HP_BETA = 'beta'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentMazePredictiveQLearning.HP_LEARNING_RATE] = 0.9
        defaults[ExperimentMazePredictiveQLearning.HP_ALPHA] = 1e-5
        defaults[ExperimentMazePredictiveQLearning.HP_BETA] = 100.0
        return defaults

    def _construct_agent(self, hparam):
        representation_learner = LSFMRepresentationLearner(
            num_states=200,
            num_actions=4,
            num_latent_states=100,
            gamma=0.9,
            num_training_iterations=1000,
            log_interval=1001
        )
        oracle = RewardPredictiveQLearningOracle(
            num_states=200,
            num_actions=4,
            lsfm_model=representation_learner,
            learning_rate=hparam[
                ExperimentMazePredictiveQLearning.HP_LEARNING_RATE],
            init_v=1.0,
            gamma=0.9,
            max_reward=1,
            score_alpha=1.0
        )
        meta_agent = MetaAgent(
            oracle,
            alpha=hparam[ExperimentMazePredictiveQLearning.HP_ALPHA],
            beta=hparam[ExperimentMazePredictiveQLearning.HP_BETA]
        )
        return meta_agent

    def _reset_agent(self, agent, hparams):
        agent.update_belief_set()


class ExperimentMazePredictiveSFLearning(ExperimentMazeMixtureAgent):
    HP_LEARNING_RATE_SF = 'learning_rate_sf'
    HP_LEARNING_RATE_REWARD = 'learing_rate_reward'
    HP_ALPHA = 'alpha'
    HP_BETA = 'beta'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_SF] = 0.5
        defaults[
            ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_REWARD] = 0.9
        defaults[ExperimentMazePredictiveSFLearning.HP_ALPHA] = 1e-9
        defaults[ExperimentMazePredictiveSFLearning.HP_BETA] = 100.0
        return defaults

    def _construct_agent(self, hparam):
        representation_learner = LSFMRepresentationLearner(
            num_states=200,
            num_actions=4,
            num_latent_states=100,
            gamma=0.9,
            num_training_iterations=1000,
            log_interval=1001
        )
        oracle = RewardPredictiveSFLearningOracle(
            num_states=200,
            num_actions=4,
            lsfm_model=representation_learner,
            learning_rate_sf=hparam[
                ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_SF],
            learning_rate_reward=hparam[
                ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_REWARD],
            init_v=1.0,
            gamma=0.9,
            max_reward=1,
            score_alpha=1.0
        )
        meta_agent = MetaAgent(
            oracle,
            alpha=hparam[ExperimentMazePredictiveSFLearning.HP_ALPHA],
            beta=hparam[ExperimentMazePredictiveSFLearning.HP_BETA]
        )
        return meta_agent

    def _reset_agent(self, agent, hparams):
        agent.update_belief_set()


class ExperimentMazeSFLearning(ExperimentMaze):
    HP_LEARNING_RATE_SF = 'learning_rate_sf'
    HP_LEARNING_RATE_REWARD = 'learing_rate_reward'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentMazeSFLearning.HP_LEARNING_RATE_SF] = 0.5
        defaults[ExperimentMazeSFLearning.HP_LEARNING_RATE_REWARD] = 0.9
        return defaults

    def _construct_agent(self, hparams):
        return SFLearning(
            num_states=200,
            num_actions=4,
            learning_rate_sf=self.hparam[
                ExperimentMazeSFLearning.HP_LEARNING_RATE_SF],
            learning_rate_reward=self.hparam[
                ExperimentMazeSFLearning.HP_LEARNING_RATE_REWARD],
            gamma=0.9,
            init_sf_mat=np.eye(4 * 200, dtype=np.float32),
            init_w_vec=np.ones(4 * 200, dtype=np.float32)
        )

    def _reset_agent(self, agent, hparams):
        agent.reset(reset_sf=True, reset_w=True)


class ExperimentMazeSFTransfer(ExperimentMazeSFLearning):
    def _reset_agent(self, agent, hparams):
        agent.reset(reset_sf=False, reset_w=True)


class ExperimentMazeQLearning(ExperimentMaze):
    HP_LEARNING_RATE = 'learning_rate'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentMazeQLearning.HP_LEARNING_RATE] = 0.9
        return defaults

    def _construct_agent(self, hparams):
        return rl.agent.QLearning(
            num_states=200,
            num_actions=4,
            learning_rate=self.hparam[ExperimentMazeQLearning.HP_LEARNING_RATE],
            gamma=0.9,
            init_Q=1.
        )

    def _reset_agent(self, agent, hparams):
        agent.reset()


class ExperimentMazeQTransfer(ExperimentMazeQLearning):
    def _reset_agent(self, agent, hparams):
        pass


# class ExperimentMazeWithGroundTruthAbstraction(ExperimentMaze, ABC):
#     def __init__(self, *params, **kwargs):
#         super().__init__(*params, **kwargs)
#         self._phi_mat_seq = load_maze_sequence()[1]
#
#     def run_repeat(self, rep_idx: int) -> dict:
#         print('Running repeat: {:2d}'.format(rep_idx))
#         latent_agent = self._construct_agent(self.hparam)
#         ep_len = []
#         for i, (task, phi_mat) in enumerate(
#                 zip(self._task_seq.task_sequence, self._phi_mat_seq)):
#             print('Running on task {}'.format(i))
#             agent = rl.agent.StateRepresentationWrapperAgent(
#                 agent=latent_agent,
#                 phi=lambda s: np.matmul(s, phi_mat)
#             )
#             ep_len_logger = rl.logging.LoggerEpisodeLength()
#             simulate_episodes(
#                 task=task,
#                 policy=rl.policy.GreedyPolicy(agent),
#                 transition_listener=rl.data.transition_listener(agent,
#                                                                 ep_len_logger),
#                 num_episodes=self.hparam[
#                     ExperimentMazeSFLearning.HP_NUM_EPISODES],
#                 max_steps=5000
#             )
#             ep_len.append(ep_len_logger.get_episode_length())
#             self._reset_agent(agent, self.hparam)
#         return {'episode_length': np.array(ep_len, dtype=np.uint16)}


# class ExperimentMazeWithGroundTruthAbstractionSFLearning(
#     ExperimentMazeWithGroundTruthAbstraction):
#     HP_LEARNING_RATE_SF = 'learning_rate_sf'
#     HP_LEARNING_RATE_REWARD = 'learing_rate_reward'
#
#     def get_default_hparam(self) -> dict:
#         defaults = super().get_default_hparam()
#         defaults[ExperimentMazeSFLearning.HP_LEARNING_RATE_SF] = 0.5
#         defaults[ExperimentMazeSFLearning.HP_LEARNING_RATE_REWARD] = 0.9
#         return defaults
#
#     def _construct_agent(self, hparams):
#         return SFLearning(
#             num_states=100,
#             num_actions=4,
#             learning_rate_sf=self.hparam[
#                 ExperimentMazeSFLearning.HP_LEARNING_RATE_SF],
#             learning_rate_reward=self.hparam[
#                 ExperimentMazeSFLearning.HP_LEARNING_RATE_REWARD],
#             gamma=0.9,
#             init_sf_mat=np.eye(4 * 100, dtype=np.float32),
#             init_w_vec=np.ones(4 * 100, dtype=np.float32)
#         )
#
#     def _reset_agent(self, agent, hparams):
#         agent.reset(reset_sf=True, reset_w=True)


class ExperimentGuitar(ExperimentHParam):
    HP_LEARNING_RATE_SF = 'learning_rate_sf'
    HP_LEARNING_RATE_REWARD = 'learning_rate_reward'
    HP_EPISODES = 'episodes'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentGuitarSFLearning.HP_LEARNING_RATE_SF] = 0.1
        defaults[ExperimentGuitarSFLearning.HP_LEARNING_RATE_REWARD] = 0.1
        defaults[ExperimentGuitarSFLearning.HP_EPISODES] = 100
        return defaults

    def _get_scale_tasks(self):
        mdp_0 = guitar.guitar_melody_mdp(('C', 'D', 'E', 'F', 'G', 'A', 'B'), reward_max=1., reward_min=-1.)
        mdp_1 = guitar.guitar_melody_mdp(('A', 'B', 'C', 'D', 'E', 'F', 'G'), reward_max=1., reward_min=-1.)
        return mdp_0, mdp_1

    def run_repeat(self, rep_idx: int) -> dict:
        mdp_0, mdp_1 = self._get_scale_tasks()
        max_reward = np.max(mdp_0.get_t_mat_r_mat()[1])

        agent = self._get_agent_task_1(mdp_0.num_states(), mdp_0.num_actions(), max_reward=max_reward)
        log_traj = rl.logging.LoggerTrajectory()
        log_ep_len_0 = rl.logging.LoggerEpisodeLength()
        log_tot_rew_0 = rl.logging.LoggerTotalReward()
        simulate_episodes(
            mdp_0,
            rl.policy.GreedyPolicy(agent),
            rl.data.transition_listener(agent, log_ep_len_0, log_tot_rew_0, log_traj),
            num_episodes=self.hparam[ExperimentGuitarSFLearning.HP_EPISODES],
            max_steps=2000
        )
        agent = self._get_agent_task_2(agent)
        log_ep_len_1 = rl.logging.LoggerEpisodeLength()
        log_tot_rew_1 = rl.logging.LoggerTotalReward()
        simulate_episodes(
            mdp_1,
            rl.policy.GreedyPolicy(agent),
            rl.data.transition_listener(agent, log_ep_len_1, log_tot_rew_1, log_traj),
            num_episodes=self.hparam[ExperimentGuitarSFLearning.HP_EPISODES],
            max_steps=2000
        )
        ep_len = np.array(
            [log_ep_len_0.get_episode_length(), log_ep_len_1.get_episode_length()],
            dtype=np.uint16
        )
        tot_rew = np.array(
            [log_tot_rew_0.get_total_reward_episodic(), log_tot_rew_1.get_total_reward_episodic()],
            dtype=np.float32
        )
        res_dict = {
            'episode_length': ep_len,
            'total_reward': tot_rew,
            's_buffer': np.concatenate([traj.all()[0] for traj in log_traj.get_trajectory_list()], axis=0),
            'a_buffer': np.concatenate([traj.all()[1] for traj in log_traj.get_trajectory_list()], axis=0),
            'r_buffer': np.concatenate([traj.all()[2] for traj in log_traj.get_trajectory_list()], axis=0),
            'sn_buffer': np.concatenate([traj.all()[3] for traj in log_traj.get_trajectory_list()], axis=0),
            't_buffer': np.concatenate([traj.all()[4] for traj in log_traj.get_trajectory_list()], axis=0)
        }
        return res_dict

    @abstractmethod
    def _get_agent_task_1(self, num_states, num_actions, max_reward):
        pass

    @abstractmethod
    def _get_agent_task_2(self, task_1_agent):
        pass


class ExperimentGuitarSFLearning(ExperimentGuitar):

    def _get_agent_task_1(self, num_states, num_actions, max_reward):
        gamma = 0.9
        if max_reward == 1.:
            v_max = (gamma ** 7 - 1) / (gamma - 1)
        else:
            v_max = 0.
        agent = SFLearning(
            num_states=num_states,
            num_actions=num_actions,
            learning_rate_sf=self.hparam[ExperimentGuitarSFLearning.HP_LEARNING_RATE_SF],
            learning_rate_reward=self.hparam[ExperimentGuitarSFLearning.HP_LEARNING_RATE_REWARD],
            gamma=gamma,
            init_sf_mat=np.eye(num_states * num_actions, dtype=np.float32),
            init_w_vec=np.ones(num_states * num_actions, dtype=np.float32) * v_max
        )
        return agent

    def _get_agent_task_2(self, task_1_agent):
        task_1_agent.reset(reset_sf=True, reset_w=True)
        return task_1_agent


class ExperimentGuitarSFTransfer(ExperimentGuitarSFLearning):

    def _get_agent_task_2(self, task_1_agent):
        task_1_agent.reset(reset_sf=False, reset_w=True)
        return task_1_agent


class ExperimentGuitarRewardPredictive(ExperimentGuitar):

    def _get_agent_task_1(self, num_states, num_actions, max_reward):
        gamma = 0.9
        if max_reward == 1.:
            v_max = (gamma ** 7 - 1) / (gamma - 1)
        else:
            v_max = 0.
        sf_learning = SFLearning(
            num_states=num_states,
            num_actions=num_actions,
            learning_rate_sf=self.hparam[ExperimentGuitarRewardPredictive.HP_LEARNING_RATE_SF],
            learning_rate_reward=self.hparam[ExperimentGuitarRewardPredictive.HP_LEARNING_RATE_REWARD],
            gamma=gamma,
            init_sf_mat=np.eye(num_states * num_actions, dtype=np.float32),
            init_w_vec=np.ones(num_states * num_actions, dtype=np.float32) * v_max
        )
        table_model = TableModel(
            num_states,
            num_actions,
            reward_sampler=lambda s: -1. * np.ones(s, dtype=np.float32),
            max_reward=max_reward
        )
        agent = TransitionListenerAgentDecorator(sf_learning, [table_model])
        return agent

    def _get_agent_task_2(self, task_1_agent):
        gamma = 0.9
        table_model = task_1_agent.transition_listener_list[0]
        representation_learner = LSFMRepresentationLearner(
            num_states=table_model.num_states(),
            num_actions=table_model.num_actions(),
            num_latent_states=14,
            gamma=gamma,
            num_training_iterations=5000,
            log_interval=1000,
            learning_rate=1e-2,
            alpha_r=1.,
            alpha_sf=.001,
            alpha_reg=0.
        )
        phi_mat_learned = representation_learner.learn_representation(*table_model.get_t_mat_r_vec())

        task_1_agent.reset(reset_sf=True, reset_w=True)
        w_val = np.max(task_1_agent.agent.get_w_vector())
        agent_latent = SFLearning(
            num_states=np.shape(phi_mat_learned)[1],
            num_actions=table_model.num_actions(),
            learning_rate_sf=self.hparam[ExperimentGuitarRewardPredictive.HP_LEARNING_RATE_SF],
            learning_rate_reward=self.hparam[ExperimentGuitarRewardPredictive.HP_LEARNING_RATE_REWARD],
            gamma=gamma,
            init_sf_mat=np.eye(np.shape(phi_mat_learned)[1] * table_model.num_actions(), dtype=np.float32),
            init_w_vec=np.ones(np.shape(phi_mat_learned)[1] * table_model.num_actions(), dtype=np.float32) * w_val
        )
        agent = rl.agent.StateRepresentationWrapperAgent(agent_latent, phi=lambda s: np.matmul(s, phi_mat_learned))
        return agent


class ExperimentSetGuitar(ExperimentSet):
    def get_best_experiment(self):
        exp_idx = np.argmax([np.sum(exp.results['total_reward']) for exp in self.experiment_list])
        exp = self.experiment_list[exp_idx]
        print('Retrieving experiment(s) with hyper-parameter(s):')
        for k, v in exp.hparam.items():
            print('\t{}: {}'.format(k, v))
        return exp


class ExperimentSetGuitarSFLearning(ExperimentSetGuitar):
    def __init__(self, experiment_list):
        lr_sf = ExperimentGuitarSFLearning.HP_LEARNING_RATE_SF
        lr_r = ExperimentGuitarSFLearning.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentGuitarSFLearning.get_class_name()))
        return ExperimentSetGuitarSFLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        for lr_sf, lr_r in product(LEARNING_RATE_LIST, LEARNING_RATE_LIST):
            exp_list.append(ExperimentGuitarSFLearning({
                ExperimentGuitarSFLearning.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentGuitarSFLearning.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentGuitarSFLearning.HP_REPEATS: 10
            }, base_dir=base_dir))
        return ExperimentSetGuitarSFLearning(exp_list)


class ExperimentSetGuitarSFTransfer(ExperimentSetGuitar):
    def __init__(self, experiment_list):
        lr_sf = ExperimentGuitarSFTransfer.HP_LEARNING_RATE_SF
        lr_r = ExperimentGuitarSFTransfer.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentGuitarSFTransfer.get_class_name()))
        return ExperimentSetGuitarSFTransfer(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        for lr_sf, lr_r in product(LEARNING_RATE_LIST, LEARNING_RATE_LIST):
            exp_list.append(ExperimentGuitarSFTransfer({
                ExperimentGuitarSFTransfer.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentGuitarSFTransfer.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentGuitarSFTransfer.HP_REPEATS: 10
            }, base_dir=base_dir))
        return ExperimentSetGuitarSFTransfer(exp_list)


class ExperimentSetGuitarRewardPredictive(ExperimentSetGuitar):
    def __init__(self, experiment_list):
        lr_sf = ExperimentGuitarRewardPredictive.HP_LEARNING_RATE_SF
        lr_r = ExperimentGuitarRewardPredictive.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(osp.join(base_dir,
                                                  ExperimentGuitarRewardPredictive.get_class_name()))
        return ExperimentSetGuitarRewardPredictive(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        for lr_sf, lr_r in product(LEARNING_RATE_LIST, LEARNING_RATE_LIST):
            exp_list.append(ExperimentGuitarRewardPredictive({
                ExperimentGuitarRewardPredictive.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentGuitarRewardPredictive.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentGuitarRewardPredictive.HP_REPEATS: 10
            }, base_dir=base_dir))
        return ExperimentSetGuitarRewardPredictive(exp_list)


class ExperimentSetMaze(ExperimentSet, ABC):
    def get_lowest_avg_episode_length(self):
        ep_len = [np.mean(np.mean(e.results['episode_length'], axis=-1), axis=-1) for e in self.experiment_list]
        exp_idx = np.argmin(ep_len)
        return self.experiment_list[exp_idx]


class ExperimentSetMazeQLearning(ExperimentSetMaze):

    def __init__(self, experiment_list):
        lr = ExperimentMazeQLearning.HP_LEARNING_RATE
        experiment_list.sort(key=lambda e: (e.hparam[lr]))
        super().__init__(experiment_list)

    def get_best_experiment(self):
        return self.get_experiment_list_by_hparam({
            ExperimentMazeQLearning.HP_LEARNING_RATE: 0.9,
        })[0]

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentMazeQLearning.get_class_name()))
        return ExperimentSetMazeQLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        for lr in LEARNING_RATE_LIST:
            exp_list.append(ExperimentMazeQLearning({
                ExperimentMazeQLearning.HP_LEARNING_RATE: lr
            }, base_dir=base_dir))
        return ExperimentSetMazeQLearning(exp_list)


class ExperimentSetMazeQTransfer(ExperimentSetMazeQLearning):

    def get_best_experiment(self):
        return self.get_experiment_list_by_hparam({
            ExperimentMazeQTransfer.HP_LEARNING_RATE: 0.9,
        })[0]

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentMazeQTransfer.get_class_name()))
        return ExperimentSetMazeQTransfer(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        for lr in LEARNING_RATE_LIST:
            exp_list.append(ExperimentMazeQTransfer({
                ExperimentMazeQTransfer.HP_LEARNING_RATE: lr
            }, base_dir=base_dir))
        return ExperimentSetMazeQTransfer(exp_list)


class ExperimentSetMazeSFLearning(ExperimentSetMaze):

    def __init__(self, experiment_list):
        lr_sf = ExperimentMazeSFLearning.HP_LEARNING_RATE_SF
        lr_r = ExperimentMazeSFLearning.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    def get_best_experiment(self):
        return self.get_experiment_list_by_hparam({
            ExperimentMazeSFLearning.HP_LEARNING_RATE_SF: 0.5,
            ExperimentMazeSFLearning.HP_LEARNING_RATE_REWARD: 0.9,
        })[0]

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentMazeSFLearning.get_class_name()))
        return ExperimentSetMazeSFLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        for lr_sf, lr_r in product(LEARNING_RATE_LIST, LEARNING_RATE_LIST):
            exp_list.append(ExperimentMazeSFLearning({
                ExperimentMazeSFLearning.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentMazeSFLearning.HP_LEARNING_RATE_REWARD: lr_r
            }, base_dir=base_dir))
        return ExperimentSetMazeSFLearning(exp_list)


class ExperimentSetMazeSFTransfer(ExperimentSetMazeSFLearning):
    def get_best_experiment(self):
        return self.get_experiment_list_by_hparam({
            ExperimentMazeSFTransfer.HP_LEARNING_RATE_SF: 0.5,
            ExperimentMazeSFTransfer.HP_LEARNING_RATE_REWARD: 0.9,
        })[0]

    @classmethod
    def load(self, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentMazeSFTransfer.get_class_name()))
        return ExperimentSetMazeSFTransfer(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        for lr_sf, lr_r in product(LEARNING_RATE_LIST, LEARNING_RATE_LIST):
            exp_list.append(ExperimentMazeSFTransfer({
                ExperimentMazeSFTransfer.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentMazeSFTransfer.HP_LEARNING_RATE_REWARD: lr_r
            }, base_dir=base_dir))
        return ExperimentSetMazeSFTransfer(exp_list)


# class ExperimentSetMazeWithGroundTruthAbstractionSFLearning(
#     ExperimentSetMazeSFLearning):
#     def get_best_experiment(self):
#         return self.get_experiment_list_by_hparam({
#             ExperimentMazeWithGroundTruthAbstractionSFLearning.HP_LEARNING_RATE_SF: 0.5,
#             ExperimentMazeWithGroundTruthAbstractionSFLearning.HP_LEARNING_RATE_REWARD: 0.9,
#         })[0]
#
#     @classmethod
#     def load(cls, base_dir='./data'):
#         exp_list = _load_experiment_list(
#             osp.join(base_dir,
#                      ExperimentMazeWithGroundTruthAbstractionSFLearning.get_class_name())
#         )
#         return ExperimentSetMazeWithGroundTruthAbstractionSFLearning(exp_list)
#
#     @classmethod
#     def construct(cls, base_dir='./data'):
#         exp_list = []
#         for lr_sf, lr_r in product(LEARNING_RATE_LIST, LEARNING_RATE_LIST):
#             exp_list.append(ExperimentMazeWithGroundTruthAbstractionSFLearning({
#                 ExperimentMazeWithGroundTruthAbstractionSFLearning.HP_LEARNING_RATE_SF: lr_sf,
#                 ExperimentMazeWithGroundTruthAbstractionSFLearning.HP_LEARNING_RATE_REWARD: lr_r
#             }, base_dir=base_dir))
#         return ExperimentSetMazeWithGroundTruthAbstractionSFLearning(exp_list)


class ExperimentSetMazeMaximizingQLearning(ExperimentSetMaze):
    def __init__(self, experiment_list):
        alpha = ExperimentMazeMaximizingQLearning.HP_ALPHA
        beta = ExperimentMazeMaximizingQLearning.HP_BETA
        lr = ExperimentMazeMaximizingQLearning.HP_LEARNING_RATE
        experiment_list.sort(
            key=lambda e: (e.hparam[alpha], e.hparam[beta], e.hparam[lr]))
        super().__init__(experiment_list)

    def get_best_experiment(self):
        return self.get_experiment_list_by_hparam({
            ExperimentMazeMaximizingQLearning.HP_LEARNING_RATE: 0.9,
            ExperimentMazeMaximizingQLearning.HP_ALPHA: 1e-3,
            ExperimentMazeMaximizingQLearning.HP_BETA: 100.
        })[0]

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(osp.join(base_dir,
                                                  ExperimentMazeMaximizingQLearning.get_class_name()))
        return ExperimentSetMazeMaximizingQLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data', alpha=None, beta=None):
        if alpha is None:
            alpha_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        else:
            alpha_list = [alpha]
        if beta is None:
            beta_list = [1., 100., 10000.]
        else:
            beta_list = [beta]
        exp_list = []
        for lr, alpha, beta in product(LEARNING_RATE_LIST, alpha_list,
                                       beta_list):
            exp_list.append(ExperimentMazeMaximizingQLearning({
                ExperimentMazeMaximizingQLearning.HP_LEARNING_RATE: lr,
                ExperimentMazeMaximizingQLearning.HP_ALPHA: alpha,
                ExperimentMazeMaximizingQLearning.HP_BETA: beta
            }, base_dir=base_dir))
        return ExperimentSetMazeMaximizingQLearning(exp_list)


class ExperimentSetMazePredictiveQLearning(ExperimentSetMaze):
    def __init__(self, experiment_list):
        alpha = ExperimentMazePredictiveQLearning.HP_ALPHA
        beta = ExperimentMazePredictiveQLearning.HP_BETA
        lr = ExperimentMazePredictiveQLearning.HP_LEARNING_RATE
        experiment_list.sort(
            key=lambda e: (e.hparam[alpha], e.hparam[beta], e.hparam[lr]))
        super().__init__(experiment_list)

    def get_best_experiment(self):
        return self.get_experiment_list_by_hparam({
            ExperimentMazePredictiveQLearning.HP_LEARNING_RATE: 0.9,
            ExperimentMazePredictiveQLearning.HP_ALPHA: 1e-5,
            ExperimentMazePredictiveQLearning.HP_BETA: 100.
        })[0]

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(osp.join(base_dir,
                                                  ExperimentMazePredictiveQLearning.get_class_name()))
        return ExperimentSetMazePredictiveQLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data', alpha=None, beta=None):
        if alpha is None:
            alpha_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        else:
            alpha_list = [alpha]
        if beta is None:
            beta_list = [1., 100., 10000.]
        else:
            beta_list = [beta]
        exp_list = []
        for lr, alpha, beta in product(LEARNING_RATE_LIST, alpha_list,
                                       beta_list):
            exp_list.append(ExperimentMazePredictiveQLearning({
                ExperimentMazePredictiveQLearning.HP_LEARNING_RATE: lr,
                ExperimentMazePredictiveQLearning.HP_ALPHA: alpha,
                ExperimentMazePredictiveQLearning.HP_BETA: beta
            }, base_dir=base_dir))
        return ExperimentSetMazePredictiveQLearning(exp_list)


class ExperimentSetMazePredictiveSFLearning(ExperimentSetMaze):
    def __init__(self, experiment_list):
        alpha = ExperimentMazePredictiveSFLearning.HP_ALPHA
        beta = ExperimentMazePredictiveSFLearning.HP_BETA
        lr_sf = ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_SF
        lr_r = ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (
            e.hparam[alpha], e.hparam[beta], e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    def get_best_experiment(self):
        return self.get_experiment_list_by_hparam({
            ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_SF: 0.5,
            ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_REWARD: 0.9,
            ExperimentMazePredictiveSFLearning.HP_ALPHA: 1e-9,
            ExperimentMazePredictiveSFLearning.HP_BETA: 100.
        })[0]

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(osp.join(base_dir,
                                                  ExperimentMazePredictiveSFLearning.get_class_name()))
        return ExperimentSetMazePredictiveSFLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data', alpha=None, beta=None):
        if alpha is None:
            alpha_list = [1e-9, 1e-5, 1e-1]
        else:
            alpha_list = [alpha]
        if beta is None:
            beta_list = [1., 100., 10000.]
        else:
            beta_list = [beta]
        exp_list = []
        for lr_sf, lr_r, alpha, beta in product(LEARNING_RATE_LIST,
                                                LEARNING_RATE_LIST, alpha_list,
                                                beta_list):
            exp_list.append(ExperimentMazePredictiveSFLearning({
                ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentMazePredictiveSFLearning.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentMazePredictiveSFLearning.HP_ALPHA: alpha,
                ExperimentMazePredictiveSFLearning.HP_BETA: beta
            }, base_dir=base_dir))
        return ExperimentSetMazePredictiveSFLearning(exp_list)


class ExperimentTaskSequenceRewardChange(ExperimentHParamParallel):
    HP_EXPLORATION = 'exploration'
    HP_TASK_SEQUENCE = 'task_sequence'
    HP_NUM_EPISODES = 'episodes'

    def __init__(self, *params, **kwargs):
        super().__init__(*params, **kwargs)
        self.task_sequence = self._get_task_sequence()

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRewardChange.HP_REPEATS] = 20
        defaults[ExperimentTaskSequenceRewardChange.HP_TASK_SEQUENCE] = 'slight'
        defaults[ExperimentTaskSequenceRewardChange.HP_EXPLORATION] = 'optimistic'
        defaults[ExperimentTaskSequenceRewardChange.HP_NUM_EPISODES] = 200
        return defaults

    def _get_task_sequence(self):
        if self.hparam[ExperimentTaskSequenceRewardChange.HP_TASK_SEQUENCE] == 'slight':
            mdp_seq = [
                TaskASlightRewardChange(),
                TaskBSlightRewardChange(),
                TaskASlightRewardChange(),
                TaskBSlightRewardChange()
            ]
        elif self.hparam[ExperimentTaskSequenceRewardChange.HP_TASK_SEQUENCE] == 'significant':
            mdp_seq = [
                TaskASignificantRewardChange(),
                TaskBSignificantRewardChange(),
                TaskASignificantRewardChange(),
                TaskBSignificantRewardChange()
            ]
        return mdp_seq

    def run_repeat(self, rep_idx: int) -> dict:
        set_seeds(12345 + rep_idx)
        episodes = self.hparam[ExperimentTaskSequenceRewardChange.HP_NUM_EPISODES]
        agent = self._construct_agent()
        ep_len_logger = rl.logging.LoggerEpisodeLength()
        if self.hparam[ExperimentTaskSequenceRewardChange.HP_EXPLORATION] == 'optimistic':
            policy = rl.policy.GreedyPolicy(agent)
            transition_listener = rl.data.transition_listener(agent, ep_len_logger)
        elif self.hparam[ExperimentTaskSequenceRewardChange.HP_EXPLORATION] == 'egreedy':
            policy = rl.policy.EGreedyPolicy(agent, 1.0)
            exp_schedule = rl.schedule.LinearInterpolatedVariableSchedule([0, 180], [1., 0.])
            exp_schedule_listener = EGreedyScheduleUpdate(policy, exp_schedule)
            transition_listener = rl.data.transition_listener(agent, exp_schedule_listener, ep_len_logger)
        for task in self.task_sequence:
            simulate_episodes(task, policy, transition_listener, episodes, max_steps=2000)
            self._reset_agent(agent)

        res_dict = {
            'episode_length': np.reshape(ep_len_logger.get_episode_length(), [len(self.task_sequence), -1])
        }
        return res_dict

    @abstractmethod
    def _construct_agent(self):
        pass

    @abstractmethod
    def _reset_agent(self, agent):
        pass


class ExperimentTaskSequenceRewardChangeQLearning(ExperimentTaskSequenceRewardChange):
    HP_LEARNING_RATE = 'lr'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRewardChangeQLearning.HP_LEARNING_RATE] = 0.9
        return defaults

    def _construct_agent(self):
        if self.hparam[ExperimentTaskSequenceRewardChange.HP_EXPLORATION] == 'optimistic':
            q_vals = np.ones([4, 100], dtype=np.float32)
        elif self.hparam[ExperimentTaskSequenceRewardChange.HP_EXPLORATION] == 'egreedy':
            q_vals = np.zeros([4, 100], dtype=np.float32)
        lr = self.hparam[ExperimentTaskSequenceRewardChangeQLearning.HP_LEARNING_RATE]
        return rl.agent.QLearning(num_states=100, num_actions=4, learning_rate=lr, gamma=0.9, init_Q=q_vals)

    def _reset_agent(self, agent):
        agent.reset()


class ExperimentTaskSequenceRewardChangeQTransfer(ExperimentTaskSequenceRewardChangeQLearning):
    def _reset_agent(self, agent):
        pass


class ExperimentSetTaskSequenceRewardChange(ExperimentSet):

    def get_best_experiment(self, exploration='optimistic', task_sequence='slight'):
        exp_list = self.get_experiment_list_by_hparam({
            ExperimentTaskSequenceRewardChange.HP_EXPLORATION: exploration,
            ExperimentTaskSequenceRewardChange.HP_TASK_SEQUENCE: task_sequence
        })
        ep_len_list = [np.mean(exp.results['episode_length']) for exp in exp_list]
        best_idx = np.argmin(ep_len_list)
        exp = exp_list[best_idx]
        for k, v in exp.hparam.items():
            print('{}: {}'.format(k, v))
        return exp


class ExperimentSetTaskSequenceRewardChangeQLearning(ExperimentSetTaskSequenceRewardChange):

    def __init__(self, experiment_list):
        lr = ExperimentTaskSequenceRewardChangeQLearning.HP_LEARNING_RATE
        experiment_list.sort(key=lambda e: (e.hparam[lr]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRewardChangeQLearning.get_class_name()))
        return ExperimentSetTaskSequenceRewardChangeQLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        for lr, task_seq, expl in product(lr_list, ['significant'], ['optimistic', 'egreedy']):
            exp_list.append(ExperimentTaskSequenceRewardChangeQLearning({
                ExperimentTaskSequenceRewardChangeQLearning.HP_LEARNING_RATE: lr,
                ExperimentTaskSequenceRewardChangeQLearning.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRewardChangeQLearning.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRewardChangeQLearning(exp_list)


class ExperimentSetTaskSequenceRewardChangeQTransfer(ExperimentSetTaskSequenceRewardChange):

    def __init__(self, experiment_list):
        lr = ExperimentTaskSequenceRewardChangeQTransfer.HP_LEARNING_RATE
        experiment_list.sort(key=lambda e: (e.hparam[lr]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRewardChangeQTransfer.get_class_name()))
        return ExperimentSetTaskSequenceRewardChangeQTransfer(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        for lr, task_seq, expl in product(lr_list, ['significant'], ['optimistic', 'egreedy']):
            exp_list.append(ExperimentTaskSequenceRewardChangeQTransfer({
                ExperimentTaskSequenceRewardChangeQTransfer.HP_LEARNING_RATE: lr,
                ExperimentTaskSequenceRewardChangeQTransfer.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRewardChangeQTransfer.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRewardChangeQTransfer(exp_list)


# SF LEARNING
class ExperimentTaskSequenceRewardChangeSFLearning(ExperimentTaskSequenceRewardChange):
    HP_LEARNING_RATE_SF = 'lr_sf'
    HP_LEARNING_RATE_REWARD = 'lr_r'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRewardChangeSFLearning.HP_LEARNING_RATE_SF] = 0.5
        defaults[ExperimentTaskSequenceRewardChangeSFLearning.HP_LEARNING_RATE_REWARD] = 0.9
        return defaults

    def _construct_agent(self):
        if self.hparam[ExperimentTaskSequenceRewardChangeSFLearning.HP_EXPLORATION] == 'optimistic':
            init_sf_mat = np.eye(100 * 4, dtype=np.float32)
            init_w_vec = np.ones(100 * 4, dtype=np.float32)
        elif self.hparam[ExperimentTaskSequenceRewardChangeSFLearning.HP_EXPLORATION] == 'egreedy':
            init_sf_mat = np.zeros([100 * 4, 100 * 4], dtype=np.float32)
            init_w_vec = np.zeros(100 * 4, dtype=np.float32)
        lr_sf = self.hparam[ExperimentTaskSequenceRewardChangeSFLearning.HP_LEARNING_RATE_SF]
        lr_r = self.hparam[ExperimentTaskSequenceRewardChangeSFLearning.HP_LEARNING_RATE_REWARD]
        agent = SFLearning(
            num_states=100,
            num_actions=4,
            learning_rate_sf=lr_sf,
            learning_rate_reward=lr_r,
            gamma=0.9,
            init_sf_mat=init_sf_mat,
            init_w_vec=init_w_vec
        )
        return agent

    def _reset_agent(self, agent):
        agent.reset(reset_sf=True, reset_w=True)

class ExperimentTaskSequenceRewardChangeSFTransfer(ExperimentTaskSequenceRewardChangeSFLearning):
    def _reset_agent(self, agent):
        agent.reset(reset_sf=False, reset_w=True)


class ExperimentTaskSequenceRewardChangeSFTransferAll(ExperimentTaskSequenceRewardChangeSFLearning):
    def _reset_agent(self, agent):
        agent.reset(reset_sf=False, reset_w=False)


class ExperimentSetTaskSequenceRewardChangeSFLearning(ExperimentSetTaskSequenceRewardChange):

    def __init__(self, experiment_list):
        lr_sf = ExperimentTaskSequenceRewardChangeSFLearning.HP_LEARNING_RATE_SF
        lr_r = ExperimentTaskSequenceRewardChangeSFLearning.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRewardChangeSFLearning.get_class_name()))
        return ExperimentSetTaskSequenceRewardChangeSFLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        param_it = product(lr_list, lr_list, ['significant'], ['optimistic', 'egreedy'])
        for lr_sf, lr_r, task_seq, expl in param_it:
            exp_list.append(ExperimentTaskSequenceRewardChangeSFLearning({
                ExperimentTaskSequenceRewardChangeSFLearning.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentTaskSequenceRewardChangeSFLearning.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentTaskSequenceRewardChangeSFLearning.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRewardChangeSFLearning.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRewardChangeSFLearning(exp_list)


class ExperimentSetTaskSequenceRewardChangeSFTransfer(ExperimentSetTaskSequenceRewardChange):

    def __init__(self, experiment_list):
        lr_sf = ExperimentTaskSequenceRewardChangeSFTransfer.HP_LEARNING_RATE_SF
        lr_r = ExperimentTaskSequenceRewardChangeSFTransfer.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRewardChangeSFTransfer.get_class_name()))
        return ExperimentSetTaskSequenceRewardChangeSFTransfer(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        it = product(lr_list, lr_list, ['significant'], ['optimistic', 'egreedy'])
        for lr_sf, lr_r, task_seq, expl in it:
            exp_list.append(ExperimentTaskSequenceRewardChangeSFTransfer({
                ExperimentTaskSequenceRewardChangeSFTransfer.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentTaskSequenceRewardChangeSFTransfer.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentTaskSequenceRewardChangeSFTransfer.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRewardChangeSFTransfer.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRewardChangeSFTransfer(exp_list)


class ExperimentSetTaskSequenceRewardChangeSFTransferAll(ExperimentSetTaskSequenceRewardChange):

    def __init__(self, experiment_list):
        lr_sf = ExperimentTaskSequenceRewardChangeSFTransferAll.HP_LEARNING_RATE_SF
        lr_r = ExperimentTaskSequenceRewardChangeSFTransferAll.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRewardChangeSFTransferAll.get_class_name()))
        return ExperimentSetTaskSequenceRewardChangeSFTransferAll(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        it = product(lr_list, lr_list, ['significant'], ['optimistic', 'egreedy'])
        for lr_sf, lr_r, task_seq, expl in it:
            exp_list.append(ExperimentTaskSequenceRewardChangeSFTransferAll({
                ExperimentTaskSequenceRewardChangeSFTransferAll.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentTaskSequenceRewardChangeSFTransferAll.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentTaskSequenceRewardChangeSFTransferAll.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRewardChangeSFTransferAll.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRewardChangeSFTransferAll(exp_list)


def construct_experiment_set_by_name(experiment_set_name, **kwargs):
    return globals()[experiment_set_name].construct(**kwargs)
