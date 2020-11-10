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
import yaml
from collections import Generator

from .evaluate import eval_reward_predictive
from .evaluate import eval_total_reward
from .utils import SFLearning, EGreedyScheduleUpdate
from .mdp import *
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


class ExperimentTaskSequenceRandomRewardChange(ExperimentHParamParallel):
    HP_EXPLORATION = 'exploration'
    HP_TASK_SEQUENCE = 'task_sequence'
    HP_NUM_EPISODES = 'episodes'

    def __init__(self, *params, num_tasks=None, **kwargs):
        """
        Experiment task sequence for our random reward change experiments.

        Biggest difference to the normal RewardChange class is how we yield tasks -
        we use Python generators! Look below as to how that works.
        We use generators here (instead of instantiating num_tasks environments) in order to
        potentially generate an endless number of tasks (Don't do this yet things will break!).
        :param params: params to pass in
        :param kwargs: dict params to pass in
        """
        super().__init__(*params, **kwargs)
        self.num_tasks = num_tasks or float('inf')
        self.task_sequence = self._get_task_sequence()

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_REPEATS] = 20
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_TASK_SEQUENCE] = 'slight'
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_EXPLORATION] = 'optimistic'
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_NUM_EPISODES] = 200
        return defaults

    def _get_task_sequence(self) -> Generator:
        """
        We get a sequence of tasks generated on-the-fly.
        The function returns a generator (https://wiki.python.org/moin/Generators) that
        instantiates new environments when iterated over.
        Each instantiation of the RandomRewardChange environment instantiates a new environment
        with a randomized reward position.
        """
        num_tasks = 0
        while num_tasks < self.num_tasks:
            num_tasks += 1
            yield RandomRewardChange()

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
