import rlutils as rl
import numpy as np
from itertools import product
from os import path as osp
from abc import abstractmethod
from tqdm import tqdm
# from collections import Generator
from .mdp import RandomRewardChange
from .significant_experiment import (
    ExperimentSet,
    ExperimentHParamParallel,
    ExperimentHParam,
    ExperimentTaskSequenceRewardChange,
    _load_experiment_list
)
from .utils import set_seeds, simulate_episodes, SFLearning, EGreedyScheduleUpdate


class ExperimentTaskSequenceRandomRewardChange(ExperimentHParamParallel):
# class ExperimentTaskSequenceRandomRewardChange(ExperimentHParam):

    HP_EXPLORATION = 'exploration'
    HP_TASK_SEQUENCE = 'task_sequence'
    HP_NUM_EPISODES = 'episodes'

    def __init__(self, *params, num_tasks=10, **kwargs):

        """
        Experiment task sequence for our random reward change experiments.

        Biggest difference to the normal RewardChange class is how we yield tasks -
        # we use Python generators! Look below as to how that works.
        # We use generators here (instead of instantiating num_tasks environments) in order to
        # potentially generate an endless number of tasks (Don't do this yet things will break!).
        :param params: params to pass in
        :param kwargs: dict params to pass in
        """
        super().__init__(*params, **kwargs)
        self.num_tasks = num_tasks
        self.task_sequence = self._get_task_sequence()

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_REPEATS] = 20
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_TASK_SEQUENCE] = 'slight'
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_EXPLORATION] = 'optimistic'
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_NUM_EPISODES] = 100
        return defaults

    # def _get_task_sequence(self) -> Generator:
    #     """
    #     We get a sequence of tasks generated on-the-fly.
    #     The function returns a generator (https://wiki.python.org/moin/Generators) that
    #     instantiates new environments when iterated over.
    #     Each instantiation of the RandomRewardChange environment instantiates a new environment
    #     with a randomized reward position.
    #     """
    #     num_tasks = 0
    #     while num_tasks < self.num_tasks:
    #         num_tasks += 1
    #         yield RandomRewardChange()
    def _get_task_sequence(self):
        pbar = tqdm(range(self.num_tasks))
        mdp_seq = []
        for i in pbar:
            mdp_seq.append(RandomRewardChange())
            pbar.set_description(f"Creating env #{i}")
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
        for task in tqdm(self.task_sequence):
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


class ExperimentTaskSequenceRandomRewardChangeQLearning(ExperimentTaskSequenceRandomRewardChange):
    HP_LEARNING_RATE = 'lr'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRandomRewardChangeQLearning.HP_LEARNING_RATE] = 0.9
        return defaults

    def _construct_agent(self):
        if self.hparam[ExperimentTaskSequenceRewardChange.HP_EXPLORATION] == 'optimistic':
            q_vals = np.ones([4, 100], dtype=np.float32)
        elif self.hparam[ExperimentTaskSequenceRewardChange.HP_EXPLORATION] == 'egreedy':
            q_vals = np.zeros([4, 100], dtype=np.float32)
        lr = self.hparam[ExperimentTaskSequenceRandomRewardChangeQLearning.HP_LEARNING_RATE]
        return rl.agent.QLearning(num_states=100, num_actions=4, learning_rate=lr, gamma=0.9, init_Q=q_vals)

    def _reset_agent(self, agent):
        agent.reset()


class ExperimentTaskSequenceRandomRewardChangeQTransfer(ExperimentTaskSequenceRandomRewardChangeQLearning):
    def _reset_agent(self, agent):
        pass


class ExperimentSetTaskSequenceRandomRewardChange(ExperimentSet):

    def get_best_experiment(self, exploration='optimistic', task_sequence='slight'):
        exp_list = self.get_experiment_list_by_hparam({
            ExperimentTaskSequenceRandomRewardChange.HP_EXPLORATION: exploration,
            ExperimentTaskSequenceRandomRewardChange.HP_TASK_SEQUENCE: task_sequence
        })
        ep_len_list = [np.mean(exp.results['episode_length']) for exp in exp_list]
        best_idx = np.argmin(ep_len_list)
        exp = exp_list[best_idx]
        for k, v in exp.hparam.items():
            print('{}: {}'.format(k, v))
        return exp


class ExperimentSetTaskSequenceRandomRewardChangeQLearning(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr = ExperimentTaskSequenceRandomRewardChangeQLearning.HP_LEARNING_RATE
        experiment_list.sort(key=lambda e: (e.hparam[lr]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRandomRewardChangeQLearning.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeQLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        for lr, task_seq, expl in product(lr_list, ['significant'], ['egreedy']):
            exp_list.append(ExperimentTaskSequenceRandomRewardChangeQLearning({
                ExperimentTaskSequenceRandomRewardChangeQLearning.HP_LEARNING_RATE: lr,
                ExperimentTaskSequenceRandomRewardChangeQLearning.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRandomRewardChangeQLearning.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeQLearning(exp_list)


class ExperimentSetTaskSequenceRandomRewardChangeQTransfer(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr = ExperimentTaskSequenceRandomRewardChangeQTransfer.HP_LEARNING_RATE
        experiment_list.sort(key=lambda e: (e.hparam[lr]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRandomRewardChangeQTransfer.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeQTransfer(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        for lr, task_seq, expl in product(lr_list, ['significant'], ['egreedy']):
            exp_list.append(ExperimentTaskSequenceRandomRewardChangeQTransfer({
                ExperimentTaskSequenceRandomRewardChangeQTransfer.HP_LEARNING_RATE: lr,
                ExperimentTaskSequenceRandomRewardChangeQTransfer.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRandomRewardChangeQTransfer.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeQTransfer(exp_list)


# SF LEARNING
class ExperimentTaskSequenceRandomRewardChangeSFLearning(ExperimentTaskSequenceRandomRewardChange):
    HP_LEARNING_RATE_SF = 'lr_sf'
    HP_LEARNING_RATE_REWARD = 'lr_r'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_SF] = 0.5
        defaults[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_REWARD] = 0.9
        return defaults

    def _construct_agent(self):
        if self.hparam[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_EXPLORATION] == 'optimistic':
            init_sf_mat = np.eye(100 * 4, dtype=np.float32)
            init_w_vec = np.ones(100 * 4, dtype=np.float32)
        elif self.hparam[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_EXPLORATION] == 'egreedy':
            init_sf_mat = np.zeros([100 * 4, 100 * 4], dtype=np.float32)
            init_w_vec = np.zeros(100 * 4, dtype=np.float32)
        lr_sf = self.hparam[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_SF]
        lr_r = self.hparam[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_REWARD]
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

class ExperimentTaskSequenceRandomRewardChangeSFTransfer(ExperimentTaskSequenceRandomRewardChangeSFLearning):
    def _reset_agent(self, agent):
        agent.reset(reset_sf=False, reset_w=True)


class ExperimentTaskSequenceRewardChangeSFTransferAll(ExperimentTaskSequenceRandomRewardChangeSFLearning):
    def _reset_agent(self, agent):
        agent.reset(reset_sf=False, reset_w=False)


class ExperimentSetTaskSequenceRandomRewardChangeSFLearning(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr_sf = ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_SF
        lr_r = ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRandomRewardChangeSFLearning.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeSFLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        param_it = product(lr_list, lr_list, ['significant'], ['egreedy'])
        for lr_sf, lr_r, task_seq, expl in param_it:
            exp_list.append(ExperimentTaskSequenceRandomRewardChangeSFLearning({
                ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeSFLearning(exp_list)


class ExperimentSetTaskSequenceRandomRewardChangeSFTransfer(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr_sf = ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_LEARNING_RATE_SF
        lr_r = ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRandomRewardChangeSFTransfer.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeSFTransfer(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        it = product(lr_list, lr_list, ['significant'], ['egreedy'])
        for lr_sf, lr_r, task_seq, expl in it:
            exp_list.append(ExperimentTaskSequenceRandomRewardChangeSFTransfer({
                ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeSFTransfer(exp_list)


class ExperimentSetTaskSequenceRandomRewardChangeSFTransferAll(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr_sf = ExperimentTaskSequenceRewardChangeSFTransferAll.HP_LEARNING_RATE_SF
        lr_r = ExperimentTaskSequenceRewardChangeSFTransferAll.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRewardChangeSFTransferAll.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeSFTransferAll(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        exp_list = []
        lr_list = [.1, .3, .5, .7, .9]
        it = product(lr_list, lr_list, ['significant'], ['egreedy'])
        for lr_sf, lr_r, task_seq, expl in it:
            exp_list.append(ExperimentTaskSequenceRewardChangeSFTransferAll({
                ExperimentTaskSequenceRewardChangeSFTransferAll.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentTaskSequenceRewardChangeSFTransferAll.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentTaskSequenceRewardChangeSFTransferAll.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRewardChangeSFTransferAll.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeSFTransferAll(exp_list)


def construct_experiment_set_by_name(experiment_set_name, **kwargs):
    return globals()[experiment_set_name].construct(**kwargs)
