#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
import numpy as np
from itertools import product
import rlutils as rl
import os
import os.path as osp
import multiprocessing as mp

from .utils import cluster_idx_to_phi_mat
from .enumerate_partitions import enumerate_n_partitions
from .evaluate import eval_total_reward


def sample_cycle_mdp(num_states, num_actions, action_seq_sampler=None,
                     self_loop_prob=0.):
    if action_seq_sampler is None:
        action_seq = np.random.choice(np.arange(num_actions), size=num_states)
    else:
        action_seq = action_seq_sampler()

    t_mat = np.stack([np.eye(num_states) for a in range(num_actions)])
    for s, a in enumerate(action_seq):
        s_next = (s + 1) % num_states
        t_mat[a, s, s] = self_loop_prob
        t_mat[a, s, s_next] = 1. - self_loop_prob

    r_mat = np.zeros([num_actions, num_states, num_states])
    r_mat[action_seq[-1], num_states - 1, 0] = 1.

    return t_mat, r_mat


def sample_inflated_cycle_mdp(state_partition, num_actions):
    phi = cluster_idx_to_phi_mat(state_partition)
    num_s, num_s_lat = np.shape(phi)
    m_mat, w_mat = sample_cycle_mdp(num_s_lat, num_actions)
    num_a = np.shape(m_mat)[0]

    phi_mat_sum = np.sum(phi, axis=0, keepdims=True)
    phi_inv = np.transpose(phi / phi_mat_sum)
    phi_t = np.transpose(phi)

    t_mat = [np.matmul(phi, np.matmul(m_mat[a], phi_inv)) for a in range(num_a)]
    r_mat = [np.matmul(phi, np.matmul(w_mat[a], phi_t)) for a in range(num_a)]
    t_mat = np.stack(t_mat)
    r_mat = np.stack(r_mat)

    mdp = rl.environment.TabularMDP(
        t_mat=t_mat,
        r_mat=r_mat,
        idx_start_list=np.arange(num_s),
        idx_goal_list=[]
    )
    return mdp


def sample_cycle_mdp_sequence(partition_list, partition_idx_seq, num_actions=3,
                              gamma=0.9, sample_factor=20):
    task_dict = {}
    for (i, p_task), (j, p_test) in product(
            enumerate(partition_list[partition_idx_seq]),
            enumerate(partition_list)):
        if i not in task_dict.keys() and i != j:
            num_task_samples = len(partition_idx_seq) * sample_factor
            task_list = [sample_inflated_cycle_mdp(p_task, num_actions) for _ in
                         range(num_task_samples)]
            total_rew_list = [
                eval_total_reward(t, p_test, repeats=5, rollout_depth=10,
                                  gamma=gamma) for t in task_list]
            total_rew_list = np.sum(total_rew_list, axis=-1)
            task_dict[i] = task_list[np.argmin(total_rew_list)]
    task_list = [task_dict[i] for i in range(len(partition_idx_seq))]
    return task_list


def sample_cycle_mdp_sequence_dataset(num_states=9,
                                      num_latent_states=3,
                                      num_actions=3,
                                      gamma=0.9,
                                      sequence_length=20,
                                      num_sequences=100):
    partition_list = enumerate_n_partitions(num_states, num_latent_states)
    partition_list = partition_list[
        np.random.choice(np.shape(partition_list)[0], size=2)]
    partition_idx = [
        np.random.choice([0, 1], p=[0.75, 0.25], size=sequence_length) for _ in
        range(num_sequences)]
    param_list = [(partition_list, p, num_actions, gamma) for p in
                  partition_idx]
    with mp.Pool() as p:
        task_seq_dataset = p.starmap(sample_cycle_mdp_sequence, param_list)
    return task_seq_dataset, partition_idx, partition_list


def _task_seq_to_mat(task_seq):
    t_mat_seq = []
    r_mat_seq = []
    for task in task_seq:
        t_mat, r_mat = task.get_t_mat_r_mat()
        t_mat_seq.append(t_mat)
        r_mat_seq.append(r_mat)
    return np.stack(t_mat_seq), np.stack(r_mat_seq)


def generate_cycle_mdp_dataset(base_dir='./data/CycleMDPDataset/'):
    os.makedirs(base_dir, exist_ok=True)
    task_seq_dataset, partition_idx_seq_dataset, partition_list = sample_cycle_mdp_sequence_dataset()
    np.save(osp.join(base_dir, 'partition_list.npy'), partition_list)
    t_mat_seq_list = []
    r_mat_seq_list = []
    for task_seq in task_seq_dataset:
        t_mat_seq, r_mat_seq = _task_seq_to_mat(task_seq)
        t_mat_seq_list.append(t_mat_seq)
        r_mat_seq_list.append(r_mat_seq)
    np.save(osp.join(base_dir, 't_mat_seq_list.npy'), t_mat_seq_list)
    np.save(osp.join(base_dir, 'r_mat_seq_list.npy'), r_mat_seq_list)
    np.save(osp.join(base_dir, 'partition_idx_seq_dataset.npy'),
            partition_idx_seq_dataset)


def load_cycle_mdp_dataset(base_dir='./data/CycleMDPDataset/'):
    t_mat_seq_list = np.load(osp.join(base_dir, 't_mat_seq_list.npy'))
    r_mat_seq_list = np.load(osp.join(base_dir, 'r_mat_seq_list.npy'))
    partition_idx_seq_list = np.load(
        osp.join(base_dir, 'partition_idx_seq_dataset.npy'))
    partition_list = np.load(osp.join(base_dir, 'partition_list.npy'))

    mdp_dataset = []
    start_list = np.arange(np.shape(t_mat_seq_list)[-1])
    for t_mat_seq, r_mat_seq in zip(t_mat_seq_list, r_mat_seq_list):
        mdp_dataset.append(
            [rl.environment.TabularMDP(t, r, start_list, []) for t, r in
             zip(t_mat_seq, r_mat_seq)])

    return mdp_dataset, partition_idx_seq_list, partition_list


if __name__ == "__main__":
    generate_cycle_mdp_dataset()
    print('done.')
