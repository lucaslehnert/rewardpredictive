import numpy as np
import rlutils as rl

from .utils import cluster_idx_to_phi_mat, lam_from_mat, reward_rollout


def eval_total_reward(task, partition, repeats=5, rollout_depth=10, gamma=0.9):
    """
    Evaluate the total reward that can be generated by using the given abstraction on the specified task.

    :param task:
    :param partition:
    :param repeats:
    :param rollout_depth:
    :param gamma:
    :return:
    """
    phi_mat = cluster_idx_to_phi_mat(partition)
    t_mat, r_vec = task.get_t_mat_r_vec()
    m_mat, w_vec = lam_from_mat(t_mat, r_vec, phi_mat)
    q_phi, _ = rl.algorithm.vi(m_mat, w_vec, gamma=gamma)

    latent_agent = rl.agent.ValueFunctionAgent(q_fun=lambda s: np.matmul(q_phi, s))
    agent = rl.agent.StateRepresentationWrapperAgent(latent_agent, phi=lambda s: np.matmul(s, phi_mat))
    policy = rl.policy.GreedyPolicy(agent)

    logger = rl.logging.LoggerTotalReward()
    for _ in range(repeats):
        rl.data.simulate_gracefully(task, policy, logger, max_steps=rollout_depth)
        # logger.on_simulation_timeout()

    return logger.get_total_reward_episodic()


def eval_reward_predictive(task, partition, repeats=5, rollout_depth=10):
    phi_mat = cluster_idx_to_phi_mat(partition)
    t_mat, r_vec = task.get_t_mat_r_vec()
    m_mat, w_vec = lam_from_mat(t_mat, r_vec, phi_mat)

    action_seq_list = np.random.randint(0, task.num_actions(), size=[repeats, rollout_depth])
    start_state_list = [rl.one_hot(i, task.num_states()) for i in np.random.randint(0, task.num_states(), size=repeats)]
    start_state_list = np.stack(start_state_list).astype(dtype=np.float32)
    rew_err = []
    for i in range(repeats):
        s = start_state_list[i]
        action_seq = action_seq_list[i]
        rew_original = reward_rollout(t_mat, r_vec, s, action_seq)
        rew_latent = reward_rollout(m_mat, w_vec, np.matmul(s, phi_mat), action_seq)
        rew_err.append(np.abs(rew_original - rew_latent))

    return np.array(rew_err, dtype=np.float32)
