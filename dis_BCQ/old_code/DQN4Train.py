'''
使用Tianshou提供的DQN, 在离散的gym环境中训练
'''
import os
import gymnasium as gym
import numpy as np
import torch
import tianshou as ts
from configs import DQNconfig
from tianshou.utils.net.common import Net

cfg = DQNconfig()
#get env info
env = gym.make(cfg.task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
#train/test env
train_envs = gym.make(cfg.task)
test_envs = gym.make(cfg.task)
#seed
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
train_envs.reset(seed=cfg.seed)
test_envs.reset(seed=cfg.seed)
#model
net = Net(state_shape, action_shape, cfg.hidden_sizes,
          device=cfg.device).to(cfg.device)
optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)
policy = ts.policy.DQNPolicy(
    net, optim, cfg.gamma, cfg.estm_steps, cfg.target_update_freq)
#buffer
buf = ts.data.VectorReplayBuffer(cfg.buffer_size, 1)
#collector
train_clct = ts.data.Collector(policy, train_envs, buf)
test_clct = ts.data.Collector(policy, test_envs)
#initialize
train_clct.collect(cfg.batch_size)


def save_best_fn(policy):
    torch.save(policy.state_dict(), cfg.save_path)


def stop_fn(r):
    return r >= cfg.reward_thresh


def train_fn(epoch, env_step):
    policy.set_eps(cfg.eps_train)


def test_fn(epoch, env_step):
    policy.set_eps(cfg.eps_test)


# trainer
res = ts.trainer.offpolicy_trainer(
    policy,
    train_clct,
    test_clct,
    cfg.epoch,
    cfg.step_per_epoch,
    cfg.S,
    cfg.test_num,
    cfg.batch_size,
    update_per_step = cfg.update_per_step,
    train_fn = train_fn,
    test_fn = test_fn,
    stop_fn = stop_fn,
    save_best_fn=save_best_fn
)

#eval
policy.eval()
policy.set_eps(cfg.eps_test)
collector = ts.data.Collector(policy, env)
result = collector.collect(n_episode=1)
rews, lens = result["rews"], result["lens"]
print(f"Final reward: {rews.mean()}, length: {lens.mean()}")