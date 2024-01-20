'''
使用训练好的DQN模型, 进行测试
'''
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
#seed
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
env.reset(seed=cfg.seed)
#model
net = Net(state_shape, action_shape, cfg.hidden_sizes,
          device=cfg.device).to(cfg.device)
optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)
policy = ts.policy.DQNPolicy(
    net, optim, cfg.gamma, cfg.estm_steps, cfg.target_update_freq)
#load trained policy
policy.load_state_dict(torch.load(cfg.save_path))
#eval
policy.eval()
policy.set_eps(cfg.eps_test)
collector = ts.data.Collector(policy, env)
result = collector.collect(n_episode=cfg.eval_episodes)
rews, lens = result["rews"], result["lens"]
print(f"Average reward: {rews.mean()}, max reward: {rews.max()}")