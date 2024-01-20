'''
使用训练好的DQN模型，生成用于offline RL的数据集
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
#run on env
buf = ts.data.VectorReplayBuffer(total_size=cfg.out_buf_size,buffer_num=1)
clct = ts.data.Collector(policy, env, buf)
clct.collect(n_episode=cfg.output_episode)
#save data
max_valid = np.where(buf.done==True)[0][-1] + 1
np.save(open(cfg.act_path,'wb'),buf.act[:max_valid])
np.save(open(cfg.state_path,'wb'),buf.obs[:max_valid])
np.save(open(cfg.nstate_path,'wb'),buf.obs_next[:max_valid])
np.save(open(cfg.rew_path,'wb'),buf.rew[:max_valid])
np.save(open(cfg.done_path,'wb'),buf.done[:max_valid])