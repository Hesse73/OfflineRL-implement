import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import tianshou as ts
from tianshou.utils.net.discrete import Actor
from tianshou.utils.net.common import ActorCritic

#Since the data has already been preprocessed(by embedding)
#the preprocess model here is just forward it.
class preprocess_net(nn.Module):
    def __init__(self, device='cpu'):
        super(preprocess_net, self).__init__()
        self.device = device

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        logits = obs
        return logits, state

#configs
dataset_dir = 'dataset'
state_shape = 256+9+1  # embedding + 9 items + counter
action_shape = 1  # item id
episode_len = 9+1
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_sizes = [256, 256, 256]
lr = 5e-5
gamma = 0.99
#future steps when estimate Q value for learning
estimation_step = 1
target_update_freq = 8000
eps_test = 0.0001
#tau threshold for G_w in paper
unlikely_action_threshold = 0.3
#imitation regularization weight
imitation_logits_penalty = 0.01
#seed
np.random.seed(seed)
torch.manual_seed(seed)
#model
feature_net = preprocess_net().to(device)
policy_net = Actor(feature_net, action_shape, device=device,
                   hidden_sizes=hidden_sizes, softmax_output=False).to(device)
imitation_net = Actor(feature_net, action_shape, device=device,
                      hidden_sizes=hidden_sizes, softmax_output=False).to(device)
actor_critic = ActorCritic(policy_net, imitation_net)
optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)
#policy
policy = ts.policy.DiscreteBCQPolicy(
    policy_net, imitation_net, optim, gamma,
    estimation_step, target_update_freq, 
    eps_test, unlikely_action_threshold,
     imitation_logits_penalty
)
#load data generated for offline learning
states = np.load(open(dataset_dir+'/obs.npy', 'rb'))
acts = np.load(open(dataset_dir+'/act.npy', 'rb'))
rewards = np.load(open(dataset_dir+'/rew.npy', 'rb'))
dones = np.load(open(dataset_dir+'/ter.npy', 'rb'))
#generate next_states
print('generating next states')
next_states = np.zeros_like(states)
for idx in range(len(states)):
    if dones[idx] != 1:
        next_states[idx] = states[idx+1].copy()
#np.save(open(dataset_dir+'/nextobs.npy','wb'),next_states)
#load data to tianshou buffer
buffer = ts.data.ReplayBuffer(size=len(states), ignore_obs_next=True)
for idx in range(len(states)):
    buffer.add(ts.data.Batch(obs=states[idx], act=acts[idx], rew=rewards[idx],
                          done=dones[idx], obs_next=next_states[idx], info={}))
print('successfully generated buffer!')
#collector
clct = ts.data.Collector(policy,)