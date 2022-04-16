import torch
import numpy as np
from tqdm import tqdm
from BCQ import discrete_BCQ
import utils

#configs
dataset_dir = 'dataset'
seed = 42
is_atari = False
num_actions = 284
state_dim = 256+9+1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BCQ_threshold = 0.3
discount = 0.99
optimizer = "Adam"
optimizer_parameters = {"lr": 3e-4}
polyak_target_update = True
target_update_freq = 1
tau = 0.005
initial_eps = 0.05
end_eps = 0.05
eps_decay_period = 1
eval_eps = 0.001
batch_size = 128
buffer_size = 1e5
max_iter = 10000
loss_show_freq = 10
buffer_path = 'output/buffer'
model_path = 'output/'
#seed
np.random.seed(seed)
torch.manual_seed(seed)
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
#generate buffer with data
buffer = utils.ReplayBuffer(
    state_dim, is_atari, batch_size, buffer_size, device)
for idx in range(len(states)):
    buffer.add(states[idx], acts[idx], next_states[idx], rewards[idx],
               dones[idx],
               episode_done=dones[idx] == 1,
               episode_start=states[idx][-1] == 0)
print('successfully generated buffer!')
buffer.save(buffer_path)
#BCQ policy
policy = discrete_BCQ(
    is_atari,
    num_actions,
    state_dim,
    device,
    BCQ_threshold,
    discount,
    optimizer,
    optimizer_parameters,
    polyak_target_update,
    target_update_freq,
    tau,
    initial_eps,
    end_eps,
    eps_decay_period,
    eval_eps
)

with tqdm(range(max_iter)) as titer:
    for t in titer:
        titer.set_description(f"Train Iter {t}")
        loss = policy.train(replay_buffer=buffer)
        if t % loss_show_freq == 0:
            titer.set_postfix(loss=loss)

policy.save(model_path)

#offline evaluate