import os
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tianshou as ts
from tqdm import tqdm

parser = argparse.ArgumentParser()
# environment setting
parser.add_argument('--env_name', type=str, default='MountainCar-v0')
parser.add_argument('--gamma', type=float, default=0.99)
# BCQ policy setting
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 256, 256])
parser.add_argument('--test_eps', type=float, default=0.01)
parser.add_argument('--target_update_freq', type=int, default=100)
parser.add_argument('--tau', type=float, default=0.3, help='BCQ threshold')
# BCQ training setting
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--max_iter', type=int, default=3000)
# others
parser.add_argument('--save_dir', type=str, default='./dqn')
parser.add_argument('--eval_episodes', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)


class BCQNet(nn.Module):
    def __init__(self, state_dim:int, hidden_dims:list, num_actions:int):
        super(BCQNet, self).__init__()
        in_dims = [state_dim] + hidden_dims[:-1]
        q_nets, imt_nets = [], []
        for in_dim, out_dim in zip(in_dims, hidden_dims):
            q_nets.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
            imt_nets.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        q_nets.append(nn.Linear(hidden_dims[-1], num_actions))
        imt_nets.append(nn.Linear(hidden_dims[-1], num_actions))
        # q network
        self.q_net = nn.Sequential(*q_nets)
        # imtation network
        self.imt_net = nn.Sequential(*imt_nets)
    
    def forward(self, state):
        q = self.q_net(state)
        imt = self.imt_net(state)
        likelihood = imt.softmax(dim=-1)  # likelihood
        return q, likelihood, imt
    

class discrete_BCQ:
    def __init__(self,
                 state_dim,
                 num_actions,
                 hidden_dims,
                 device,
                 lr=1e-3,
                 batch_size=64,
                 max_iter=1000,
                 tau=0.3,
                 gamma=0.99,
                 target_update_freq=50,
                 test_eps=0.05
                 ):
        # save args
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.test_eps = test_eps
        
        # model & optimizer
        self.Q = BCQNet(state_dim, hidden_dims, num_actions).to(device)
        self.Q_target = BCQNet(state_dim, hidden_dims, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        
        
    def train(self, offline_buffer):
        titer = tqdm(range(self.max_iter), desc='Train')
        sum_loss = 0
        for iter in titer:
            # sample
            data, indices = offline_buffer.sample(self.batch_size)
            data.to_torch(device=self.device)
            
            # compute target q value (select action by current q)
            with torch.no_grad():
                q, likelihood, imt = self.Q(data.obs_next)
                is_valid = likelihood/likelihood.max(dim=1, keepdim=True)[0] > self.tau
                next_action = torch.where(is_valid, q, q.min() -1).argmax(dim=1, keepdim=True)
                
                q, _, _ = self.Q_target(data.obs_next)
                target_Q = data.rew + data.done * self.gamma * q.gather(1, next_action).flatten()
            
            # current q value
            current_Q, _, imt = self.Q(data.obs)
            current_Q = current_Q.gather(1, data.act.unsqueeze(-1)).flatten()
            
            # compute loss
            q_loss = F.smooth_l1_loss(current_Q, target_Q)
            imt_loss = F.nll_loss(input=imt, target=data.act)
            
            loss = q_loss + imt_loss +  + 1e-2 * imt.pow(2).mean()
            
            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # update target
            if iter % self.target_update_freq == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())
                
                
            sum_loss += loss.item()
            titer.set_postfix_str(f'loss={sum_loss / (iter + 1):.3f}')
        
    @torch.no_grad()
    def select_action(self, state):
        # epsilon greedy
        if np.random.uniform(0,1) > self.test_eps:
            q, likelihood, _ = self.Q(state)
            is_valid = likelihood/likelihood.max(dim=1, keepdim=True)[0] > self.tau
            return torch.where(is_valid, q, q.min() -1).argmax(dim=1).cpu().item()
        else:
            return np.random.randint(self.num_actions)
        
    @torch.no_grad()
    def eval(self, env, eval_episodes):
        res = []
        transfm = lambda x: torch.from_numpy(x).view(1,-1).float().to(self.device)
        titer = tqdm(range(eval_episodes), desc='Evaluate')
        for episode in titer:
            state = transfm(env.reset()[0])  # env.reset() returns state, info
            sum_rew, done = 0, False
            while not done:
                act = self.select_action(state)
                nstate, rew, terminated, truncated, _ = env.step(act)
                done = truncated or terminated 
                state = transfm(nstate)
                sum_rew += rew
            res.append(sum_rew)
            titer.set_postfix_str(f'avg reward={sum(res)/(episode+1):.3f}')
        print('Average reward: %f , Max reward: %f .' %
              (sum(res)/len(res), max(res)))
        
    @torch.no_grad()
    def show(self, env):
        transfm = lambda x: torch.from_numpy(x).view(1,-1).float().to(self.device)
        state = transfm(env.reset(seed=42)[0])
        done = False
        while not done:
            act = self.select_action(state)
            nstate, rew, terminated, truncated, _ = env.step(act)
            done = truncated or terminated 
            state = transfm(nstate)
            

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env info
    env = gym.make(args.env_name)
    state_shape = env.observation_space.shape[0] or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    # load dataset
    dataset_path = os.path.join(args.save_dir, 'offline_dataset')
    offline_buffer = ts.data.VectorReplayBuffer.load_hdf5(dataset_path)
    print('Loading offline dataset, size:', len(offline_buffer))
    # train BCQ
    bcq = discrete_BCQ(state_dim=state_shape, num_actions=action_shape, 
                       hidden_dims=args.hidden_dims, device=device,
                       lr=args.lr, batch_size=args.batch_size, 
                       max_iter=args.max_iter, tau=args.tau, gamma=args.gamma,
                       target_update_freq=args.target_update_freq,
                       test_eps=args.test_eps) 
    bcq.train(offline_buffer)
    # eval BCQ
    bcq.eval(env, eval_episodes=args.eval_episodes)
    # show BCQ's running
    show_env = gym.make(args.env_name, render_mode='human')
    bcq.show(show_env)