'''
根据生成好的离线数据运行BCQ算法
'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gym
from tqdm import tqdm
from sklearn.metrics import f1_score
from configs import BCQconfig


class BCQNet(torch.nn.Module):
    def __init__(self, state_shape, hidden_dims, action_shape):
        super(BCQNet, self).__init__()
        nets = [torch.nn.Linear(state_shape, hidden_dims[0]),
                torch.nn.ReLU(inplace=True)]
        for i in range(len(hidden_dims) - 1):
            nets.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            nets.append(torch.nn.ReLU(inplace=True))
        nets.append(torch.nn.Linear(hidden_dims[-1], action_shape))

        self.model = torch.nn.Sequential(*nets)

    def forward(self, x):
        return self.model(x)


class Generative(torch.nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Generative, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_shape, 128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, action_shape)
        )

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)


class BCQ():
    def __init__(self, config):
        self.cfg = config
        #seed
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.env = gym.make(self.cfg.task)
        self.state_shape = np.prod(self.env.observation_space.shape)
        self.action_shape = self.env.action_space.n
        #Q网络和Target-Q网络，以及Q网络的优化器
        self.Q_net = BCQNet(self.state_shape, self.cfg.hidden_dims,
                            self.action_shape).to(self.cfg.device)
        self.Target_Q = BCQNet(
            self.state_shape, self.cfg.hidden_dims, self.action_shape).to(self.cfg.device)
        #生成模型
        self.G_w = Generative(
            self.state_shape, self.action_shape).to(self.cfg.device)

    def __take_action(self, state, force_random=False):
        if force_random:
            return np.random.randint(self.action_shape)
        #epsilon-greedy
        if np.random.random() < self.cfg.eps:
            action = np.random.randint(self.action_shape)
        else:
            #state = torch.tensor([state]).to(self.cfg.device)
            with torch.no_grad():
                thresh = torch.max(self.G_w(state), 1)[
                    0]*self.cfg.constr_thresh
                q_values = self.Q_net(state)
                #select max a' for every s' where G_w(a'|s') > thresh（s'）
                #to complete this, we can do so:
                #argmax ((G_w(s') > thresh(s'))*2(MAX Q - MIN Q) + Q(s'))
                thresh = thresh.reshape(-1, 1)
                thresh = thresh.repeat(1, self.action_shape)
                g_w_ns = self.G_w(state)
                q_values += torch.ge(g_w_ns, thresh) * \
                    (2*q_values.max().item() - 2*q_values.min().item())
                action = torch.max(q_values, 1)[1].item()
        return action

    def __load_dataset(self):
        self.state_buf = torch.from_numpy(
            np.load(self.cfg.state_path)).float().to(self.cfg.device)
        self.act_buf = torch.from_numpy(
            np.load(self.cfg.act_path)).to(self.cfg.device)
        self.nstate_buf = torch.from_numpy(
            np.load(self.cfg.nstate_path)).float().to(self.cfg.device)
        self.rew_buf = torch.from_numpy(
            np.load(self.cfg.rew_path)).float().to(self.cfg.device)
        self.done_buf = torch.from_numpy(
            np.load(self.cfg.done_path)).to(self.cfg.device)
        self.buffer = TensorDataset(
            self.state_buf, self.act_buf, self.nstate_buf, self.rew_buf, self.done_buf)
        self.buf_size = len(self.buffer)
        print('dataset loaded, %d in total' % self.buf_size)

    """
    def __query_buffer(self, state, action, all_result=False):
        '''
        fetch s',r from buffer with s,a
        '''
        state_ = state.reshape(1, len(state)).repeat(self.buf_size, 1)
        action_ = torch.tensor([action]*self.buf_size).to(self.cfg.device)
        idxs = torch.min(torch.isclose(self.state_buf, state_), 1)[0]
        idxs &= torch.eq(self.act_buf, action_)
        NSs = deepcopy(self.nstate_buf[idxs])
        REWs = deepcopy(self.rew_buf[idxs])
        Dones = deepcopy(self.done_buf[idxs])
        if len(REWs) == 0:
            return None, None, None
        elif all_result:
            return NSs, REWs, Dones
        else:
            idx = np.random.randint(len(NSs))
            return NSs[idx], REWs[idx], Dones[idx]
    """

    def __fit_generative(self):
        dataloader = DataLoader(
            self.buffer, batch_size=self.cfg.G_batch_size, drop_last=True)
        criterion = nn.NLLLoss()
        optim = torch.optim.Adam(
            self.G_w.parameters(), lr=self.cfg.lr, eps=self.cfg.adam_eps)
        print('fitting Generative...')
        with tqdm(range(self.cfg.G_max_epoch)) as tepoch:
            for epoch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                for states, actions,  _, _, _ in dataloader:
                    optim.zero_grad()
                    out = self.G_w(states)
                    loss = criterion(out, actions)
                    loss.backward()
                    optim.step()
                    l = loss.data.item()
                tepoch.set_postfix(loss=l)
        self.__save_generative()

    def __eval_generative(self):
        #这里的Eval应该用一个预先分割好的测试集
        with torch.no_grad():
            self.G_w.eval()
            #idxs = np.random.choice(
            #    np.arange(self.buf_size), 1000, replace=False)
            states, actions, _, _, _ = self.buffer[:-1000]
            outputs = self.G_w(states)
            _, predicted = torch.max(outputs, 1)
            print('Genrative f1-score:', f1_score(actions.cpu(), predicted.cpu()))

    def __save_generative(self):
        torch.save(self.G_w.state_dict(), self.cfg.model_path+'genrt.ckp')

    def __load_generative(self):
        self.G_w.load_state_dict(torch.load(self.cfg.model_path+'genrt.ckp'))

    def train(self, train_G_w=True):
        self.__load_dataset()
        if train_G_w:
            self.__fit_generative()
            self.__eval_generative()
        else:
            self.__load_generative()
            self.__eval_generative()
        """
        def reset():
            while True:
                #randomly pick a state as reset value
                state = self.state_buf[np.random.randint(self.buf_size)]
                act = self.__take_action(state, force_random=False)
                rew, nstate, done = self.__query_buffer(
                    state, act, all_result=False)
                if rew != None:
                    break
            return state, act, rew, nstate, done

        state, act, rew, nstate, done = reset()
        """
        optim = torch.optim.Adam(
            self.Q_net.parameters(), lr=self.cfg.lr, eps=self.cfg.adam_eps)
        sum_loss = 0
        with tqdm(range(self.cfg.max_iter)) as titer:
            for t in titer:
                titer.set_description(f"Train Iter {t}")
                """
                while done:
                    state, act, rew, nstate, done = reset()
                while True:
                    NSs, REWs, Dones = self.__query_buffer(
                        state, act, all_result=True)
                    if NSs != None:
                        break
                    state, act, rew, nstate, done = reset()
                """
                #sample from return
                idxs = np.random.choice(
                    np.arange(self.buf_size), self.cfg.mini_batch_size)
                Ss, Acts, NSs, REWs, Dones = self.buffer[idxs]
                #get a'
                #constraint: G_w(a'|s') > max G_w(s') * tau
                with torch.no_grad():
                    thresh = torch.max(self.G_w(NSs), 1)[
                        0]*self.cfg.constr_thresh
                    q_values = self.Q_net(NSs)
                    #select max a' for every s' where G_w(a'|s') > thresh（s'）
                    #to complete this, we can do so:
                    #argmax ((G_w(s') > thresh(s'))*2(MAX Q - MIN Q) + Q(s'))
                    thresh = thresh.reshape(-1, 1)
                    thresh = thresh.repeat(1, self.action_shape)
                    g_w_ns = self.G_w(NSs)
                    q_values += torch.ge(g_w_ns, thresh) * \
                        (2*q_values.max().item() - 2*q_values.min().item())
                    NAs = torch.max(q_values, 1)[1]
                    #target_Q
                    NQ_vs =REWs.reshape(-1, 1) + self.cfg.gamma * \
                        ~Dones.reshape(-1, 1)*self.Target_Q(NSs).gather(1, NAs.reshape(-1, 1))
                #train Q-net
                optim.zero_grad()
                Q_vs = self.Q_net(Ss).gather(1, Acts.reshape(-1, 1))
                l = F.smooth_l1_loss(NQ_vs, Q_vs)
                l.backward()
                sum_loss += l.item()
                optim.step()
                #update rate
                if t % self.cfg.target_update_rate:
                    self.Target_Q.load_state_dict(self.Q_net.state_dict())
                if t % self.cfg.show_loss_freq == 0:
                    titer.set_postfix(loss=sum_loss/self.cfg.show_loss_freq)
                    sum_loss = 0
        self.__save_BCQ()

    def eval(self, env):
        self.__load_BCQ()
        self.__load_generative()
        self.Q_net.eval()
        env.reset(seed=self.cfg.seed)
        res = []
        def transfm(x): return torch.from_numpy(
            x.reshape(1, -1)).float().to(self.cfg.device)
        with tqdm(range(self.cfg.eval_episodes)) as tepisodes:
            for episode in tepisodes:
                tepisodes.set_description(f"Test Iter {episode}")
                state = transfm(env.reset())
                episode_rew = 0
                done = False
                while not done:
                    act = self.__take_action(state)
                    nstate, rew, done, _ = env.step(act)
                    state = transfm(nstate)
                    episode_rew += rew
                res.append(episode_rew)
        print('Average reward: %f , Max reward: %f .' %
              (sum(res)/len(res), max(res)))

    def __save_BCQ(self):
        torch.save(self.Q_net.state_dict(), self.cfg.model_path+'QNet.ckp')
        torch.save(self.Target_Q.state_dict(),
                   self.cfg.model_path+'TargetQ.ckp')

    def __load_BCQ(self):
        self.Q_net.load_state_dict(torch.load(self.cfg.model_path+'QNet.ckp'))
        self.Target_Q.load_state_dict(
            torch.load(self.cfg.model_path+'TargetQ.ckp'))


if __name__ == '__main__':
    cfg = BCQconfig()
    bcq = BCQ(cfg)
    bcq.train(train_G_w = False)
    env = gym.make('CartPole-v1')
    bcq.eval(env=env)
