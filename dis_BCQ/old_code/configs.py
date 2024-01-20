import os

for dir_name in ['dqn', 'bcq']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

class DQNconfig():
    def __init__(self):
        #self.task='AirRaid-v4'
        self.task = 'CartPole-v1'
        self.device = 'cuda'
        self.hidden_sizes = [256,256,256]
        self.lr=1e-3
        self.gamma = 0.99
        ## n步估计
        self.estm_steps=2
        ## 目标网络更新频率
        self.target_update_freq=50
        self.buffer_size = 20000
        self.reward_thresh = 500
        self.batch_size = 256
        self.save_path = './dqn/model.ckp'
        ## ε-greedy
        self.eps_train = 0.1
        self.eps_test = 0.05
        self.epoch = 10
        self.step_per_epoch = 10000
        ## 算法中gradient step的部分
        self.G = 5
        ## 算法中的S
        self.S = 10
        self.update_per_step = self.G/self.S
        self.test_num = 100
        self.seed = 42
        ## eval的轮数
        self.eval_episodes = 100
        ## dqn4buffer
        self.output_episode = 1000
        self.out_buf_size = 1e6
        self.act_path = './dqn/act.npy'
        self.state_path = './dqn/state.npy'
        self.nstate_path = './dqn/state_next.npy'
        self.rew_path = './dqn/reward.npy'
        self.done_path = './dqn/done.npy'


class BCQconfig():
    def __init__(self):
        self.seed = 42
        self.act_path = './dqn/act.npy'
        self.state_path = './dqn/state.npy'
        self.nstate_path = './dqn/state_next.npy'
        self.rew_path = './dqn/reward.npy'
        self.done_path = './dqn/done.npy'
        self.task = 'CartPole-v1'
        self.hidden_dims = [256,256,256,256]
        self.mini_batch_size = 256
        self.constr_thresh = 0.3
        self.target_update_rate = 20
        #self.loss_kappa = 1
        self.eps = 0.001
        self.lr = 5e-5
        self.adam_eps = 0.00015
        self.max_iter = 5001
        self.gamma = 0.9
        self.enc_h = 128
        self.enc_out = 128
        self.dec_in = 8
        self.dec_h = 8
        self.G_batch_size = 4096
        self.G_max_epoch = 100
        self.model_path = './bcq/'
        self.device= 'cuda'
        self.eval_episodes = 100
        self.show_loss_freq = 20