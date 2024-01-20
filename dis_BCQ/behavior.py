import os
import argparse
import gymnasium as gym
import numpy as np
import torch
import tianshou as ts
from tianshou.utils.net.common import Net

parser = argparse.ArgumentParser()
# environment setting
parser.add_argument('--env_name', type=str, default='MountainCar-v0')
parser.add_argument('--gamma', type=float, default=0.99)
# DQN policy setting
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 256, 256])
parser.add_argument('--train_eps', type=float, default=0.1)
parser.add_argument('--test_eps', type=float, default=0.05)
parser.add_argument('--target_update_freq', type=int, default=50)
# DQN training setting
parser.add_argument('--load_policy', action='store_true')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--step_per_epoch', type=int, default=10000)
parser.add_argument('--step_per_collect', type=int, default=10)
parser.add_argument('--update_per_step', type=float, default=0.1)
parser.add_argument('--buffer_size', type=int, default=10000)
parser.add_argument('--test_num', type=int, default=100)
# offline dataset setting
parser.add_argument('--generate_data', action='store_true')
parser.add_argument('--data_episode', type=int, default=2000)
parser.add_argument('--data_buffer_size', type=int, default=1e6)
# others
parser.add_argument('--save_dir', type=str, default='./dqn')
parser.add_argument('--eval_episodes', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env info
    env = gym.make(args.env_name)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    # trian/test env
    train_envs = gym.make(args.env_name)
    test_envs = gym.make(args.env_name)
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    train_envs.reset(seed=args.seed)
    test_envs.reset(seed=args.seed)
    # DQN model
    net = Net(state_shape, action_shape, args.hidden_dims, device=device).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = ts.policy.DQNPolicy(net, optimizer, discount_factor=args.gamma, 
                                 target_update_freq=args.target_update_freq)
    # buffer & collector
    buffer = ts.data.VectorReplayBuffer(args.buffer_size, buffer_num=1)
    train_collector = ts.data.Collector(policy, train_envs, buffer)
    test_collector = ts.data.Collector(policy, test_envs)
    # get policy path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    policy_path = os.path.join(args.save_dir, 'policy.ckpt')
    if not args.load_policy:
        # train behavior policy
        print('Training DQN policy')
        result = ts.trainer.offpolicy_trainer(
            policy, train_collector, test_collector,
            max_epoch=args.epoch, step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            update_per_step=args.update_per_step,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            train_fn=lambda epoch, env_step: policy.set_eps(args.train_eps),
            test_fn=lambda epoch, env_step: policy.set_eps(args.test_eps),
            stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
            save_best_fn=lambda policy: torch.save(policy.state_dict(), policy_path))
    else:
        # load behavior policy
        print('Loading DQN policy')
    # evaluate
    policy.load_state_dict(torch.load(policy_path))
    print('Evaluating DQN policy')
    policy.eval()
    policy.set_eps(args.test_eps)
    eval_collector = ts.data.Collector(policy, env)
    result = eval_collector.collect(n_episode=args.eval_episodes)
    rews = result['rews']
    print(f"Average reward: {rews.mean()}, max reward: {rews.max()}")
    # generate offline dataset
    if args.generate_data:
        # collect data 
        print('Collecting data using trained policy')
        data_buffer = ts.data.VectorReplayBuffer(total_size=args.data_buffer_size, 
                                                    buffer_num=1)
        data_collector = ts.data.Collector(policy, env, data_buffer)
        data_collector.collect(n_episode=args.data_episode)
        # save data
        dataset_path = os.path.join(args.save_dir, 'offline_dataset')
        data_buffer.save_hdf5(dataset_path)
        print('Offline dataset saved into:', dataset_path)
        