import gym
import torch
import argparse
import random
import numpy as np

from PIL import Image
from os.path import join
from model import Model
from utils import create_dir_if_not_exists, preprocess
from collections import deque

def create_parser():
    # Creates a parser for command-line arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--replay_memory_capacity', type=int, default=25000)
    parser.add_argument('--min_replay_memory', type=int, default=1000)
    parser.add_argument('--num_episodes', type=int, default=150000)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.1)

    return parser

if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()

    # Device device type and number of gpus
    opts.device = torch.device('cuda' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')
    opts.n_gpus = torch.cuda.device_count() if str(opts.device) == 'cuda' else 0
    print('Device Type: {} | Number of GPUs: {}'.format(opts.device, opts.n_gpus))

    # Initializations
    save_path = join(opts.output_dir, 'Pong')
    create_dir_if_not_exists(save_path)
    env = gym.make('PongDeterministic-v0')
    policy_net = Model(env.action_space.n).to(opts.device)
    target_net = Model(env.action_space.n).to(opts.device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters() ,lr=opts.learning_rate)

    # Start Training
    replay_memory = deque()
    episode_rewards = []
    for episode in range(opts.num_episodes):
        print('Episode = {}'.format(episode))
        terminated = False
        episode_reward = 0
        initial_obs = preprocess(env.reset())
        observations = deque(4 * [initial_obs])

        while not terminated:
            # Pick the next action
            if opts.epsilon >= random.random():
                action = env.action_space.sample()
            else:
                policy_net.eval()
                x = torch.FloatTensor(observations).unsqueeze(0).to(opts.device)
                action = policy_net.pick_action(x)
                policy_net.train()
            new_obs, reward, terminated, _ = env.step(action)
            episode_reward += reward
            new_obs = preprocess(new_obs)
            env.render() # Can be commented out

            # Update observations and replay_memory
            prev_state = torch.FloatTensor(observations)
            observations.popleft(); observations.append(new_obs)
            next_state = torch.FloatTensor(observations)
            replay_memory.append((prev_state, action, reward, next_state, terminated))
            if len(replay_memory) > opts.replay_memory_capacity:
                replay_memory.popleft()

            # Sample random minibatch of transitions from the replay_memory
            # and update the Q-network parameters
            if len(replay_memory) >= opts.min_replay_memory:
                optimizer.zero_grad()
                minibatch = random.sample(replay_memory, opts.batch_size)

                prev_states = torch.cat([x[0].unsqueeze(0) for x in minibatch], dim=0).to(opts.device)
                next_states = torch.cat([x[3].unsqueeze(0) for x in minibatch], dim=0).to(opts.device)

                prev_state_preds = policy_net(prev_states)
                next_states_preds = torch.max(target_net(next_states).detach(), dim=1)[0]
                next_states_preds = next_states_preds.cpu().data.numpy().tolist()

                preds, ys = [], []
                for i in range(opts.batch_size):
                    preds.append(prev_state_preds[i, minibatch[i][1]])
                    if minibatch[i][-1]: ys.append(minibatch[i][2])
                    else: ys.append(minibatch[i][2] + opts.gamma * next_states_preds[i])
                preds = torch.stack(preds)
                ys = torch.FloatTensor(ys).to(opts.device)
                loss = torch.mean((preds - ys) ** 2)
                loss.backward()
                optimizer.step()

        # Save the Q Network after each episode
        torch.save(policy_net, join(save_path, 'net'))

        # Update the target network, copying all weights and biases in DQN
        if (episode+1) % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Update episode_rewards
        episode_rewards.append(episode_reward)
        if len(episode_rewards) == 100:
            print('Average total reward last 100 episodes is {}'.format(np.average(episode_rewards)))
            episode_rewards = []
