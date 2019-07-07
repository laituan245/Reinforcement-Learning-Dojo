import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import LongTensor, FloatTensor
from gym.wrappers import Monitor

# Main Classes and Helper Functions
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, nb_actions):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU(True)
        self.linear2 = nn.Linear(hidden_size, nb_actions)

    def forward(self, x):
        return self.linear2(self.relu1(self.linear1(x)))

    def sample_action(self, obs):
        x = FloatTensor(obs)
        x = x.unsqueeze(0)
        probs = F.softmax(self.forward(x), dim=1)
        probs = list(probs.squeeze().data.numpy())
        probs[1] = 1.0 - probs[0] # Avoiding error by numerical issue
        return np.random.choice(2, 1, p=probs)[0]

class Episode:
    def __init__(self):
        self.episode_steps = []
        self.total_reward = 0

    def add_step(self, obs, action, reward):
        self.episode_steps.append((obs, action, reward))
        self.total_reward += reward

class AugmentedList:
    def __init__(self, items, shuffle_between_epoch=False):
        self.items = items
        self.cur_idx = 0
        self.shuffle_between_epoch = shuffle_between_epoch

    def next_items(self, batch_size):
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx : end_idx]
        else:
            first_part = items[start_idx : self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0 : remain_size]
            self.cur_idx = remain_size
            returned_batch = [item for item in first_part + second_part]
            if self.shuffle_between_epoch:
                random.shuffle(self.items)
            return returned_batch

    @property
    def size(self):
        return len(self.items)

def sample_episode(env, policy_network, should_render=False):
    episode = Episode()
    obs = env.reset()
    while True:
        if should_render:
            env.render()
        picked_action = policy_network.sample_action(obs)
        new_obs, reward, done, _ = env.step(picked_action)
        episode.add_step(obs, picked_action, reward)
        if done: break
        obs = new_obs
    return episode

# Initialize a new CartPole environment
env = gym.make('CartPole-v0')
env._max_episode_steps = 500

# Initialize a PolicyNetwork
input_size = env.observation_space.shape[0]
hidden_size = 100
nb_actions = env.action_space.n
policy_network = PolicyNetwork(input_size, hidden_size, nb_actions)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy_network.parameters(), lr=0.01)

# Training Loop
batch_size = 128
for itx in range(30):
    # Sample new episodes
    episodes = []
    for i in range(100):
        episodes.append(sample_episode(env, policy_network))
    average_total_reward = np.mean([e.total_reward for e in episodes])
    print('itx {}: average total reward = {}'.format(itx, average_total_reward))
    # Sort the episodes by total reward
    for i in range(len(episodes)):
        for j in range(i + 1, len(episodes)):
            if episodes[j].total_reward > episodes[i].total_reward:
                episodes[i], episodes[j] = episodes[j], episodes[i]
    # Use the best episodes for training
    episodes = episodes[:int(0.3 * len(episodes))]
    train_examples = []
    for episode in episodes:
        for (obs, action, reward) in episode.episode_steps:
            train_examples.append((obs, action))
    train = AugmentedList(train_examples)
    # Start training
    nb_iters = int(train.size / batch_size)
    for i in range(nb_iters):
        policy_network.zero_grad()
        batch = train.next_items(batch_size)
        xs = FloatTensor([example[0] for example in batch])
        ys = LongTensor([example[1] for example in batch])
        preds = policy_network(xs)
        loss = criterion(preds, ys)
        loss.backward()
        optimizer.step()

# Initialize a final environment
final_env = gym.make('CartPole-v0')
final_env._max_episode_steps = 500
final_env = Monitor(final_env, 'video')
sample_episode(final_env, policy_network, should_render=True)
