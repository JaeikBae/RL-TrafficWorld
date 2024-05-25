# %%
%cd /ws
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Replace gym environment with TrafficWorld
from TrafficWorld import TrafficWorld, ACTIONS

import numpy as np

# Initialize TrafficWorld environment
env = TrafficWorld('data/map.csv')
initial_state, _ = env.reset()
flattened_state = env.flatten_state(initial_state)
n_observations = len(flattened_state)
n_actions = len(ACTIONS)
print(n_observations, n_actions)

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %%

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, map_shape, n_other_features, n_actions):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        conv_output_size = self._get_conv_output(map_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size + n_other_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_output(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, map_data, other_data):
        map_data = map_data.unsqueeze(1)  # Add channel dimension
        conv_out = self.conv_layers(map_data)
        conv_out = conv_out.view(conv_out.size(0), -1)
        combined = torch.cat((conv_out, other_data), dim=1)
        return self.fc_layers(combined)


# %%
BATCH_SIZE = 1024
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

map_shape = (1, env.map_data.shape[0], env.map_data.shape[1])  # assuming single channel input for CNN
n_other_features = n_observations - np.prod(env.map_data.shape)  # total observations minus the map data

policy_net = DQN(map_shape, n_other_features, n_actions).to(device)
target_net = DQN(map_shape, n_other_features, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
# %%

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            map_data = state[:, :np.prod(env.map_data.shape)].view(-1, *env.map_data.shape).to(device)
            other_data = state[:, np.prod(env.map_data.shape):].to(device)
            return policy_net(map_data, other_data).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

def plot_durations(name, reward=None, show_result=False):
    plt.figure(1)
    plt.clf()
    
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # print reward on another graph
    if rewards is not None:
        # make reawds to -1500 which is the minimum value
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        
        plt.figure(2)
        plt.clf()
        plt.title('Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        if len(rewards_t) >= 100:
            reward_means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            reward_means = torch.cat((torch.zeros(99), reward_means))
            plt.plot(reward_means.numpy())
        
        plt.savefig(f"./plots/{name}_reward.png")
    
    plt.figure(1)
    plt.savefig(f"./plots/{name}.png")
    
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    else:
        plt.show()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    non_final_next_map_data = non_final_next_states[:, :np.prod(env.map_data.shape)].view(-1, *env.map_data.shape).to(device)
    non_final_next_other_data = non_final_next_states[:, np.prod(env.map_data.shape):].to(device)
    state_map_data = state_batch[:, :np.prod(env.map_data.shape)].view(-1, *env.map_data.shape).to(device)
    state_other_data = state_batch[:, np.prod(env.map_data.shape):].to(device)

    state_action_values = policy_net(state_map_data, state_other_data).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_map_data, non_final_next_other_data).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# %%
load_epoch = 0
try:
    policy_net.load_state_dict(torch.load(f'./models/traffic_world_{load_epoch}.pth', map_location=device))
    target_net.load_state_dict(policy_net.state_dict())
    print("Model loaded")
except Exception as e:
    print(e)
    print("Model not found")
# %%
# print("device : ", device)
# num_episodes = 1000000 if torch.cuda.is_available() else 1000000

# rewards = []
# for i_episode in range(load_epoch, num_episodes):
#     state, _ = env.reset()
#     state = torch.tensor(env.flatten_state(state), dtype=torch.float32, device=device).unsqueeze(0)
#     sum_reward = 0
#     for t in count():
#         action = select_action(state)
#         next_state, reward, done, info = env.step(action.item())
#         """
#         info = {
#             'episode_end': False,
#             'episode_end_reason': None,
#             'current_light': self.traffic_world_map.curr_light,
#             'time': self.traffic_world_map.t
#         }
#         """
#         reward = torch.tensor([reward], device=device)
#         sum_reward += reward.item()

#         if not done:
#             next_state = torch.tensor(env.flatten_state(next_state), dtype=torch.float32, device=device).unsqueeze(0)
#         else:
#             rewards.append(sum_reward/(t+1))
#             print(f"Episode {i_episode + 1} finished. Reward : {reward} - {info['episode_end_reason']}")
#             next_state = None

#         memory.push(state, action, next_state, reward)
#         state = next_state
#         optimize_model()

#         target_net_state_dict = target_net.state_dict()
#         policy_net_state_dict = policy_net.state_dict()
#         for key in policy_net_state_dict:
#             target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
#         target_net.load_state_dict(target_net_state_dict)

#         if done:
#             episode_durations.append(t + 1)
#             break

#     if i_episode % 500 == 0:
#         torch.save(policy_net.state_dict(), f'./models/traffic_world_{i_episode}.pth')
#         plot_durations(i_episode, reward=rewards)
#         plt.ioff()
#         plt.show()


# print('Complete')
# # plot_durations("end", show_result=True)
# plt.ioff()
# plt.show()
# %%
# Load the model and visualize
load_epoch = 2000
map_shape = (1, env.map_data.shape[0], env.map_data.shape[1])  # assuming single channel input for CNN
n_other_features = n_observations - np.prod(env.map_data.shape)

target_net = DQN(map_shape, n_other_features, n_actions).to(device)
target_net.load_state_dict(torch.load(f'./models/traffic_world_{load_epoch}.pth', map_location=device))

initial_state, _ = env.reset()  # 초기 상태를 가져옴
flattened_state = env.flatten_state(initial_state)
map_data = torch.tensor(flattened_state[:np.prod(env.map_data.shape)], dtype=torch.float32, device=device).view(-1, *env.map_data.shape)
other_data = torch.tensor(flattened_state[np.prod(env.map_data.shape):], dtype=torch.float32, device=device).unsqueeze(0)

for t in count():
    action = target_net(map_data, other_data).max(1)[1].view(1, 1)
    next_state, reward, done, info = env.step(action.item())
    flattened_next_state = env.flatten_state(next_state)
    map_data = torch.tensor(flattened_next_state[:np.prod(env.map_data.shape)], dtype=torch.float32, device=device).view(-1, *env.map_data.shape)
    other_data = torch.tensor(flattened_next_state[np.prod(env.map_data.shape):], dtype=torch.float32, device=device).unsqueeze(0)
    
    print(f"Action : {action.item()} - Reward : {reward} - {info['episode_end_reason']}")
    if done:
        print(f"Episode finished. Reward : {reward} - {info['episode_end_reason']}")
        env.close()
        break
    else:
        env.render(action=action.item())




# %%
