# %%
%cd /ws
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

from TrafficWorld import TrafficWorld, ACTIONS

# Initialize TrafficWorld environment
env = TrafficWorld('data/map.csv', route_length=2)
initial_state, _ = env.reset()
flattened_state = env.flatten_state(initial_state)
n_observations = len(flattened_state)
n_actions = len(ACTIONS)
print(n_observations, n_actions)

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.fc_layers(x)

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.3
EPS_END = 0.1
EPS_DECAY = 10000
TAU = 0.1
LR = 1e-5

input_dim = n_observations

policy_net = DQN(input_dim, n_actions).to(device)
target_net = DQN(input_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
policy_net.train()

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(20000)

scaler = GradScaler()

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []
policy_rewards = []

def plot_durations(name):
    # clear interactive output
    plt.figure(1)
    plt.cla()
    plt.title('Policy')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(policy_rewards)
    plt.savefig(f'./plots/{name}.png')
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

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    scaler.step(optimizer)
    scaler.update()

def eval_target(is_test=False):
    state, _ = env.reset()
    state = torch.tensor(env.flatten_state(state), dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    target_net.eval()
    for t in count():
        action = target_net(state).max(1)[1].view(1, 1)
        next_state, reward, done, info = env.step(action.item())
        total_reward += reward
        if is_test:
            print("Action: ", ACTIONS[action.item()], end=" ")
            print("Reward: ", reward, end=" ")
            print("Total Reward: ", total_reward)
        if not done:
            next_state = torch.tensor(env.flatten_state(next_state), dtype=torch.float32, device=device).unsqueeze(0)
        else:
            print(" " * 100, end="\r")
            print(f"Eval finished. Reward: {total_reward} / Reason: {info['episode_end_reason']}", end="\r")
            break
        state = next_state
    target_net.train()
    return total_reward

# %%
import pickle
load_at = 0
# load pkl
# policy_rewards = pickle.load(open("policy_rewards.pkl", "rb"))[:load_at]
# target_rewards = pickle.load(open("target_rewards.pkl", "rb"))[:load_at]
# episode_durations = pickle.load(open("episode_durations.pkl", "rb"))[:load_at]
# policy_net.load_state_dict(torch.load(f'./models/traffic_world_{load_at}.pth'))
# target_net.load_state_dict(policy_net.state_dict())
dones = []
print("device : ", device)
num_episodes = 1000000 if torch.cuda.is_available() else 1000000
for i_episode in range(load_at, num_episodes):
    state, _ = env.reset()
    state = torch.tensor(env.flatten_state(state), dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0
    for t in count():
        action = select_action(state)
        next_state, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        episode_reward += reward.item()

        if not done:
            next_state = torch.tensor(env.flatten_state(next_state), dtype=torch.float32, device=device).unsqueeze(0)
        else:
            policy_rewards.append(episode_reward)
            if info['episode_end_reason'] == 'DONE':
                dones.append(i_episode)
            print(" " * 100, end="\r")
            print(f"Episode {i_episode + 1}/{episode_reward}/{info['episode_end_reason']}/{env.end_at}/{env.done_end_at}/{len(dones)}", end="\r")
            next_state = None
            env.reset()

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        for target_param, local_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

        if done:
            episode_durations.append(t + 1)
            break
    
    if i_episode % 1000 == 0:
        torch.save(policy_net.state_dict(), f'./models/traffic_world_{i_episode}.pth')
        plot_durations(f'traffic_world_{i_episode}')
        with open("policy_rewards.pkl", "wb") as f:
            pickle.dump(policy_rewards, f)
        with open("episode_durations.pkl", "wb") as f:
            pickle.dump(episode_durations, f)

print('Complete')
plt.ioff()
plt.show()


# %%
# test

env = TrafficWorld('data/map.csv', route_length=2, seed=999)

load_at = 97000
print(f"Load at {load_at}")
net = DQN(input_dim, n_actions).to(device)
net.load_state_dict(torch.load(f'./models/traffic_world_{load_at}.pth'))

history = []
def eval(is_test=False):
    state, _ = env.reset()
    print(env.car.path)
    state_t = torch.tensor(env.flatten_state(state), dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    history.append(env.car.get_position())
    for t in count():
        action = net(state_t).max(1)[1].view(1, 1)
        next_state, reward, done, info = env.step(action.item())
        history.append(env.car.get_position())
        total_reward += reward
        if is_test:
            print("Action: ", ACTIONS[action.item()], end=" ")
            print("Reward: ", reward, end=" ")
            print("Total Reward: ", total_reward)
        if not done:
            next_state = torch.tensor(env.flatten_state(next_state), dtype=torch.float32, device=device).unsqueeze(0)
        else:
            print(" " * 100, end="\r")
            print(f"Eval finished. Reward: {total_reward} / Reason: {info['episode_end_reason']}", end="\r")
            break
        state_t = next_state
    return state['map']['map_data']

import numpy as np

def visualize_into_plot(map, hist):
    # plot
    fig, ax = plt.subplots()
    ax.imshow(map, cmap='gray')
    for i, h in enumerate(hist):
        ax.plot(h[1], h[0], 'ro', markersize=3)
        ax.text(h[1]+1, h[0], str(i), color='red', fontsize=6)
        
    plt.show()

map = eval(is_test=True)
visualize_into_plot(map=map, hist=history)
# %%
