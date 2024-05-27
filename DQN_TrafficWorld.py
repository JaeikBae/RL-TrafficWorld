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
from torch.cuda.amp import GradScaler

from TrafficWorld import TrafficWorld, ACTIONS

# Initialize TrafficWorld environment
env = TrafficWorld('data/map.csv', route_length=3)
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
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
        )

    def forward(self, x):
        return self.fc_layers(x)

# %%
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

input_dim = n_observations

policy_net = DQN(input_dim, n_actions).to(device)
target_net = DQN(input_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
memory = ReplayMemory(10000)

scaler = GradScaler()

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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []
path_done = []
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
    durations_t = durations_t[durations_t < 1000]
    durations_t = durations_t[durations_t > 0]
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if reward is not None:
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        
        plt.figure(2)
        plt.clf()
        plt.title('Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        rewards_t = rewards_t[rewards_t < 1500]
        rewards_t = rewards_t[rewards_t > -1500]
        plt.plot(rewards_t.numpy())
        
        if len(rewards_t) >= 100:
            reward_means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            reward_means = torch.cat((torch.zeros(99), reward_means))
            plt.plot(reward_means.numpy())
        
        plt.savefig(f"./plots/{name}_reward.png")
    
    plt.figure(1)
    plt.savefig(f"./plots/{name}.png")

    plt.figure(3)
    plt.clf()
    plt.title('Path Done')
    plt.xlabel('Episode')
    plt.ylabel('Path Done')
    plt.plot(path_done)
    
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    else:
        plt.show()

# %%
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
    scheduler.step()

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
print("device : ", device)
num_episodes = 1000000 if torch.cuda.is_available() else 1000000

rewards = []
for i_episode in range(load_epoch, num_episodes):
    state, _ = env.reset()
    state = torch.tensor(env.flatten_state(state), dtype=torch.float32, device=device).unsqueeze(0)
    sum_reward = 0
    for t in count():
        action = select_action(state)
        next_state, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        sum_reward += reward.item()

        if not done:
            next_state = torch.tensor(env.flatten_state(next_state), dtype=torch.float32, device=device).unsqueeze(0)
        else:
            path_done.append(env.car.path.curr)
            rewards.append(sum_reward/(t+1))
            print(f"Episode {i_episode + 1}/{env.end_at} finished. Reward : {reward} - {info['episode_end_reason']}")
            next_state = None

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break

    if i_episode % 500 == 0:
        torch.save(policy_net.state_dict(), f'./models/traffic_world_{i_episode}.pth')
        plot_durations(i_episode, reward=rewards)
        plt.ioff()
        plt.show()
        import pickle
        with open("rewards.pkl", "wb") as f:
            pickle.dump(rewards, f)
        with open("episode_durations.pkl", "wb") as f:
            pickle.dump(episode_durations, f)

print('Complete')
plt.ioff()
plt.show()
# %%
