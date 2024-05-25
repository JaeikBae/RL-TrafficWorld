# %%
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

# Initialize TrafficWorld environment
env = TrafficWorld('data/map.csv')
initial_state, _ = env.reset()
flattened_state = env.flatten_state(initial_state)
n_observations = len(flattened_state)
n_actions = len(ACTIONS)

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, n_observations, n_actions):
        print(n_observations, n_actions)
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            # 3499 4
            # 차량 주변만...?
            nn.Linear(n_observations, 3499),
            nn.ReLU(),
            nn.Linear(3499, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.layers(x)

# %%
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
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

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# %%
load_epoch = 111000
try:
    policy_net.load_state_dict(torch.load(f'traffic_world_{load_epoch}.pth'))
    target_net.load_state_dict(policy_net.state_dict())
    print("Model loaded")
except:
    print("Model not found")
# %%
num_episodes = 1 if torch.cuda.is_available() else 1

for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(env.flatten_state(state), dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        next_state, reward, done, info = env.step(action.item())
        """
        info = {
            'episode_end': False,
            'episode_end_reason': None,
            'current_light': self.traffic_world_map.curr_light,
            'time': self.traffic_world_map.t
        }
        """
        reward = torch.tensor([reward], device=device)
        if not done:
            next_state = torch.tensor(env.flatten_state(next_state), dtype=torch.float32, device=device).unsqueeze(0)
        else:
            print(f"Episode {i_episode + 1} finished. Reward : {reward} - {info['episode_end_reason']}")
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
            # plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

torch.save(policy_net.state_dict(), 'traffic_world.pth')
# %%
# load model and visualize
load = 111000
policy_net.load_state_dict(torch.load(f'traffic_world{load}.pth'))
initial_state, _ = env.reset()  # 초기 상태를 가져옴
state = torch.tensor(env.flatten_state(initial_state), dtype=torch.float32, device=device).unsqueeze(0)

for t in count():
    action = policy_net(state).max(1)[1].view(1, 1)
    next_state, reward, done, info = env.step(action.item())
    state = torch.tensor(env.flatten_state(next_state), dtype=torch.float32, device=device).unsqueeze(0)
    env.render(interval=0.5, action=action.item())
    if done:
        print(f"Episode finished. Reward : {reward} - {info['episode_end_reason']}")
        break

env.close()


# %%
