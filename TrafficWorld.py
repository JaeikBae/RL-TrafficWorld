# GridWorld traffic simulation RL project
# agent(car) moves in the grid world
# car must obey the traffic light and lane rules.
# if stop line is 4, 5, 6, it is left, straight, right only lane respectively.
# the cars tragectory is given by random navigation instructions.

# in every time step, the agent can move to one of the four directions (up, down, left, right)
# rewards
# reach the destination : 100
# every time step : -1
# collision with wall : -100
# doen't follow the navigation instructions : -100
# doesn't follow the lane rules : -100
# doesn't follow the traffic light : -50

COLLISION_REWARD = -100
WRONG_DIRECTION_REWARD = -100
WRONG_LANE_REWARD = -100
WRONG_LIGHT_REWARD = -50
TIME_STEP_REWARD = -1
DEST_REWARD = 100

CAR_COLOR = 'orange'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from time import sleep

class TrafficWorldMap:
    def __init__(self, map_data):
        self.colors = [
            'gray', # 벽
            'black', # 차도
            'white', # 차선
            'yellow', # 중앙선
            'cyan', # 정지선 좌측
            'blue', # 정지선 직진
            'purple', # 정지선 우측
            'red', # 적색신호
            'green' # 녹색신호
        ]
        self.fig, self.ax = plt.subplots()
        self.t = 0
        self.curr = 7
        self.map_data = map_data

    def change_light(self):
        for i in range(self.map_data.shape[0]):
            for j in range(self.map_data.shape[1]):
                if self.map_data[i][j] == 7:
                    self.map_data[i][j] = 8
                elif self.map_data[i][j] == 8:
                    self.map_data[i][j] = 7

    def visualize_map(self):
        self.ax.clear()
        for i in range(self.map_data.shape[0]):
            for j in range(self.map_data.shape[1]):
                color = self.colors[self.map_data[i][j]]
                rect = patches.Rectangle((j, self.map_data.shape[0] - i - 1), 1, 1, linewidth=0.1, edgecolor='black', facecolor=color)
                self.ax.add_patch(rect)
        plt.xlim(0, self.map_data.shape[1])
        plt.ylim(0, self.map_data.shape[0])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.text(0, 0, self.t, fontsize=12, color='black')

    def update(self, frame):
        self.t = (self.t + 1) % 10
        if self.t < 7 and self.curr == 8:
            self.change_light()
            self.curr = 7
        elif self.t >= 7 and self.curr == 7:
            self.change_light()
            self.curr = 8
        self.visualize_map()

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=100, interval=1000)
        plt.show()

class TrafficWorld:
    def __init__(self, map_data):
        self.map_data = map_data
        self.map_shape = map_data.shape
        self.car = Car(self.map_shape)
        self.traffic_world_map = TrafficWorldMap(self.map_data)

    def step(self, action):
        self.car.move(action)
        reward = TIME_STEP_REWARD
        if self.car.is_collision(self.map_data):
            reward += COLLISION_REWARD
        if self.car.is_wrong_direction(self.map_data):
            reward += WRONG_DIRECTION_REWARD
        if self.car.is_wrong_lane(self.map_data):
            reward += WRONG_LANE_REWARD
        if self.car.is_wrong_light(self.map_data):
            reward += WRONG_LIGHT_REWARD
        if self.car.is_destination():
            reward += DEST_REWARD
        return reward

    def run(self):
        self.traffic_world_map.run()

class Car:
    def __init__(self, map_shape):
        self.map_shape = map_shape
        self.x = np.random.randint(map_shape[1])
        self.y = np.random.randint(map_shape[0])
        self.dx = [0, 0, -1, 1]
        self.dy = [-1, 1, 0, 0]

    def move(self, action):
        self.x += self.dx[action]
        self.y += self.dy[action]
        if self.x < 0:
            self.x = 0
        if self.x >= self.map_shape[1]:
            self.x = self.map_shape[1] - 1
        if self.y < 0:
            self.y = 0
        if self.y >= self.map_shape[0]:
            self.y = self.map_shape[0] - 1