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
# on the lane : -1
# on the center line : -100

COLLISION_REWARD = -100
WRONG_DIRECTION_REWARD = -100
WRONG_LANE_REWARD = -100
WRONG_LIGHT_REWARD = -50
TIME_STEP_REWARD = -1
ON_LANE_REWARD = -1
ON_CENTER_LINE_REWARD = -100
DEST_REWARD = 100

CAR_COLOR = 'orange'

from TrafficWorldMap import TrafficWorldMap
from Car import Car
import numpy as np

class TrafficWorld:
    def __init__(self, map_path):
        self.map_data = np.genfromtxt(map_path, delimiter=',', dtype=int)
        self.map_shape = self.map_data.shape
        self.car = Car(self.map_shape)
        self.traffic_world_map = TrafficWorldMap(self.map_data)

    def car_move(self, action):
        self.car.move(action)

    def get_reward(self):
        # 0: wall, 1: road, 2: lane, 3: center line, 4: stop line left, 5: stop line straight, 6: stop line right, 7: red light, 8: green light
        reward = TIME_STEP_REWARD
        cx, cy = self.car.get_position()

        if self.map_data[cy][cx] == 0: # at the wall
            reward += COLLISION_REWARD
        elif self.map_data[cy][cx] == 1: # at the road
            pass
        elif self.map_data[cy][cx] == 2: # at the lane
            reward += ON_LANE_REWARD
        elif self.map_data[cy][cx] == 3: # at the center line
            reward += ON_CENTER_LINE_REWARD
        elif self.map_data[cy][cx] in [4, 5, 6]: # at the stop line
            if self.car.next_path() != self.map_data[cy][cx]: # doesn't follow the lane rules
                reward += WRONG_DIRECTION_REWARD
        elif self.map_data[cy][cx] == 7: # at the red light
            reward += WRONG_LIGHT_REWARD
        elif self.map_data[cy][cx] == 8: # at the green light
            pass

    def run(self):
        self.traffic_world_map.show(*self.car.get_position(), CAR_COLOR)

if __name__ == '__main__':
    traffic_world = TrafficWorld('data/map.csv')
    traffic_world.run()