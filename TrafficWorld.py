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
# on the center line : -100

COLLISION_REWARD = -100
WRONG_DIRECTION_REWARD = -100
WRONG_PATH_REWARD = -100
WRONG_LANE_REWARD = -100
WRONG_LIGHT_REWARD = -50
TIME_STEP_REWARD = -1
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
        self.car = Car(self.map_shape, 10)
        self.traffic_world_map = TrafficWorldMap(self.map_data)

    def car_move(self, action):
        print(action)
        self.car.move(action)
        cy, cx = self.car.get_position()
        self.traffic_world_map.set_cx_cy(cx, cy)

    def get_reward(self):
        # 0: wall
        # 1: stopline
        # 2: line
        # 3: center line
        # +-4: 1st lane, +-5: 2nd lane, +-6: 3rd lane
        # 7: red light, 8: green light
        reward = TIME_STEP_REWARD

        cy, cx = self.car.get_position()

        if self.map_data[cy][cx] == 0: # at the wall
            """
            TODO
                end the simulation and reset the car's position
                increase COLLISION_REWARD to -1000?
            """
            reward += COLLISION_REWARD

        elif self.map_data[cy][cx] == 1: # at the stop line
            """
            TODO
                update the car's path
                if the car reached the destination, give DEST_REWARD
                somthing about traffic light
            """
            pass

        elif self.map_data[cy][cx] == 2: # at the line
            """
            NOTE
                when car moves, it aumaticaly moves to the target lane.
                so the car can't be on the line.
            """
            pass

        elif self.map_data[cy][cx] == 3: # at the center line
            """
            TODO
                end the simulation and reset the car's position
                increase ON_CENTER_LINE_REWARD to -1000?
                may integrate with the wall collision?
            """
            reward += ON_CENTER_LINE_REWARD

        elif self.map_data[cy][cx] in [-6, -5, -4, 4, 5, 6]: # at the each lane
            """
            TODO
                if the car doing reverse run, give WRONG_DIRECTION_REWARD
                save the car's lane to determine the car following the lane rules at the stop line
                before reaches the stop line, the car can move to the other lane (if it isn't reverse run)
            """
            reward += WRONG_DIRECTION_REWARD

        elif self.map_data[cy][cx] in 7: # at the redtraffic light
            """
            TODO
                give WRONG_LIGHT_REWARD at entering time step only
            """
            reward += WRONG_LIGHT_REWARD

        elif self.map_data[cy][cx] in 8: # at the green traffic light
            """
            NOTE
                on green light, no penalty
            """
            pass

        if self.car.next_path() == 0: # doesn't follow the navigation instructions
            """
            TODO
                configure "if" statement
                if the car doesn't follow the navigation instructions, 
                give WRONG_PATH_REWARD or terminate the simulation?
            """
            pass

        return reward

    def run(self):
        self.traffic_world_map.start_visualization()

if __name__ == '__main__':
    traffic_world = TrafficWorld('data/map.csv')
    import threading
    t = threading.Thread(target=traffic_world.run)
    t.start()
    import keyboard
    from time import sleep
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            if keyboard.is_pressed('w'):
                traffic_world.car_move('w')
            if keyboard.is_pressed('s'):
                traffic_world.car_move('s')
            if keyboard.is_pressed('a'):
                traffic_world.car_move('a')
            if keyboard.is_pressed('d'):
                traffic_world.car_move('d')
            print(traffic_world.get_reward())