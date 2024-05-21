import numpy as np
import threading
from Car import Car
from TrafficWorldMap import TrafficWorldMap

ACTIONS = ['a', 's', 'd', 'w']
COLLISION_REWARD = -100
WRONG_DIRECTION_REWARD = -100
WRONG_PATH_REWARD = -100
WRONG_LANE_REWARD = -100
WRONG_LIGHT_REWARD = -50
TIME_STEP_REWARD = -1
ON_CENTER_LINE_REWARD = -100
FAIL_REWARD = -500
SUCCESS_INTERSECTION_REWARD = 100
DEST_REWARD = 1000

CAR_COLOR = 'orange'
class TrafficWorld:
    def __init__(self, map_path, route_length=10):
        self.map_data = np.genfromtxt(map_path, delimiter=',', dtype=int)
        self.map_shape = self.map_data.shape
        self.route_length = route_length
        self.traffic_world_map = TrafficWorldMap(self.map_data)
        self.car = Car(self.map_shape, self.route_length)
        self.isCarOnIntersection = False
        self.reward_sum = 0
        self.done = False
        self.info = {
            'episode_end': False,
            'episode_end_reason': None,
            'current_light': self.traffic_world_map.curr_light,
            'time': self.traffic_world_map.t
        }

        self.last_heading = None
        self.intersection_direction = None

    def reset(self):
        state, info = self.get_state()
        self.car.reset()
        self.traffic_world_map.reset()
        self.reward_sum = 0
        self.done = False
        self.info = {
            'episode_end': False,
            'episode_end_reason': None,
            'current_light': self.traffic_world_map.curr_light,
            'time': self.traffic_world_map.t
        }
        return state, info

    def step(self, action):
        reward, reason = self.car_move(action)
        state, info = self.get_state()
        self.done = info['episode_end']
        return state, reward, self.done, info

    def render(self):
        if not hasattr(self, 'visualization_thread') or not self.visualization_thread.is_alive():
            self.visualization_thread = threading.Thread(target=self.traffic_world_map.start_visualization)
            self.visualization_thread.start()
        self.traffic_world_map.set_cx_cy(*self.car.get_position())
        self.traffic_world_map.time_step()

    def get_state(self):
        state = {
            'map': self.traffic_world_map.get_state(),
            'car': self.car.get_state(),
            'reward_sum': self.reward_sum,
            'is_car_on_intersection': self.isCarOnIntersection,
        }
        self.info = {
            'episode_end': self.info['episode_end'],
            'episode_end_reason': self.info['episode_end_reason'],
            'current_light': self.traffic_world_map.curr_light,
            'time': self.traffic_world_map.t
        }
        return state, self.info
    
    def flatten_state(self, state):
        # Flatten the state dictionary to a single list
        flattened_state = []
        flattened_state.extend(state['car']['position'])
        flattened_state.append(state['car']['heading'])
        flattened_state.extend(state['car']['path'])
        flattened_state.append(state['car']['current_path_index'])
        flattened_state.append(state['reward_sum'])
        flattened_state.append(state['is_car_on_intersection'])
        flattened_state.extend(state['map']['map_data'].flatten())
        flattened_state.append(state['map']['current_light'])
        flattened_state.append(state['map']['time'])
        return np.array(flattened_state, dtype=np.float32)

    def car_move(self, action):
        to_str = ["Left", "Straight", "Right", "Stop"]
        path = self.car.next_path()
        if path is None:
            print('No path left')
            exit()
        self.traffic_world_map.set_text(f'Next path: {to_str[path if path is not None else -1]}')
        self.car.move(action)
        cy, cx = self.car.get_position()
        self.traffic_world_map.set_cx_cy(cx, cy)
        self.traffic_world_map.time_step()
        reward, reason = self.get_reward()
        self.reward_sum += reward
        return self.reward_sum, reason

    def episode_end(self, reason):
        if reason != 'success':
            self.reward_sum += FAIL_REWARD
        self.info = {
            'episode_end': True,
            'episode_end_reason': reason,
            'current_light': self.traffic_world_map.curr_light,
            'time': self.traffic_world_map.t
        }
        self.car.reset()
        self.traffic_world_map.reset()
        self.done = True

    def get_reward(self):
        reward = TIME_STEP_REWARD
        reason = None
        cy, cx = self.car.get_position()

        if self.map_data[cy][cx] == 0:
            reward += COLLISION_REWARD
            reason = 'wall'
            self.episode_end(reason)

        elif self.map_data[cy][cx] == 1:
            pass

        elif self.map_data[cy][cx] == 2:
            # TODO : Penalty for staying on lane(pixel)
            pass

        elif self.map_data[cy][cx] == 3:
            reward += ON_CENTER_LINE_REWARD
            reason = 'center_line'
            self.episode_end(reason)

        elif self.map_data[cy][cx] in [-6, -5, -4, 4, 5, 6]:
            # TODO : Penalty for Wrong-way driving
            if False:
                reward += WRONG_DIRECTION_REWARD
            pass

        # Entering Intersection on RED : Traffic light signal violation
        elif self.map_data[cy][cx] == 7:            
            if not self.isCarOnIntersection:
                self.car_heading_at_entering = self.car.get_heading()
                self.isCarOnIntersection = True
                reward += WRONG_LIGHT_REWARD

        # Entering intersection on GREEN
        elif self.map_data[cy][cx] == 8:
            # TODO : Positive Reward for right action
            if not self.isCarOnIntersection:
                self.car_heading_at_entering = self.car.get_heading()
                self.isCarOnIntersection = True

        if self.isCarOnIntersection and self.map_data[cy][cx] not in [7, 8]: 
            self.car.path_progress()
            prev_path = self.car.prev_path()
            prev_heading = self.car_heading_at_entering
            car_heading = self.car.get_heading()

            is_will_end = False
            reason = None

            # reverse run detection
            if car_heading == 0 or car_heading == 1:
                if 1 in [self.map_data[cy][cx+1], self.map_data[cy][cx], self.map_data[cy][cx-1]]:
                    reward += WRONG_DIRECTION_REWARD
                    is_will_end = True; reason = 'reverse_run'
            elif car_heading == 2 or car_heading == 3:
                if 1 in [self.map_data[cy-1][cx], self.map_data[cy][cx], self.map_data[cy+1][cx]]:
                    reward += WRONG_DIRECTION_REWARD
                    is_will_end = True; reason = 'reverse_run'

            # path following detection
            if prev_path == 0 and car_heading != (prev_heading + 3) % 4:
                reward += WRONG_PATH_REWARD
                is_will_end = True; reason = 'wrong_path'
            elif prev_path == 1 and car_heading != prev_heading:
                reward += WRONG_PATH_REWARD
                is_will_end = True; reason = 'wrong_path'
            elif prev_path == 2 and car_heading != (prev_heading + 1) % 4:
                reward += WRONG_PATH_REWARD
                is_will_end = True; reason = 'wrong_path'

            if is_will_end:
                reason = 'wrong_direction'
                self.episode_end(reason)
            
            self.isCarOnIntersection = False
            # car successfully passed the intersection
            reward += SUCCESS_INTERSECTION_REWARD

        return reward, reason

if __name__ == '__main__':
    traffic_world = TrafficWorld('data/map.csv')
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
            reward, reason = traffic_world.get_reward()
            traffic_world.reward_sum += reward
            print(f'reward: {reward} / total reward: {traffic_world.reward_sum}')
            paths = ['Left', 'Straight', 'Right']
            print(f'next path: {paths[traffic_world.car.next_path()]}')
            if traffic_world.done:
                traffic_world.reset()