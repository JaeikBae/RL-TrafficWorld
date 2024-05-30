import numpy as np
import threading
from Car import Car
from TrafficWorldMap import TrafficWorldMap
from time import sleep

from copy import deepcopy

ACTIONS = ['left_turn', 'right_turn', 'stop', 'forward']
# TIME_OVER_REWARD = -3
# COLLISION_REWARD = -5
# ON_CENTER_LINE_REWARD = -3
# WRONG_DIRECTION_INTER_REWARD = -3
# WRONG_DIRECTION_ROAD_REWARD = -10
# WRONG_PATH_REWARD = -3
# FAIL_REWARD = -20

# WRONG_LANE_REWARD = -1
# WRONG_LIGHT_REWARD = -1
# TIME_STEP_REWARD = -1
# STOP_REWARD = -1
# GOOD_DIRECTION_REWARD = 5
# ENTER_INTERSECTION_REWARD = 10
# SUCCESS_INTERSECTION_REWARD = 20
# DEST_REWARD = 100

#2
TIME_OVER_REWARD = -1
COLLISION_REWARD = -1
ON_CENTER_LINE_REWARD = -1
WRONG_DIRECTION_INTER_REWARD = -1
WRONG_DIRECTION_ROAD_REWARD = -1
WRONG_PATH_REWARD = -1

WRONG_LANE_REWARD = -1
WRONG_LIGHT_REWARD = -1
TIME_STEP_REWARD = -1
FAIL_REWARD = -10
STOP_REWARD = -1
GOOD_DIRECTION_REWARD = 2
ENTER_INTERSECTION_REWARD = 5
SUCCESS_INTERSECTION_REWARD = 10
DEST_REWARD = 100

CAR_COLOR = 'orange'

class TrafficWorld:
    def __init__(self, map_path, route_length, seed=None):
        self.map_data = np.genfromtxt(map_path, delimiter=',', dtype=int)
        self.map_shape = self.map_data.shape
        self.route_length = route_length
        self.traffic_world_map = TrafficWorldMap(self.map_data)
        self.car = Car(self.map_shape, self.route_length, seed=seed)
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
        self.prev_pixel = None
        self.end_at = 0
        self.done_end_at = 0
        self.map_path = map_path

    def reset(self):
        self.__init__(self.map_path, self.route_length)
        self.car.reset()
        self.traffic_world_map.reset()
        state, info = self.get_state()
        return state, info

    def step(self, action):
        reward, reason = self.car_move(action)
        state, info = deepcopy(self.get_state())
        return state, reward, self.done, info

    def render(self, action=None):
        if not hasattr(self, 'visualization_thread') or not self.visualization_thread.is_alive():
            self.visualization_thread = threading.Thread(target=self.traffic_world_map.start_visualization)
            self.visualization_thread.start()
        position = self.car.get_position()
        self.traffic_world_map.set_cx_cy(position[1], position[0])
        to_str_path = ['Left', 'Straight', 'Right']
        to_str_heading = ['Up', 'Right', 'Down', 'Left']
        next_path = self.car.next_path()
        if action is not None and next_path is not None:
            self.traffic_world_map.set_text([
                f'Action: {ACTIONS[action]}',
                f'Next path: {to_str_path[next_path]}'
            ])
        else:
            self.traffic_world_map.set_text([
                f'Action: None',
                f'Next path: None'
            ])
        sleep(0.25)

    def close(self):
        self.traffic_world_map.close()

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
        flattened_state.append(state['map']['current_light'])
        flattened_state.append(state['map']['time'])
        flattened_state.extend(state['map']['map_data'].flatten())
        return np.array(flattened_state, dtype=np.float32)

    def car_move(self, action):
        to_str = ['left_turn', 'right_turn', 'stop', 'forward']
        path = self.car.next_path()
        if path is None:
            print()
            print('No path left')
            self.reward_sum += DEST_REWARD
            self.episode_end("DONE")
        self.traffic_world_map.set_text(f'Next path: {to_str[path if path is not None else -1]}')
        self.car.move(action)
        cy, cx = self.car.get_position()
        self.traffic_world_map.set_cx_cy(cx, cy)
        self.traffic_world_map.time_step()
        reward, reason = self.get_reward(action)
        self.reward_sum += reward
        return self.reward_sum, reason

    def episode_end(self, reason):
        self.end_at = self.traffic_world_map.t
        self.done_end_at = self.car.path.curr
        if reason != 'DONE':
            self.reward_sum += FAIL_REWARD
        self.info = {
            'episode_end': True,
            'episode_end_reason': reason,
            'current_light': self.traffic_world_map.curr_light,
            'time': self.traffic_world_map.t
        }
        self.done = True

    def get_reward(self, action):
        reward = TIME_STEP_REWARD 
        if self.traffic_world_map.t >= 100:
            reward += TIME_OVER_REWARD
            self.episode_end('time_over')
            return reward, 'time_over'
        reason = None
        cy, cx = self.car.get_position()

        if self.map_data[cy][cx] == 0:
            reward += COLLISION_REWARD
            reason = 'wall'
            self.episode_end(reason)
            return reward, reason

        elif self.map_data[cy][cx] == 1:
            if self.prev_pixel in [4, -4] and self.car.next_path() != 0:
                reward += WRONG_LANE_REWARD
            elif self.prev_pixel in [5, -5] and self.car.next_path() != 1:
                reward += WRONG_LANE_REWARD
            elif self.prev_pixel in [6, -6] and self.car.next_path() != 2:
                reward += WRONG_LANE_REWARD

        elif self.map_data[cy][cx] == 3:
            reward += ON_CENTER_LINE_REWARD
            reason = 'center_line'
            self.episode_end(reason)
            return reward, reason

        elif self.map_data[cy][cx] in [-6, -5, -4, 4, 5, 6] and not self.isCarOnIntersection:
            # if car is on the road

            # if action is stop
            if action == 'stop' or action == 2:
                reward += STOP_REWARD
            
            # check the direction of the road : check 4 pixels on each side and look for center line pixel value(3)
            for i in range(4, -5, -1):
                # if road is vertical
                if self.map_data[cy][np.clip(cx-i, 0, 46)] == 3:
                    # if road direction is right & car heading is left
                    if self.map_data[cy][cx] < 0:
                        if self.car.get_heading() == 0:
                            reward += WRONG_DIRECTION_ROAD_REWARD
                            reason = 'reverse_run:road'
                            self.episode_end(reason)
                            return reward, reason
                    # if road direction is left & car heading is right
                    elif self.map_data[cy][cx] > 0:
                        if self.car.get_heading() == 2:
                            reward += WRONG_DIRECTION_ROAD_REWARD
                            reason = 'reverse_run:road'
                            self.episode_end(reason)
                            return reward, reason

                    if action != 'stop' and action != 2:
                        reward += GOOD_DIRECTION_REWARD
                    break
                
                # if road is horizontal
                if self.map_data[np.clip(cy-i, 0, 46)][cx] == 3:
                    # if road direction is down & car heading is up
                    if self.map_data[cy][cx] < 0:
                        if self.car.get_heading() == 3:
                            reward += WRONG_DIRECTION_ROAD_REWARD
                            reason = 'reverse_run:road'
                            self.episode_end(reason)
                            return reward, reason
                    # if road direction is up & car heading is down
                    elif self.map_data[cy][cx] > 0:
                        if self.car.get_heading() == 1:
                            reward += WRONG_DIRECTION_ROAD_REWARD
                            reason = 'reverse_run:road'
                            self.episode_end(reason)
                            return reward, reason
                    
                    if action != 'stop' and action != 2:
                        reward += GOOD_DIRECTION_REWARD
                    break

        # Entering Intersection on RED : Traffic light signal violation
        elif self.map_data[cy][cx] == 7:
            if not self.isCarOnIntersection:
                if self.prev_pixel != 1:
                    reward += WRONG_DIRECTION_INTER_REWARD
                    reason = 'reverse_run:intersection'
                    self.episode_end(reason)
                    return reward, reason
                reward += ENTER_INTERSECTION_REWARD
                self.car_heading_at_entering = self.car.get_heading()
                self.isCarOnIntersection = True

        # Entering intersection on GREEN
        elif self.map_data[cy][cx] == 8:
            if not self.isCarOnIntersection:
                if self.prev_pixel != 1:
                    reward += WRONG_DIRECTION_INTER_REWARD
                    reason = 'reverse_run:intersection'
                    self.episode_end(reason)
                    return reward, reason
                reward += ENTER_INTERSECTION_REWARD
                self.car_heading_at_entering = self.car.get_heading()
                self.isCarOnIntersection = True

        if self.isCarOnIntersection and self.map_data[cy][cx] not in [7, 8]: 
            target_path = self.car.path.get_next_action()
            prev_heading = self.car_heading_at_entering
            car_heading = self.car.get_heading()

            reason = None

            # reverse run detection
            if car_heading == 1 or car_heading == 3:
                if 1 in [self.map_data[cy][np.clip(cx+1, 0, 46)], self.map_data[cy][cx], self.map_data[cy][np.clip(cx-1, 0, 46)]]:
                    reward += WRONG_DIRECTION_INTER_REWARD
                    reason = 'reverse_run:intersection'
                    self.episode_end(reason)
                    return reward, reason
            elif car_heading == 0 or car_heading == 2:
                if 1 in [self.map_data[np.clip(cy-1, 0, 46)][cx], self.map_data[cy][cx], self.map_data[np.clip(cy+1, 0, 46)][cx]]:
                    reward += WRONG_DIRECTION_INTER_REWARD
                    reason = 'reverse_run:intersection'
                    self.episode_end(reason)
                    return reward, reason

            # path following detection
            if target_path == 0 and car_heading != (prev_heading + 3) % 4:
                reward += WRONG_PATH_REWARD
                reason = 'wrong_path'
                self.episode_end(reason)
                return reward, reason
            elif target_path == 1 and car_heading != prev_heading:
                reward += WRONG_PATH_REWARD
                reason = 'wrong_path'
                self.episode_end(reason)
                return reward, reason
            elif target_path == 2 and car_heading != (prev_heading + 1) % 4:
                reward += WRONG_PATH_REWARD
                reason = 'wrong_path'
                self.episode_end(reason)
                return reward, reason
            
            self.isCarOnIntersection = False
            # car successfully passed the intersection
            reward += SUCCESS_INTERSECTION_REWARD * (self.car.path.curr+1)
            self.car.path_progress()

        self.prev_pixel = self.map_data[cy][cx]
        return reward, reason
