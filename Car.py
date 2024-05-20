import numpy as np

from RandomPath import RandomPath

class Car:
    def __init__(self, map_shape, path_length, seed=None):
        self.map_shape = map_shape
        self.path_length = path_length
        self.seed = seed
        self.x = 32
        self.y = 45
        self.dx = [0, 1, 0, -1]
        self.dy = [-1, 0, 1, 0]
        self.heading = 0 # 0: up, 1: right, 2: down, 3: left
        self.path = RandomPath(path_length, seed)

    def get_state(self):
        return {
            'position': self.get_position(),
            'heading': self.get_heading(),
            'path': self.path.path,  # 현재 경로 전체
            'current_path_index': self.path.curr,  # 현재 경로 인덱스
        }

    def move(self, action):
        # translate action to dx, dy
        if action == 'a' or action == 'left_turn':
            action = 0
        elif action == 'd' or action == 'right_turn':
            action = 1
        elif action == 's' or action == 'stop':
            action = 2

        if action == 0:
            self.heading = (self.heading + 3) % 4
            dx = self.dx[self.heading]
            dy = self.dy[self.heading]
        elif action == 1:
            self.heading = (self.heading + 1) % 4
            dx = self.dx[self.heading]
            dy = self.dy[self.heading]
        elif action == 2:
            dx = 0
            dy = 0
        else:
            dx = self.dx[self.heading]
            dy = self.dy[self.heading]

        self.x += dx
        self.y += dy

        if self.x < 0:
            self.x = 0
        elif self.x >= self.map_shape[1]:
            self.x = self.map_shape[1] - 1
        if self.y < 0:
            self.y = 0
        elif self.y >= self.map_shape[0]:
            self.y = self.map_shape[0] - 1

    def move_one_more(self):
        self.move('w') # move forward one more step. heading does not change.

    def get_position(self):
        return self.y, self.x
    
    def get_heading(self):
        return self.heading
    
    def path_progress(self): 
        return self.path.progress()
    
    def next_path(self):
        return self.path.get_next_action()
    
    def prev_path(self):
        return self.path.get_prev_action()
    
    def reset(self):
        self.__init__(self.map_shape, self.path_length, self.seed)
        