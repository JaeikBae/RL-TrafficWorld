import numpy as np

from RandomPath import RandomPath

class Car:
    def __init__(self, map_shape, path_length, seed=None):
        self.map_shape = map_shape
        self.path_length = path_length
        self.seed = seed
        self.y = 29
        self.x = 25
        self.dy = [-1, 0, 1, 0]
        self.dx = [0, 1, 0, -1]
        self.heading = 0 # 0: up, 1: right, 2: down, 3: left
        self.path = RandomPath(path_length, seed)

    def get_state(self):
        return {
            'position': self.get_position(),
            'heading': self.get_heading(),
            'path': self.path.path,
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
            ddx = self.dx[self.heading]
            ddy = self.dy[self.heading]
        elif action == 1:
            self.heading = (self.heading + 1) % 4
            ddx = self.dx[self.heading]
            ddy = self.dy[self.heading]
        elif action == 2:
            ddx = 0
            ddy = 0
        else:
            ddx = self.dx[self.heading]
            ddy = self.dy[self.heading]

        self.x += ddx
        self.y += ddy

        if self.x < 0:
            self.x = 0
        if self.x >= self.map_shape[1]:
            self.x = self.map_shape[1] - 1
        if self.y < 0:
            self.y = 0
        if self.y >= self.map_shape[0]:
            self.y = self.map_shape[0] - 1
        
    def move_one_forward(self):
        self.move('w')

    def move_one_right(self):
        if self.heading == 0:
            self.x += 1
        elif self.heading == 1:
            self.y += 1
        elif self.heading == 2:
            self.x -= 1
        elif self.heading == 3:
            self.y -= 1

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
        self.__init__(self.map_shape, self.path_length, seed=self.seed)
        