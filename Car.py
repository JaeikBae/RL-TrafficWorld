import numpy as np

from RandomPath import RandomPath

# TODO : set start position

class Car:
    def __init__(self, map_shape, path_length, seed = 0):
        self.map_shape = map_shape
        self.x = 0
        self.y = 0
        self.dx = [0, 0, -1, 1]
        self.dy = [-1, 1, 0, 0]
        self.path = RandomPath(path_length, seed)
        self.path_left = path_length

    def move(self, action):
        if action == 'w':
            action = 0
        elif action == 's':
            action = 1
        elif action == 'a':
            action = 2
        elif action == 'd':
            action = 3
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

    def get_position(self):
        return self.y, self.x
    
    def next_path(self):
        return self.path.get_action(self.path_left)