import numpy as np

from RandomPath import RandomPath

class Car:
    def __init__(self, map_shape, path_length, seed = 0):
        self.map_shape = map_shape
        self.x = np.random.randint(map_shape[1])
        self.y = np.random.randint(map_shape[0])
        self.dx = [0, 0, -1, 1]
        self.dy = [-1, 1, 0, 0]
        self.path = RandomPath(path_length, seed)
        self.path_left = path_length

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

    def get_position(self):
        return self.y, self.x
    
    def next_path(self):
        return self.path.get_action(self.path_left)