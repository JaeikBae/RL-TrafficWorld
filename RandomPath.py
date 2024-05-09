import numpy as np

class RandomPath:
    def __init__(self, length, seed = 0):
        self.path = []
        np.random.seed(seed)
        self.generate_path(length)

    def generate_path(self, length):
        # 0: Left, 1: Straight, 2: Right
        for _ in range(length):
            self.path.append(np.random.randint(3))

    def get_action(self, left):
        return self.path[len(self.path) - left]
    
    def __str__(self) -> str:
        to_str = [
            'Left',
            'Straight',
            'Right'
        ]
        # [id] - [path]
        return '\n'.join([f'{i} - {to_str[path]}' for i, path in enumerate(self.path)])
        

if __name__ == '__main__':
    path = RandomPath(10)
    print(path)