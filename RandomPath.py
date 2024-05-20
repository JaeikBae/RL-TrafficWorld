import numpy as np

class RandomPath:
    def __init__(self, length, seed = None):
        self.path = []
        self.curr = 0
        if seed is not None:
            np.random.seed(seed)
        self.generate_path(length)

    def generate_path(self, length):
        x = 1
        y = 1
        dx = [0, 1, 0, -1]
        dy = [-1, 0, 1, 0]
        heading = 0 # 0: up, 1: right, 2: down, 3: left
        while True:
            if len(self.path) == length:
                break
            candidate = np.random.choice([0, 1, 2]) # 0: left, 1: straight, 2: right
            if candidate == 0:
                new_heading = (heading + 3) % 4
            elif candidate == 1:
                new_heading = heading
            else:
                new_heading = (heading + 1) % 4

            cx = x + dx[new_heading]
            cy = y + dy[new_heading]

            # check if the new position is valid
            if cx < 0 or cx >= 3 or cy < 0 or cy >= 3:
                continue
            
            x = cx
            y = cy
            heading = new_heading

            self.path.append(candidate)


    def progress(self):
        self.curr += 1
    
    def get_next_action(self):
        if self.curr == len(self.path):
            return None
        return self.path[self.curr]
    
    def get_prev_action(self):
        if self.curr == 0:
            return None
        return self.path[self.curr - 1]
    
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