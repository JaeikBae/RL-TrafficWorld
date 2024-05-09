import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RED_LIGHT = 7
GREEN_LIGHT = 8

# TODO : how to show the car on the map

class TrafficWorldMap:
    def __init__(self, map_data):
        self.colors = [
            'gray', # 벽
            'black', # 차도
            'white', # 차선
            'yellow', # 중앙선
            'cyan', # 정지선 좌측
            'blue', # 정지선 직진
            'purple', # 정지선 우측
            'red', # 적색신호
            'green' # 녹색신호
        ]
        self.fig, self.ax = plt.subplots()
        self.t = 0
        self.curr_light = RED_LIGHT
        self.map_data = map_data
        self.cx = 0
        self.cy = 0
        self.color = 'orange'

    def __change_light(self):
        for i in range(self.map_data.shape[0]):
            for j in range(self.map_data.shape[1]):
                if self.map_data[i][j] == RED_LIGHT:
                    self.map_data[i][j] = GREEN_LIGHT
                elif self.map_data[i][j] == GREEN_LIGHT:
                    self.map_data[i][j] = RED_LIGHT

    def __visualize_map(self):
        self.ax.clear()
        for i in range(self.map_data.shape[0]):
            for j in range(self.map_data.shape[1]):
                color = self.colors[self.map_data[i][j]]
                rect = patches.Rectangle((j, self.map_data.shape[0] - i - 1), 1, 1, linewidth=0.1, edgecolor='black', facecolor=color)
                self.ax.add_patch(rect)
        # show car
        rect = patches.Rectangle((self.cx, self.map_data.shape[0] - self.cy - 1), 1, 1, linewidth=0.1, edgecolor='black', facecolor=self.color)
        self.ax.add_patch(rect)
        plt.xlim(0, self.map_data.shape[1])
        plt.ylim(0, self.map_data.shape[0])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.text(0, 0, self.t, fontsize=12, color='black')

    def __update(self):
        self.t = (self.t + 1) % 10
        if self.t < 7 and self.curr_light == GREEN_LIGHT:
            self.__change_light()
            self.curr_light = 7
        elif self.t >= 7 and self.curr_light == RED_LIGHT:
            self.__change_light()
            self.curr_light = 8
        self.__visualize_map()

    def set_cx_cy(self, cx, cy, color):
        self.cx = cx
        self.cy = cy
        self.color = color
        print(f'cx: {self.cx}, cy: {self.cy}, color: {self.color}')

    def show(self, cx, cy, color):
        self.set_cx_cy(cx, cy, color)
        plt.show()

if __name__ == '__main__':
    def keyboards():
        import keyboard
        from time import sleep
        cx = 0
        cy = 0
        while True:
            sleep(1)
            if keyboard.is_pressed('a'):
                cx -= 1
            elif keyboard.is_pressed('d'):
                cx += 1
            elif keyboard.is_pressed('w'):
                cy -= 1
            elif keyboard.is_pressed('s'):
                cy += 1
            elif keyboard.is_pressed('q'):
                break
            traffic_world_map.show(cx, cy, 'orange')
    map_data = np.genfromtxt('data/map.csv', delimiter=',', dtype=int)
    traffic_world_map = TrafficWorldMap(map_data)
    traffic_world_map.show(0, 0, 'orange')
    keyboards()