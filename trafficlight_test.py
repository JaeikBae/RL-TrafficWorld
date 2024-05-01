# change traffic light color and visualize the map in real-time
# 5 seconds green light, 2 seconds red light
# traffic light color is 7 or 8
# map from map.csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from time import sleep


colors = [
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

fig, ax = plt.subplots()

t = 0
curr = 7
map_data = np.genfromtxt('map.csv', delimiter=',', dtype=int)

def change_light(map_data):
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            if map_data[i][j] == 7:
                map_data[i][j] = 8
            elif map_data[i][j] == 8:
                map_data[i][j] = 7
    return map_data

def visualize_map(map_data):
    ax.clear()
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            color = colors[map_data[i][j]]
            rect = patches.Rectangle((j, map_data.shape[0] - i - 1), 1, 1, linewidth=0.1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
    plt.xlim(0, map_data.shape[1])
    plt.ylim(0, map_data.shape[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.text(0, 0, t, fontsize=12, color='black')

def update(frame):
    global t, curr, map_data
    t = (t + 1) % 10
    if t < 7 and curr == 8:
        map_data = change_light(map_data)
        curr = 7
    elif t >= 7 and curr == 7:
        map_data = change_light(map_data)
        curr = 8
    visualize_map(map_data)

ani = animation.FuncAnimation(fig, update, interval=1000)
plt.show()