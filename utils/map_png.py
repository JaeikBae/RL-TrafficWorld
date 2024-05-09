# map visualizer from map.csv to map.png
# 0 : 벽
# 1 : 차도
# 2 : 차선
# 3 : 중앙선
# 4 : 정지선 좌측
# 5 : 정지선 직진
# 6 : 정지선 우측
# 7 : 적색신호
# 8 : 녹색신호

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def visualize_map(map_data):
    fig, ax = plt.subplots()
    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            color = colors[map_data[i][j]]
            rect = patches.Rectangle((j, map_data.shape[0] - i - 1), 1, 1, linewidth=0.1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
    plt.xlim(0, map_data.shape[1])
    plt.ylim(0, map_data.shape[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('map.png')

map_data = np.genfromtxt('map.csv', delimiter=',', dtype=int)
visualize_map(map_data)