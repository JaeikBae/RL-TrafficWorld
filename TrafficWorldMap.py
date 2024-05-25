import pygame # type: ignore
import numpy as np

RED_LIGHT = 7
GREEN_LIGHT = 8

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
        self.t = 0
        self.curr_light = RED_LIGHT
        self.map_data = map_data
        self.cx = 32
        self.cy = 45
        self.color = 'orange'
        
        self.text = ""

    def get_state(self, cy, cx):
        return {
            'map_data': self.map_data,
            'current_light': self.curr_light,
            'time': self.t
        }

    def set_text(self, text):
        self.text = text

    def time_step(self):
        self.t = self.t + 1
        self.change_light()

    def change_light(self):
        isWillChange = False
        if self.curr_light == GREEN_LIGHT and self.t % 10 in [0, 1, 2, 3, 4]:
            self.curr_light = RED_LIGHT
            isWillChange = True
        elif self.curr_light == RED_LIGHT and self.t % 10 in [5, 6, 7, 8, 9]:
            self.curr_light = GREEN_LIGHT
            isWillChange = True

        if isWillChange:
            for y in range(self.map_data.shape[0]):
                for x in range(self.map_data.shape[1]):
                    if self.map_data[y][x] == RED_LIGHT:
                        self.map_data[y][x] = GREEN_LIGHT
                    elif self.map_data[y][x] == GREEN_LIGHT:
                        self.map_data[y][x] = RED_LIGHT
            

    def set_cx_cy(self, cx, cy):
        self.cx = cx
        self.cy = cy

    def show_car(self):
        # 차량을 표시하기 위한 사각형 설정
        car_rect = pygame.Rect(self.cx * 10, self.cy * 10, 10, 10)  # 사각형 크기와 위치 조정
        pygame.draw.rect(self.screen, pygame.Color(self.color), car_rect)  # 사각형 그리기

    def start_visualization(self):
        pygame.init()
        screen = pygame.display.set_mode((self.map_data.shape[1] * 10, self.map_data.shape[0] * 10 + 70))
        self.screen = screen
        running = True
        while running:
            # print time on title
            pygame.display.set_caption('Time: {}'.format(self.t))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            screen.fill(pygame.Color('black'))
            for y in range(self.map_data.shape[0]):
                for x in range(self.map_data.shape[1]):
                    rect = pygame.Rect(x * 10, y * 10, 10, 10)
                    color = pygame.Color(self.colors[np.abs(self.map_data[y][x])])
                    pygame.draw.rect(screen, color, rect)

            self.show_car()
            # add some texts at bottom
            font = pygame.font.SysFont(None, 20)
            line1 = "Time: {}".format(self.t)
            line2 = self.text[0]
            line3 = self.text[1]
            coded_line1 = font.render(line1, True, (255, 255, 255))
            coded_line2 = font.render(line2, True, (255, 255, 255))
            coded_line3 = font.render(line3, True, (255, 255, 255))
            screen.blit(coded_line1, (10, self.map_data.shape[0] * 10+10))
            screen.blit(coded_line2, (10, self.map_data.shape[0] * 10+30))
            screen.blit(coded_line3, (10, self.map_data.shape[0] * 10+50))
            pygame.display.flip()
            
        pygame.quit()

    def close(self):
        pygame.quit()

    def move_car(self, direction):
        if direction == 'w':  # 위로 이동
            self.cy -= 1
        elif direction == 's':  # 아래로 이동
            self.cy += 1
        elif direction == 'a':  # 왼쪽으로 이동
            self.cx -= 1
        elif direction == 'd':  # 오른쪽으로 이동
            self.cx += 1

        # 맵의 경계를 넘지 않도록 조정
        self.cx = max(0, min(self.cx, self.map_data.shape[1] - 1))
        self.cy = max(0, min(self.cy, self.map_data.shape[0] - 1))

    def reset(self):
        self.t = 0
        self.curr_light = RED_LIGHT
        self.cx = 32
        self.cy = 45
        self.color = 'orange'


if __name__ == '__main__':
    import numpy as np
    pygame.init()
    
    map_data = np.genfromtxt('data/map.csv', delimiter=',', dtype=np.int32)
    world_map = TrafficWorldMap(map_data)
    
    screen = pygame.display.set_mode((map_data.shape[1] * 10, map_data.shape[0] * 10))
    world_map.screen = screen
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    world_map.move_car('w')
                elif event.key == pygame.K_a:
                    world_map.move_car('a')
                elif event.key == pygame.K_s:
                    world_map.move_car('s')
                elif event.key == pygame.K_d:
                    world_map.move_car('d')
        
        screen.fill(pygame.Color('gray'))  # 배경 색상 설정

        # 맵 데이터를 이용해 타일 그리기
        for y in range(map_data.shape[0]):
            for x in range(map_data.shape[1]):
                rect = pygame.Rect(x * 10, y * 10, 10, 10)
                color = pygame.Color(world_map.colors[map_data[y][x]])
                pygame.draw.rect(screen, color, rect)

        world_map.show_car()  # 차량 표시
        world_map.change_light()  # 신호등 변경
        
        pygame.display.flip()
        pygame.time.wait(100)  # 시뮬레이션 속도 조정

    pygame.quit()
