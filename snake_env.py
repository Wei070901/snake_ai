# snake_env.py
import pygame
import random
import numpy as np
from gym import Env, spaces

# snake_env.py

class SnakeEnv(Env):
    def __init__(self, grid_size=12, block_size=40, render_mode=False):
        super(SnakeEnv, self).__init__()
        pygame.init()
        
        self.render_mode = render_mode  # 新增渲染模式參數
        self.grid_size = grid_size
        self.block_size = block_size
        self.width, self.height = block_size * grid_size, block_size * grid_size
        if self.render_mode:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake Game")
        else:
            self.display = None  # 不初始化顯示器
        
        # 顏色定義
        self.light_green = (170, 215, 81)
        self.dark_green = (162, 209, 73)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.dark_blue = (0, 0, 139)
        
        # 動作空間：上、下、左、右
        self.action_space = spaces.Discrete(4)
        
        # 狀態空間：簡化後的狀態表示
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        
        # 初始化字體和時鐘
        if self.render_mode:
            self.font_style = pygame.font.SysFont(None, 35)
            self.score_font = pygame.font.SysFont(None, 25)
            self.clock = pygame.time.Clock()
        
        self.reset()
    
    def reset(self):
        self.game_over = False
        self.game_close = False
        
        # 初始位置對齊網格
        self.x1 = self.width // 2 // self.block_size * self.block_size
        self.y1 = self.height // 2 // self.block_size * self.block_size
        
        self.x1_change = 0
        self.y1_change = 0
        
        self.snake_list = []
        self.length_of_snake = 1
        
        # 食物位置對齊網格
        self.foodx = round(random.randrange(0, self.width - self.block_size) / self.block_size) * self.block_size
        self.foody = round(random.randrange(0, self.height - self.block_size) / self.block_size) * self.block_size
        
        # 初始化食物距離感知
        self.food_distance_x = (self.foodx - self.x1) / self.width
        self.food_distance_y = (self.foody - self.y1) / self.height
        
        if self.render_mode:
            self.render()
        
        return self.get_state()
    
    def step(self, action):
        reward = 0  # 添加步進獎勵
        done = False
        info = {}
        
        # 計算當前蛇頭和食物的距離
        current_distance = np.sqrt((self.foodx - self.x1)**2 + (self.foody - self.y1)**2)
        # 計算新的蛇頭和食物之間的距離
        new_distance = np.sqrt((self.foodx - (self.x1 + self.x1_change))**2 + (self.foody - (self.y1 + self.y1_change))**2)
        
        self.food_distance_x = (self.foodx - self.x1) / self.width
        self.food_distance_y = (self.foody - self.y1) / self.height
        if self.x1 >= self.width or self.x1 < 0 or self.y1 >= self.height or self.y1 < 0:
            done = True
            reward += -15  # 可以調整為更大的負值，如 -1.5，讓模型更加「害怕」撞牆
        
        distance_reward = (1 + (self.length_of_snake - 1) * 2)  # 獎勵隨蛇長度遞增
        if new_distance < current_distance:
            reward += distance_reward  # 更接近食物，加分
        else:
            reward -= distance_reward  # 遠離食物，扣分
        
        # 動作處理，防止蛇反向移動
        if action == 0 and self.y1_change == 0:  # 上
            self.x1_change = 0
            self.y1_change = -self.block_size
        elif action == 1 and self.y1_change == 0:  # 下
            self.x1_change = 0
            self.y1_change = self.block_size
        elif action == 2 and self.x1_change == 0:  # 左
            self.x1_change = -self.block_size
            self.y1_change = 0
        elif action == 3 and self.x1_change == 0:  # 右
            self.x1_change = self.block_size
            self.y1_change = 0
        
        # 更新位置
        self.x1 += self.x1_change
        self.y1 += self.y1_change
        
        # 檢查是否撞牆
        if self.x1 >= self.width or self.x1 < 0 or self.y1 >= self.height or self.y1 < 0:
            done = True
            if self.render_mode:
                self.render()
            return self.get_state(), reward, done, info
        
        # 更新蛇的位置
        snake_head = [self.x1, self.y1]
        self.snake_list.append(snake_head)
        if len(self.snake_list) > self.length_of_snake:
            del self.snake_list[0]
        
        # 檢查是否撞到自己
        for x in self.snake_list[:-1]:
            if x == snake_head:
                done = True
                reward += -10
                if self.render_mode:
                    self.render()
                return self.get_state(), reward, done, info
        
        # 檢查是否吃到食物
        if self.x1 == self.foodx and self.y1 == self.foody:
            self.foodx = round(random.randrange(0, self.width - self.block_size) / self.block_size) * self.block_size
            self.foody = round(random.randrange(0, self.height - self.block_size) / self.block_size) * self.block_size
            self.length_of_snake += 1
            reward += 10 + (self.length_of_snake - 1) * 2  # 每多一節，增加 2 分
        
        # 渲染畫面
        if self.render_mode:
            self.render()
        
        return self.get_state(), reward, done, info
    
    def render(self, mode='human'):
        if not self.render_mode:
            return
        
        # 處理 Pygame 事件以保持窗口響應
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 繪製棋盤格
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                color = self.light_green if (row + col) % 2 == 0 else self.dark_green
                pygame.draw.rect(self.display, color, (col * self.block_size, row * self.block_size, self.block_size, self.block_size))

        # 畫食物
        pygame.draw.rect(self.display, self.red, [self.foodx, self.foody, self.block_size, self.block_size])
        
        # 畫蛇
        for x in self.snake_list:
            pygame.draw.rect(self.display, self.dark_blue, [x[0], x[1], self.block_size, self.block_size])
            pygame.draw.rect(self.display, self.blue, [x[0] + 2, x[1] + 2, self.block_size - 4, self.block_size - 4])
        
        # 顯示分數
        self.display_score(self.length_of_snake - 1)
        
        pygame.display.update()
        self.clock.tick(2000)  # 控制遊戲速度
    
    def display_score(self, score):
        value = self.score_font.render("Score: " + str(score), True, (255, 255, 255))
        self.display.blit(value, [0, 0])
        self.score = score
    
    def get_state(self):
        # 獲取當前狀態（簡化後的狀態表示）
        state = np.array([
            self.x1 / self.width,
            self.y1 / self.height,
            self.foodx / self.width,
            self.foody / self.height,
            self.x1_change / self.block_size,
            self.y1_change / self.block_size,
            self.length_of_snake / (self.grid_size ** 2),
            #計算蛇頭和食物的水平或垂直距離
            self.food_distance_x,
            self.food_distance_y,
            # 障礙物感知（0 表示安全，1 表示危險）
            1 if self.x1 + self.x1_change < 0 or self.x1 + self.x1_change >= self.width or [self.x1 + self.x1_change, self.y1 + self.y1_change] in self.snake_list else 0,  # 前方障礙
            1 if (self.y1_change == 0 and (self.x1 + self.block_size >= self.width or [self.x1 + self.block_size, self.y1] in self.snake_list)) or \
                (self.x1_change == 0 and (self.y1 - self.block_size < 0 or [self.x1, self.y1 - self.block_size] in self.snake_list)) else 0,  # 左方障礙
            1 if (self.y1_change == 0 and (self.x1 - self.block_size < 0 or [self.x1 - self.block_size, self.y1] in self.snake_list)) or \
                (self.x1_change == 0 and (self.y1 + self.block_size >= self.height or [self.x1, self.y1 + self.block_size] in self.snake_list)) else 0  # 右方障礙
        ], dtype=np.float32)
        
        return state
    
    def close(self):
        pygame.quit()
