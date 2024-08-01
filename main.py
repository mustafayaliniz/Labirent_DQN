import pygame
import numpy as np
import sys
import random
from labirentler import Maze
import time
from dqn_model import DQNAgent
import torch

class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((600, 540))
        pygame.display.set_caption('Labirent DQN')

        self.labirent_baslangic = 0
        self.labirent = Maze.maze(self.labirent_baslangic)
        if self.labirent is None:
            print("Labirent yüklenemedi")
            pygame.quit()
            sys.exit()

        self.block_size = 60
        self.player_pos = [1, 1]
        self.banana_pos = [1, 7]

        self.player_image = pygame.image.load('images/monkey.bmp')
        self.banana_image = pygame.image.load('images/banana.bmp')

        self.state_size = len(self.labirent) * len(self.labirent[0])
        self.action_size = 4

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dqn_agent = DQNAgent(self.state_size, self.action_size, self.device)
        self.target_agent = DQNAgent(self.state_size, self.action_size, self.device)
        self.update_target_model()

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.batch_size = 32
        self.replay_buffer = []
        self.max_buffer_size = 10000

        self.total_q_change = 0
        self.num_steps = 0
        self.sayac = 0
        self.deneme = 0
        self.odul = 0
        self.start_time = time.time()
        self.training_duration = 120  # 2 dakika
        print(f'kullanilan cihaz: {self.device}')
    def draw_maze(self):
        for row in range(len(self.labirent)):
            for col in range(len(self.labirent[row])):
                color = (255, 255, 255) if self.labirent[row][col] == 1 else (0, 0, 0)
                pygame.draw.rect(self.screen, color,
                                 (col * self.block_size, row * self.block_size, self.block_size, self.block_size))

    def draw_monkey(self):
        self.screen.blit(self.player_image,
                         (self.player_pos[0] * self.block_size, self.player_pos[1] * self.block_size))

    def draw_banana(self):
        self.screen.blit(self.banana_image,
                         (self.banana_pos[0] * self.block_size, self.banana_pos[1] * self.block_size))

    def move_player(self, action):
        directions = ['up', 'down', 'left', 'right']
        direction_vectors = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}
        dx, dy = direction_vectors[directions[action]]
        new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

        if (0 <= new_pos[0] < len(self.labirent[0])) and (0 <= new_pos[1] < len(self.labirent)) and (
              self.labirent[new_pos[1]][new_pos[0]] == 0):
            reward = -0.01  # Küçük ceza

            if new_pos == self.banana_pos:
                reward = 1
                self.labirent = Maze.maze(self.labirent_baslangic)
                self.player_pos = [1, 1]
                self.sayac += 1
                self.deneme += 1
                self.start_time = time.time()
                self.odul += 1
            else:
                self.player_pos = new_pos

            self.store_transition(self.get_state(self.player_pos), action, reward, self.get_state(new_pos))
            self.replay_experience()

    def get_state(self, pos):
        state = np.zeros(self.state_size, dtype=np.float32)
        state[pos[1] * len(self.labirent[0]) + pos[0]] = 1
        return state

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.dqn_agent.predict(state)[0])

    def update_target_model(self):
        self.target_agent.model.load_state_dict(self.dqn_agent.model.state_dict())

    def store_transition(self, state, action, reward, next_state):
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state))

    def replay_experience(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state in batch:
            target = reward + self.gamma * np.amax(self.target_agent.predict(next_state)[0])
            target_f = self.dqn_agent.predict(state)
            target_f[0][action] = target
            loss=self.dqn_agent.fit(state, target_f[0])
            print(f'loss: {loss}')
    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            state = self.get_state(self.player_pos)
            action = self.choose_action(state)

            self.move_player(action)

            self.screen.fill((0, 0, 0))
            self.draw_maze()
            self.draw_monkey()
            self.draw_banana()

            current_time = time.time()
            elapsed_time = current_time - self.start_time
            font = pygame.font.Font(None, 36)
            elapsed_time_text = font.render(f'Geçen Süre: {elapsed_time:.2f} saniye ', True, (0, 0, 0))
            deneme_txt = font.render(f'Deneme: {self.deneme}', True, (0, 0, 0))
            odul_txt = font.render(f'Ödül: {self.odul}', True, (0, 0, 0))

            text_rect = elapsed_time_text.get_rect()
            text_rect.bottomleft = (10, 540 - 10)  # Sol alt köşe
            epouch = deneme_txt.get_rect()
            epouch.bottomright = (540, 540 - 10)
            prize = odul_txt.get_rect()
            prize.topleft = (10, 10)

            self.screen.blit(elapsed_time_text, text_rect)
            self.screen.blit(deneme_txt, epouch)
            self.screen.blit(odul_txt, prize)

            if elapsed_time > self.training_duration:
                self.labirent = Maze.maze(self.labirent_baslangic)
                self.player_pos = [1, 1]
                self.start_time = time.time()
                self.deneme += 1

            # Epsilon'u azalt
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

            # Hedef modeli belirli aralıklarla güncelle
            if self.num_steps % 1000 == 0:
                self.update_target_model()

            pygame.display.flip()
            self.num_steps += 1


if __name__ == '__main__':
    Main().run()
