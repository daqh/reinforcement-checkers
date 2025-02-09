import gymnasium as gym
from gymnasium import spaces
import numpy as np
from board import CheckersBoard, CheckersMove
import pygame
from pygame import gfxdraw
import pygame.surfarray
import gymnasium as gym

class CheckersEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CheckersEnv, self).__init__()
        self.board = CheckersBoard()
        
        self.action_space = spaces.MultiDiscrete([8, 8, 8, 8])  
        self.observation_space = spaces.Box(low=-2, high=2, shape=(4, 8, 8), dtype=np.int32)

    def step(self, action):
        valid_moves = self.board.valid_moves()  
        piece_captured, reward, pos = self.board.move(action)
        done = self.board.winner(len(valid_moves)) != 0
        return np.array(self.board.board), reward, done, {}

    def set_adversary(self, adversary):
        self.adversary = adversary

    def reset(self, *args, **kwargs):
        self.board = CheckersBoard()
        return np.array(self.board.board), {}

    def render(self, mode: str = None):
        if mode == None:
            return
        assert mode in ['human', 'rgb_array'], 'mode must be either "human" or "rgb_array"'
        WIDTH, HEIGHT = 800, 800
        if mode == 'human':
            pygame.init()
            s = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            s = pygame.Surface((WIDTH, HEIGHT))
        s.fill((22, 36, 71))

        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    gfxdraw.filled_polygon(s, [(i * WIDTH // 8, j * HEIGHT // 8),
                                                    ((i + 1) * WIDTH // 8, j * HEIGHT // 8),
                                                    ((i + 1) * WIDTH // 8, (j + 1) * HEIGHT // 8),
                                                    (i * WIDTH // 8, (j + 1) * HEIGHT // 8)], (200, 200, 200))
                    
                    # TODO: Draw pieces
        
        if mode == 'human':
            pygame.display.flip()
            return None
        else:
            return pygame.surfarray.pixels3d(s)
    
    def close(self):
        pass
