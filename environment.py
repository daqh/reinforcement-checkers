import pygame
from pygame import gfxdraw
import pygame.surfarray
import gymnasium as gym
import matplotlib.pyplot as plt
from board import CheckersBoard

class CheckersEnv(gym.Env):

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
        pygame.quit()