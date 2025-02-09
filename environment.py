import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame import gfxdraw
import pygame.surfarray
import gymnasium as gym
from board import CheckersBoard, CheckersMove
from time import sleep

class CheckersEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=0.5):
        super(CheckersEnv, self).__init__()
        self.board = CheckersBoard()

        self.alpha = alpha

        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.int32)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(2, 8, 8), dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([8, 8, 8, 8])
        self.s = None

    def step(self, action):
        # Obtain the coordinates of the start and end of the move
        # Since the agent move is not necessary a valid move,
        # We use these cordinates to select the closest valid move in the space of valid moves
        # The similarity between the agent move and the valid moves is calculated using the euclidean distance

        # Get the set of all valid moves
        valid_moves = self.board.valid_moves()
        done = self.board.winner(len(valid_moves)) != 0
        if done:
            return np.array(self.board.get_observation()), 0, done, False, {}

        moves_latent = np.array([[*m.start, *m.end] for m in valid_moves])
        moves_diff = np.linalg.norm(action - moves_latent, axis=1)
        closest_move = np.argmin(moves_diff)
        piece_captured, reward, pos = self.board.move(valid_moves[closest_move])
        done = self.board.winner(len(valid_moves)) != 0
        if done:
            return np.array(self.board.get_observation()), reward, done, False, {}

        action, _ = self.adversary.predict(self.board.get_observation())
        valid_moves = self.board.valid_moves()
        done = self.board.winner(len(valid_moves)) != 0
        if done:
            return np.array(self.board.get_observation()), reward, done, False, {}
        moves_latent = np.array([[*m.start, *m.end] for m in valid_moves])
        moves_diff = np.linalg.norm(action - moves_latent, axis=1)
        closest_move = np.argmin(moves_diff)
        piece_captured, a_reward, pos = self.board.move(valid_moves[closest_move])
        done = self.board.winner(len(valid_moves)) != 0

        # Clean pygame screen
        self.render('human')
        sleep(0.0125)

        return np.array(self.board.get_observation()), (1 - self.alpha) * -a_reward + self.alpha * reward, done, False, {}

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
        if mode == 'human' and self.s == None:
            pygame.init()
            self.s = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            self.s = pygame.Surface((WIDTH, HEIGHT)) if self.s == None else self.s
        pygame.event.get()
        self.s.fill((22, 36, 71))

        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    gfxdraw.filled_polygon(self.s, [(i * WIDTH // 8, j * HEIGHT // 8),
                                                    ((i + 1) * WIDTH // 8, j * HEIGHT // 8),
                                                    ((i + 1) * WIDTH // 8, (j + 1) * HEIGHT // 8),
                                                    (i * WIDTH // 8, (j + 1) * HEIGHT // 8)], (200, 200, 200))

        for i in range(8):
            for j in range(8):
                if self.board.board[i][j] == 1:
                    gfxdraw.filled_circle(self.s, j * WIDTH // 8 + WIDTH // 16, i * HEIGHT // 8 + HEIGHT // 16, WIDTH // 16, (0, 0, 0))
                elif self.board.board[i][j] == -1:
                    gfxdraw.filled_circle(self.s, j * WIDTH // 8 + WIDTH // 16, i * HEIGHT // 8 + HEIGHT // 16, WIDTH // 16, (255, 255, 255))
                elif self.board.board[i][j] == 2:
                    gfxdraw.filled_circle(self.s, j * WIDTH // 8 + WIDTH // 16, i * HEIGHT // 8 + HEIGHT // 16, WIDTH // 16, (0, 0, 0))
                    gfxdraw.filled_circle(self.s, j * WIDTH // 8 + WIDTH // 16, i * HEIGHT // 8 + HEIGHT // 16, WIDTH // 32, (55, 55, 55))
                elif self.board.board[i][j] == -2:
                    gfxdraw.filled_circle(self.s, j * WIDTH // 8 + WIDTH // 16, i * HEIGHT // 8 + HEIGHT // 16, WIDTH // 16, (255, 255, 255))
                    gfxdraw.filled_circle(self.s, j * WIDTH // 8 + WIDTH // 16, i * HEIGHT // 8 + HEIGHT // 16, WIDTH // 32, (200, 200, 200))
        
        if mode == 'human':
            pygame.display.flip()
            return None
        else:
            return pygame.surfarray.pixels3d(self.s)

    def close(self):
        pygame.quit()
