import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame import gfxdraw
import pygame.surfarray
import gymnasium as gym
# from board import CheckersBoard, CheckersMove
from game import Checkers
from time import sleep

class CheckersEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=0.5):
        super(CheckersEnv, self).__init__()
        self.board = Checkers(empty_corner=True)

        self.alpha = alpha

        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 8, 8), dtype=np.int32)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(2, 8, 8), dtype=np.int32)
        self.action_space = spaces.Discrete(64 * 64)
        # self.action_space = spaces.MultiDiscrete([64, 64])
        self.s = None
        self.b_victories = 0
        self.w_victories = 0

    def step(self, action):
        # Obtain the coordinates of the start and end of the move
        # Since the agent move is not necessary a valid move,
        # We use these cordinates to select the closest valid move in the space of valid moves
        # The similarity between the agent move and the valid moves is calculated using the euclidean distance

        action = np.array([action // 64, action % 64])
        # Get the set of all valid moves
        legal_moves = np.array(self.board.legal_moves())
        moves_diff = np.linalg.norm(action - legal_moves, axis=1)
        closest_move = legal_moves[moves_diff.argmin()]
        _, _, _, _, winner, captured = self.board.move(*closest_move)
        self.render('human')
        if winner:
            # print('The winner is', winner)
            self.b_victories += 1 if winner == 'black' else 0
            self.w_victories += 1 if winner == 'white' else 0
            print('B/W:', self.b_victories, self.w_victories, self.b_victories / (self.b_victories + self.w_victories))
            return np.array(self.board.get_observation()), captured * 2 + (12 if winner == 'black' else -12), True, False, {}

        action, _ = self.adversary.predict(self.board.get_observation())
        action = np.array([action // 64, action % 64])
        legal_moves = np.array(self.board.legal_moves())
        moves_diff = np.linalg.norm(action - legal_moves, axis=1)
        closest_move = legal_moves[moves_diff.argmin()]
        _, _, _, _, winner, a_captured = self.board.move(*closest_move)
        self.render('human')
        if winner:
            # print('The winner is', winner)
            self.b_victories += 1 if winner == 'black' else 0
            self.w_victories += 1 if winner == 'white' else 0
            print('B/W:', self.b_victories, self.w_victories, self.b_victories / (self.b_victories + self.w_victories))
            return np.array(self.board.get_observation()), -a_captured * 2 + (12 if winner == 'black' else -12), True, False, {}

        return np.array(self.board.get_observation()), (1 - self.alpha) * -a_captured * 2 + self.alpha * captured * 2, False, False, {}

    def set_adversary(self, adversary):
        self.adversary = adversary

    def reset(self, *args, **kwargs):
        self.board = Checkers(empty_corner=True)
        return np.array(self.board.get_observation()), {}

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

        for p in self.board.board['black']['men']:
            pos = self.board.sq2pos(p)
            gfxdraw.filled_circle(self.s, pos[1] * WIDTH // 8 + WIDTH // 16, pos[0] * HEIGHT // 8 + HEIGHT // 16, WIDTH // 16, (0, 0, 0))
        for p in self.board.board['white']['men']:
            pos = self.board.sq2pos(p)
            gfxdraw.filled_circle(self.s, pos[1] * WIDTH // 8 + WIDTH // 16, pos[0] * HEIGHT // 8 + HEIGHT // 16, WIDTH // 16, (255, 255, 255))
        for p in self.board.board['black']['kings']:
            pos = self.board.sq2pos(p)
            gfxdraw.filled_circle(self.s, pos[1] * WIDTH // 8 + WIDTH // 16, pos[0] * HEIGHT // 8 + HEIGHT // 16, WIDTH // 16, (0, 0, 0))
            gfxdraw.filled_circle(self.s, pos[1] * WIDTH // 8 + WIDTH // 16, pos[0] * HEIGHT // 8 + HEIGHT // 16, WIDTH // 32, (55, 55, 55))
        for p in self.board.board['white']['kings']:
            pos = self.board.sq2pos(p)
            gfxdraw.filled_circle(self.s, pos[1] * WIDTH // 8 + WIDTH // 16, pos[0] * HEIGHT // 8 + HEIGHT // 16, WIDTH // 16, (255, 255, 255))
            gfxdraw.filled_circle(self.s, pos[1] * WIDTH // 8 + WIDTH // 16, pos[0] * HEIGHT // 8 + HEIGHT // 16, WIDTH // 32, (200, 200, 200))

        if mode == 'human':
            pygame.display.flip()
            return None
        else:
            return pygame.surfarray.pixels3d(self.s)

    def close(self):
        pygame.quit()
