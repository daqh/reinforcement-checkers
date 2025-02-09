import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame import gfxdraw
import pygame.surfarray
import gymnasium as gym
from board import CheckersBoard, CheckersMove

class CheckersEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CheckersEnv, self).__init__()
        self.board = CheckersBoard()

        self.action_space = spaces.Box(low=-2, high=2, shape=(2, 8, 8), dtype=np.int32)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(4, 8, 8), dtype=np.int32)

    def step(self, action):
        start = action[0].argmax() // 8, action[0].argmax() % 8
        end = action[1].argmax() // 8, action[1].argmax() % 8
        action = np.array([*start, *end])
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
        start = action[0].argmax() // 8, action[0].argmax() % 8
        end = action[1].argmax() // 8, action[1].argmax() % 8
        action = np.array([*start, *end])
        valid_moves = self.board.valid_moves()
        done = self.board.winner(len(valid_moves)) != 0
        if done:
            return np.array(self.board.get_observation()), 0, done, False, {}
        moves_latent = np.array([[*m.start, *m.end] for m in valid_moves])
        moves_diff = np.linalg.norm(action - moves_latent, axis=1)
        closest_move = np.argmin(moves_diff)
        piece_captured, a_reward, pos = self.board.move(valid_moves[closest_move])
        done = self.board.winner(len(valid_moves)) != 0
        # self.board.print_board()
        return np.array(self.board.get_observation()), -a_reward + reward, done, False, {}

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
        pygame.quit()
