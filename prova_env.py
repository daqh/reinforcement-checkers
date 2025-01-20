import gymnasium as gym
from gymnasium import spaces
import numpy as np
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
        self.board.print_board()
        return np.array(self.board.get_observation()), -a_reward + reward, done, False, {}

    def set_adversary(self, adversary):
        self.adversary = adversary

    def reset(self, *args, **kwargs):
        self.board = CheckersBoard()
        return np.array(self.board.board), {}

    def render(self, mode='human'):
        self.board.print_board()

    def close(self):
        pass
