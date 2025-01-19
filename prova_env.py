import gymnasium as gym
from gymnasium import spaces
import numpy as np
from board import CheckersBoard, CheckersMove

class CheckersEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CheckersEnv, self).__init__()
        self.board = CheckersBoard()
        
        self.action_space = spaces.MultiDiscrete([8, 8, 8, 8])  
        self.observation_space = spaces.Box(low=-2, high=2, shape=(8, 8), dtype=np.int32)
    
    def step(self, action):

        valid_moves = self.board.valid_moves()  
        if action in valid_moves:
            piece_captured, reward, pos = self.board.move(action)
            done = self.board.winner(len(valid_moves)) != 0
            return np.array(self.board.board), reward, done, {}
        else:
            return np.array(self.board.board), -1, False, {"invalid_move": True}  

    def reset(self):
        self.board = CheckersBoard()
        return np.array(self.board.board)

    def render(self, mode='human'):
        self.board.print_board()

    def close(self):
        pass
