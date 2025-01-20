import random
from board import CheckersBoard, CheckersMove
import time

import random
from prova_env import CheckersEnv

from stable_baselines3 import PPO

def simulate_game():
    env = CheckersEnv()

    model = PPO("MlpPolicy", env, verbose=1)
    env.set_adversary(model)

    model.learn(total_timesteps=10000)

simulate_game()
