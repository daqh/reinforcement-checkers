import random
from board import CheckersBoard, CheckersMove
import time

import random
from prova_env import CheckersEnv

from stable_baselines3 import PPO, SAC

def simulate_game():
    env = CheckersEnv()

    ppo = PPO("MlpPolicy", env, verbose=1)
    env.set_adversary(ppo)
    ppo.learn(total_timesteps=10000)

    sac = SAC("MlpPolicy", env, verbose=1)
    env.set_adversary(ppo)
    sac.learn(total_timesteps=10000)

simulate_game()
