import random
from board import CheckersBoard, CheckersMove
import time
import random
from environment import CheckersEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC, ppo
import matplotlib.pyplot as plt
from copy import deepcopy

def simulate_game():
    env_1 = CheckersEnv(0.9)
    env_2 = CheckersEnv()

    model_1 = PPO("MlpPolicy", env_1, verbose=1, device="cuda")
    model_2 = SAC("MlpPolicy", env_2, verbose=1, device="cuda")
    
    # env_1.set_adversary(model_2)
    env_2.set_adversary(model_1)

    env_1.render('human')

    for _ in range(5):
        # print("Training model 1")
        # model_1.learn(total_timesteps=1000, progress_bar=True)
        print("Training model 2")
        model_2.learn(total_timesteps=1000, progress_bar=True)

if __name__ == "__main__":
    simulate_game()
