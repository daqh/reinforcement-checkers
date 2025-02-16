import random
import time
import random
from environment import CheckersEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C, DQN
import matplotlib.pyplot as plt
from copy import deepcopy
from models import RandomModel
import torch
import tensorboard
from stable_baselines3.common import evaluation, policies
import pickle

def simulate_game(model_name):
    env = CheckersEnv(0.5)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        tensorboard_log="./logs/",
        *pickle.load(open("dqn_best_params.pkl", "rb")),
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[256, 512, 1024, 2048],
        ),
    )
    env.set_adversary(RandomModel())
    env.render('human')
    print("Training model 1")
    model = model.learn(total_timesteps=10000000, progress_bar=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="DQN",
        choices=["DQN", "PPO", "A2C"],
        help="Model to optimize hyperparameters for",
    )
    args = parser.parse_args()
    model_name = args.model
    simulate_game(model_name)
