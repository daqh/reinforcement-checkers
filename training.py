import random
import time
import random
from environment import CheckersEnv
from stable_baselines3 import PPO, A2C, DQN
import matplotlib.pyplot as plt
from models import RandomModel
import torch
import tensorboard
from stable_baselines3.common import evaluation, policies
import pickle

timesteps = {
    'DQN': 10000000,
    'PPO': 1000000,
    'A2C': 10000000
}

def simulate_game(model_name):
    env = CheckersEnv(0.5)
    if model_name == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            tensorboard_log="./dqn_logs/",
            **pickle.load(open(f"{model_name}_best_params.pkl", "rb")),
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[256, 512, 512, 1024],
            ),
        )
    elif model_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            tensorboard_log="./ppo_logs/",
            **pickle.load(open(f"{model_name}_best_params.pkl", "rb")),
            policy_kwargs=dict(
                activation_fn=torch.nn.LeakyReLU,
                net_arch=[256, 512, 512, 1024],
            ),
        )
    elif model_name == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            tensorboard_log="./a2c_logs/",
            **pickle.load(open(f"{model_name}_best_params.pkl", "rb")),
            policy_kwargs=dict(
                activation_fn=torch.nn.LeakyReLU,
                net_arch=[256, 512, 512, 1024],
            ),
        )
    env.set_adversary(RandomModel())
    # env.render('human')
    print("Training model 1")
    model = model.learn(total_timesteps=timesteps[model_name], progress_bar=True)
    model.save(f"{model_name}.pkl")

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
