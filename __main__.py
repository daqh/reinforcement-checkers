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
import optuna
import optuna_dashboard
import tensorboard
from stable_baselines3.common import evaluation, policies

def optimize_hyperparameters(trial):
    config = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
    }
    return config

TRAINING_TIMESTEPS = 1000000

def optimize_agent(trial):
    model_params = optimize_hyperparameters(trial)

    env_1 = CheckersEnv(0.5)

    model_1 = DQN(
        "MlpPolicy",
        env_1,
        verbose=1,
        device="cuda",
        gamma=0.99,
        train_freq=(3, "episode"),
        exploration_fraction=0.95,
        exploration_final_eps=0.05,
        tensorboard_log="./logs/",
        **model_params,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[256, 512, 1024, 2048],
        ),
    )
    model_2 = RandomModel()

    env_1.set_adversary(model_2)

    model_1.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        log_interval=20,
    )

    mean_reward, _ = evaluation.evaluate_policy(model_1, env_1, n_eval_episodes=10)

    return mean_reward

def simulate_game():
    env_1 = CheckersEnv(0.5)
    # env_2 = CheckersEnv()

    model_1 = DQN(
        "MlpPolicy",
        env_1,
        verbose=1,
        device="cuda",
        gamma=0.99,
        train_freq=(3, "episode"),
        exploration_fraction=0.95,
        exploration_final_eps=0.05,
        tensorboard_log="./logs/",
        learning_rate=0.0001,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[256, 512, 1024, 2048],
        ),
    )
    # model_2 = A2C("MlpPolicy", env_2, verbose=1, device="cuda")
    model_2 = RandomModel()
    
    env_1.set_adversary(model_2)
    # env_2.set_adversary(model_1)

    env_1.render('human')

    print("Training model 1")
    model_1 = model_1.learn(total_timesteps=10000000, progress_bar=True)
    # print("Training model 2")
    # model_2 = model_2.learn(total_timesteps=1000, progress_bar=True)

if __name__ == "__main__":
    # simulate_game()

    study = optuna.create_study(
        direction="maximize",
        # study_name="STUDY_NAME",
        # storage="sqlite:///" + "DB_DIR" + ".db",
        load_if_exists=True,
    )
    study.optimize(optimize_agent, n_trials=50, gc_after_trial=True)

