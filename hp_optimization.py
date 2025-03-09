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
import tensorboard
from stable_baselines3.common import evaluation, policies

TRAINING_TIMESTEPS = 500000

def optimize_hyperparameters(trial, model_name: str):
    if model_name == "DQN":
        config = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "batch_size": trial.suggest_int("batch_size", 2, 128),
            "buffer_size": trial.suggest_int("buffer_size", 1000, 100000),
            "learning_starts": trial.suggest_int("learning_starts", 0, 1000),
            "train_freq": (trial.suggest_int("train_freq", 1, 10), "episode"),
            "gradient_steps": trial.suggest_int("gradient_steps", 1, 10),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.01, 0.99),
            "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
        }
    elif model_name == "PPO":
        config = {
            "n_steps": trial.suggest_int("n_steps", 16, 2048),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 0.01),
            "gamma": trial.suggest_loguniform("gamma", 0.9, 0.9999),
            "n_epochs": trial.suggest_int("n_epochs", 1, 10),
            "batch_size": trial.suggest_int("batch_size", 2, 256),
            "ent_coef": trial.suggest_loguniform("ent_coef", 1e-8, 1e-1), 
        }
    elif model_name == "A2C":
        config = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 0.01),
        "gamma": trial.suggest_loguniform("gamma", 0.9, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
        "n_steps": trial.suggest_int("n_steps", 16, 2048),
        "ent_coef": trial.suggest_loguniform("ent_coef", 1e-8, 1e-1),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.1, 10),
    }
    return config

from random import randint

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

    print(f"Optimizing hyperparameters for {model_name}")

    try:
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_study",
            storage="sqlite:///HP.db",
            load_if_exists=False,
        )
    except:
        optuna.delete_study(f"{model_name}_study", storage="sqlite:///HP.db")
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_study",
            storage="sqlite:///HP.db",
        )

    def optimize_agent(trial):
        model_params = optimize_hyperparameters(trial, model_name)
        env = CheckersEnv(0.5)
        if model_name == "DQN":
            model = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                device="cuda",
                tensorboard_log=f"./dqn_hp_logs_{randint(0,99999)}/",
                **model_params,
                policy_kwargs=dict(
                    activation_fn=torch.nn.LeakyReLU,
                    net_arch=[256, 512, 512, 1024],
                ),
            )
        elif model_name == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="cuda",
                tensorboard_log=f"./ppo_hp_logs_{randint(0,99999)}/",
                **model_params,
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
                tensorboard_log=f"./a2c_hp_logs_{randint(0,99999)}/",
                **model_params,
                policy_kwargs=dict(
                    activation_fn=torch.nn.LeakyReLU,
                    net_arch=[256, 512, 512, 1024],
                ),
            )

        env.set_adversary(RandomModel())
        model.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            log_interval=20,
            progress_bar=False,
        )
        mean_reward, _ = evaluation.evaluate_policy(model, env, n_eval_episodes=10)
        return mean_reward

    study.optimize(optimize_agent, n_trials=50, gc_after_trial=True, n_jobs=8, show_progress_bar=True)

    import pickle

    # Save best params
    with open(f"{model_name}_best_params.pkl", "wb") as f:
        pickle.dump(study.best_params, f)
