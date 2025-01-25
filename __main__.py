import random
from board import CheckersBoard, CheckersMove
import time
import random
from prova_env import CheckersEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC
import matplotlib.pyplot as plt

def simulate_game():
    rpes_1 = []
    rpes_2 = []
    
    env_1 = CheckersEnv()
    env_2 = CheckersEnv()

    model_1 = PPO("MlpPolicy", env_1, verbose=1)
    model_2 = SAC("MlpPolicy", env_2, verbose=1)
    
    env_1.set_adversary(model_2)
    env_2.set_adversary(model_1)
    
    for _ in range(3):
        model_1.learn(total_timesteps=1000, progress_bar=True)
        rpe_mu, rpe_sigma = evaluate_policy(model_1, env_1, n_eval_episodes=1)
        print(f"evaluate model_1: {rpe_mu}")
        rpes_1.append(rpe_mu)
        model_2.learn(total_timesteps=1000, progress_bar=True)
        rpe_mu, rpe_sigma = evaluate_policy(model_2, env_2, n_eval_episodes=1)
        print(f"evaluate model_2: {rpe_mu}")
        rpes_2.append(rpe_mu)
    
    plt.plot(rpes_1)
    plt.plot(rpes_2)
    plt.savefig("rpe.png") 
    # TODO: play n games and track results

if __name__ == "__main__":
    simulate_game()