from stable_baselines3 import PPO, A2C, DQN
from environment import CheckersEnv

def evaluate(model_a, model_b):
    env = CheckersEnv(0.5)
    model_a.set_env(env)
    env.set_adversary(model_b)

    for _ in range(10000):
        obs = env.reset()[0]
        done = False
        trunc = False
        rew = 0
        while not done and not trunc:
            action, _ = model_a.predict(obs)
            obs, reward, done, trunc, _ = env.step(action)
            rew += reward
    print("Game over", rew)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        type=str,
        default="DQN",
        choices=["DQN", "PPO", "A2C"],
        help="Model to optimize hyperparameters for",
    )
    parser.add_argument(
        "-b",
        type=str,
        default="DQN",
        choices=["DQN", "PPO", "A2C"],
        help="Model to optimize hyperparameters for",
    )
    args = parser.parse_args()
    model_a_name = args.a
    model_b_name = args.b

    if model_a_name == "DQN":
        model_a = DQN.load(f"{model_a_name}.pkl")
    elif model_a_name == "PPO":
        model_a = PPO.load(f"{model_a_name}.pkl")
    elif model_a_name == "A2C":
        model_a = A2C.load(f"{model_a_name}.pkl")
    else:
        raise ValueError("Invalid model name")

    if model_b_name == "DQN":
        model_b = DQN.load(f"{model_b_name}.pkl")
    elif model_b_name == "PPO":
        model_b = PPO.load(f"{model_b_name}.pkl")
    elif model_b_name == "A2C":
        model_b = A2C.load(f"{model_b_name}.pkl")
    else:
        raise ValueError("Invalid model name")

    evaluate(model_a, model_b)
