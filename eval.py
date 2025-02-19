import dill

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

    model_a = dill.load(open(f"{model_a_name}.pkl", "rb"))
    model_b = dill.load(open(f"{model_b_name}.pkl", "rb"))

    print(f"Model A: {model_a}")