# Reinforcement Checkers

![Match Between untrained PPO and the Random Model](/asset/header.png)

This repository shows how to train multiple reinforcement learning algorithms in an adversarial setting using the checkers game.

## 1. Hyperparameter Optimization
```
python3 hp_optimization.py --model DQN
```

## 2. Training
```
python3 training.py --model DQN
```

![Match Between untrained PPO and the Random Model](/asset/match.gif)

## 3. Evaluation
```
python3 eval.py --model DQN PPO
```
