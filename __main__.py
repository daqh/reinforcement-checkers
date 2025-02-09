import random
from board import CheckersBoard, CheckersMove
import time

import random
from environment import CheckersEnv

def main():
    env = CheckersEnv()

    obs = env.reset()
    done = False
    total_reward = {1: 0, -1: 0}  

    print("Inizio partita")
    env.render()

    while not done:
        valid_moves = env.board.valid_moves()  

        if valid_moves:
            move = random.choice(valid_moves)

            obs, reward, done, info = env.step(move)

            total_reward[-env.board.turn] += reward 

            env.render()  
            print(f"Mossa: {move.start} -> {move.end}, Reward: {reward}, Done: {done}")
        else:
            print(f"Nessuna mossa valida disponibile per il giocatore {env.board.turn}.")
            done = True  

    print(f"Partita terminata, Reward totale per il giocatore 1: {total_reward[1]}, giocatore 2: {total_reward[-1]}")
    print(f"Vincitore: {env.board.winner(len(valid_moves))}")

if __name__ == "__main__":
    main()
