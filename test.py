import random
from board import CheckersBoard, CheckersMove
import time

# def random_move(board: CheckersBoard) -> CheckersMove:
#     valid_moves = board.valid_moves()
#     for m in valid_moves:
#         print(m.start, m.end)
#     if valid_moves:
#         return random.choice(valid_moves)
#     else:
#         return None

# def simulate_game():
#     board = CheckersBoard()
#     board.print_board()

#     total_rewards = {1: 0, -1: 0}

#     winner = 0

#     while winner == 0:
#         move = random_move(board)
#         winner = board.winner(move)
#         if winner != 0:
#             break
#         # if move is None:
#         #     winner = -board.turn
#         #     print(f"\nGame over! Player {winner} wins because the opponent has no valid moves!")
#         #     # board.print_board()
#         #     break

#         print(f"Player {board.turn}'s turn:")
#         print(f"\nMove: {move.start} -> {move.end}")

#         piece_captured, reward, pos = board.move(move)
#         total_rewards[-board.turn] += reward
        
#         print(f"Reward for this move: {reward} for {-board.turn}")
#         if pos is not None:
#             print(f"Piece captured: {pos}")
#         print(f"Total rewards: Player 1: {total_rewards[1]} | Player 2: {total_rewards[-1]}")
#         board.print_board()
#         print("-" * 100)
#         # time.sleep(3.5)

#     print("-" * 100)
#     print(f"\nMio Player {winner} wins!")
#     print(f"Final Scores: Player 1: {total_rewards[1]} | Player 2: {total_rewards[-1]}")
#     return total_rewards

# t = simulate_game()

# # while True:
# #     t = simulate_game()
# #     if t[1] > 12 or t[-1] > 12:
# #         print(t[1],t[-1])
# #         break

import random
from prova_env import CheckersEnv

def simulate_game():
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

simulate_game()
