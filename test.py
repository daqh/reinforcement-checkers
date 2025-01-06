import random
from board import CheckersBoard, CheckersMove
import time

def random_move(board: CheckersBoard) -> CheckersMove:
    valid_moves = board.valid_moves()
    for m in valid_moves:
        print(m.start, m.end)
    if valid_moves:
        return random.choice(valid_moves)
    else:
        return None

def simulate_game():
    board = CheckersBoard()
    board.print_board()

    total_rewards = {1: 0, -1: 0}

    winner = 0

    while board.winner == 0:
        move = random_move(board)
        
        if move is None:
            winner = -board.turn
            print(f"\nGame over! Player {winner} wins because the opponent has no valid moves!")
            # board.print_board()
            break

        print(f"Player {board.turn}'s turn:")
        print(f"\nMove: {move.start} -> {move.end}")

        piece_captured, reward, pos = board.move(move)
        total_rewards[-board.turn] += reward
        
        print(f"Reward for this move: {reward} for {-board.turn}")
        if pos is not None:
            print(f"Piece captured: {pos}")
        print(f"Total rewards: Player 1: {total_rewards[1]} | Player 2: {total_rewards[-1]}")
        board.print_board()
        print("-" * 100)
        # time.sleep(3.5)

    print("-" * 100)
    print(f"\nMio Player {winner} wins!")
    print(f"\nPlayer {board.winner} wins!")
    print(f"Final Scores: Player 1: {total_rewards[1]} | Player 2: {total_rewards[-1]}")
    return total_rewards

# t = simulate_game()

while True:
    t = simulate_game()
    if t[1] > 12 or t[-1] > 12:
        print(t[1],t[-1])
        break
