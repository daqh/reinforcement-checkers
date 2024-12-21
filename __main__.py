from checkers.game import Checkers
from checkers.agents.alpha_beta import MinimaxPlayer
from checkers.agents.mcts import MctsPlayer

import matplotlib.pyplot as plt

def main():
    '''A sanity test for a look-ahead agent'''
    ch = Checkers(turn='black')

    players = {
        # 'black': MinimaxPlayer('black', search_depth=3),
        'black': MctsPlayer('black'),
        'white': MctsPlayer('white'),
        # 'white': MinimaxPlayer('white', search_depth=3)
    }

    whites = []
    blacks = []
    winner = None
    while not winner:
        blacks.append(len(ch.board['black']['men']) + len(ch.board['black']['kings']))
        whites.append(len(ch.board['white']['men']) + len(ch.board['white']['kings']))

        player = players[ch.turn]
        from_sq, to_sq = player.next_move(ch.board, ch.last_moved_piece)
        board, turn, last_moved_piece, moves, winner = ch.move(from_sq, to_sq)
    
    blacks.append(len(ch.board['black']['men']) + len(ch.board['black']['kings']))
    whites.append(len(ch.board['white']['men']) + len(ch.board['white']['kings']))
    
    plt.bar(list(range(len(whites))), whites, color='gray', label='White', width=1.0)
    plt.bar(list(range(len(blacks))), blacks, bottom=whites, color='black', label='Black', width=1.0)
    plt.xlim(0, len(whites))
    plt.ylim(0, 24)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title('Number of Pieces during Game')
    plt.xlabel('Move')
    plt.ylabel('Number of Pieces')
    plt.savefig('amount.png')

    print(f'Winner: {winner}')
    ch.print_board()

if __name__ == '__main__':
    main()

