class CheckersMove:
    def __init__(self, start, end):
        """
        Initialize a CheckersMove object.
        :param start: Tuple (row, col) indicating the start position.
        :param end: Tuple (row, col) indicating the end position.
        """
        self.start = start
        self.end = end

class CheckersBoard:
    def __init__(self):
        """
        Initialize an 8x8 Checkers board.
        1 represents player 1's pieces, -1 represents player 2's pieces.
        Empty spaces are 0.
        """
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.__turn = 1  # Player 1 starts.
        self._initialize_board()

    def _initialize_board(self):
        """Set up the initial pieces on the board."""
        for row in range(3):  # Player 1 pieces (1)
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = 1

        for row in range(5, 8):  # Player 2 pieces (-1)
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = -1

    @property
    def turn(self) -> int:
        """Return the current player's turn (1 for player 1, -1 for player 2)."""
        return self.__turn

    def valid_moves(self) -> list[CheckersMove]:
        """
        Generate all valid moves for the current player.
        """
        moves = []
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == self.__turn:  # Current player's piece
                    moves.extend(self._generate_piece_moves((row, col)))
        return moves

    def _generate_piece_moves(self, position) -> list[CheckersMove]:
        """
        Generate valid moves for a single piece.
        :param position: Tuple (row, col) of the piece's position.
        """
        row, col = position
        moves = []

        directions = [(-1, -1), (-1, 1)] if self.__turn == 1 else [(1, -1), (1, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board[new_row][new_col] == 0:
                moves.append(CheckersMove(position, (new_row, new_col)))

            # Check for jumps
            jump_row, jump_col = row + 2 * dr, col + 2 * dc
            if (
                0 <= jump_row < 8
                and 0 <= jump_col < 8
                and self.board[new_row][new_col] == -self.__turn
                and self.board[jump_row][jump_col] == 0
            ):
                moves.append(CheckersMove(position, (jump_row, jump_col)))

        return moves

    def move(self, move: CheckersMove):
        """
        Execute a move on the board.
        :param move: CheckersMove object.
        """
        start_row, start_col = move.start
        end_row, end_col = move.end

        # Update the board with the move
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = 0

        # Check if it was a jump and remove the jumped piece
        if abs(end_row - start_row) == 2:
            jumped_row = (start_row + end_row) // 2
            jumped_col = (start_col + end_col) // 2
            self.board[jumped_row][jumped_col] = 0

        # Switch turn
        self.__turn = -self.__turn

    @property
    def winner(self) -> int:
        """
        Determine the winner of the game.
        :return: 1 if player 1 wins, -1 if player 2 wins, 0 if no winner yet.
        """
        player_1_pieces = sum(cell == 1 for row in self.board for cell in row)
        player_2_pieces = sum(cell == -1 for row in self.board for cell in row)

        if player_1_pieces == 0:
            return -1  # Player 2 wins
        elif player_2_pieces == 0:
            return 1  # Player 1 wins
        else:
            return 0  # No winner yet
