class CheckersMove:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class CheckersBoard:
    def __init__(self):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        self.__turn = 1 
        self._initialize_board()

    def _initialize_board(self):
        for row in range(3):  
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = 1

        for row in range(5, 8): 
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = -1

    def print_board(self):
        print("Current board:")
        
        col_headers = "   " + " ".join(f"{col:>2}" for col in range(8))
        print(col_headers)
        print("   " + "-" * (3 * 8)) 

        for row_index, row in enumerate(self.board):
            row_str = f"{row_index:<2} | " + " ".join(
                ' O ' if cell == 0 else 
                (' 1 ' if cell == 1 else 
                '-1 ' if cell == -1 else 
                ' 2 ' if cell == 2 else 
                '-2 ')
                for cell in row
            )
            print(row_str)

        print()

    @property
    def turn(self) -> int:
        return self.__turn

    def valid_moves(self) -> list[CheckersMove]:
        moves = []
        captures_available = False
        
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == self.__turn or self.board[row][col] == self.__turn * 2:
                    piece_moves = self._generate_piece_moves((row, col))
                    moves.extend(piece_moves)

                    if any(abs(move.start[0] - move.end[0]) == 2 for move in piece_moves):
                        captures_available = True

        if captures_available:
            moves = [move for move in moves if abs(move.start[0] - move.end[0]) == 2]

        return moves

    def _generate_piece_moves(self, position) -> list[CheckersMove]:
        row, col = position
        moves = []

        directions = [(-1, -1), (-1, 1)] if self.__turn == -1 else [(1, -1), (1, 1)]

        if abs(self.board[row][col]) != 2:
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board[new_row][new_col] == 0:
                    moves.append(CheckersMove(position, (new_row, new_col)))

                jump_row, jump_col = row + 2 * dr, col + 2 * dc
                if (
                    0 <= jump_row < 8
                    and 0 <= jump_col < 8
                    and self.board[new_row][new_col] == -self.__turn
                    and self.board[jump_row][jump_col] == 0
                ):
                    moves.append(CheckersMove(position, (jump_row, jump_col)))

        if abs(self.board[row][col]) == 2:
            for dr in [-1, 1]: 
                for dc in [-1, 1]:  
                    for i in range(1, 8): 
                        new_row, new_col = row + dr * i, col + dc * i
                        if 0 <= new_row < 8 and 0 <= new_col < 8:
                            print("Cazzoooo",self.board[new_row][new_col])
                            if self.board[new_row][new_col] == -self.__turn or self.board[new_row][new_col] == -self.__turn * 2:
                                jump_row, jump_col = new_row + dr, new_col + dc
                                if abs(new_row - row) == 1 and abs(new_col - col) == 1:
                                    if 0 <= jump_row < 8 and 0 <= jump_col < 8 and self.board[jump_row][jump_col] == 0:
                                        moves.append(CheckersMove(position, (jump_row, jump_col)))
                                break

                            elif self.board[new_row][new_col] == 0:
                                moves.append(CheckersMove(position, (new_row, new_col)))

                        else:
                            break

        return moves

    def calculate_reward(self, move: CheckersMove, piece_captured: bool) -> float:
        reward = 0.0
        if piece_captured:
            reward += 1
            
        return reward

    def move(self, move: CheckersMove) -> tuple[bool, int]:
        start_row, start_col = move.start
        end_row, end_col = move.end

        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = 0
        
        pos = tuple()
        piece_captured = False
        if abs(end_row - start_row) == 2:
            jumped_row = (start_row + end_row) // 2
            jumped_col = (start_col + end_col) // 2
            if self.board[jumped_row][jumped_col] != 0:
                print("Mammt: ",self.board[jumped_row][jumped_col])
                self.board[jumped_row][jumped_col] = 0
                piece_captured = True
                pos = (jumped_row, jumped_col)
                # print(f"Eaten piece at {pos}")
        
        if end_row == 0 and self.board[end_row][end_col] == -1:
            self.board[end_row][end_col] = -2 
        elif end_row == 7 and self.board[end_row][end_col] == 1:
            self.board[end_row][end_col] = 2 

        reward = self.calculate_reward(move, piece_captured)

        self.__turn = -self.__turn

        return piece_captured, reward, pos


    @property
    def winner(self) -> int:

        player_1_pieces = sum(cell == 1 or cell == 2 for row in self.board for cell in row)
        player_2_pieces = sum(cell == -1 or cell == -2 for row in self.board for cell in row)

        if player_1_pieces == 0:
            return -1
        elif player_2_pieces == 0:
            return 1
        else:
            return 0
