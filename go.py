import itertools
from textwrap import wrap
N = 9 
WHITE, BLACK, EMPTY = 'O', 'X', '.'
EMPTY_BOARD = EMPTY*(N**2) 

def swap_colors(color):
    if color == BLACK:
        return WHITE
    elif color == WHITE:
        return BLACK
    else:
        return color

def turn_color(turn):
    if turn%2 == 0:
        return BLACK
    return WHITE

def squash(c):
    '''convert (x,y) coordinate to squashed coordinate in {0,..., N**2-1} '''
    return N * c[0] + c[1]

def unsquash(sq_c):
    '''convert squashed coordinate sq_c to coordinate c'''
    return divmod(sq_c, N)

def is_on_board(c):
    return c[0] % N == c[0] and c[1] % N == c[1]

def get_valid_neighbors(sq_c):
    x, y = unsquash(sq_c)
    possible_neighbors = ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
    return [squash(n) for n in possible_neighbors if is_on_board(n)]

NEIGHBORS = [get_valid_neighbors(sq_c) for sq_c in range(N*N)]

def find_reached(board, sq_c):
    color = board[sq_c]
    chain = set([sq_c])
    reached = set()
    frontier = [sq_c]
    while frontier:
        current_sq_c = frontier.pop()
        chain.add(current_sq_c)
        for sq_n in NEIGHBORS[current_sq_c]:
            if board[sq_n] == color and not sq_n in chain:
                frontier.append(sq_n)
            elif board[sq_n] != color:
                reached.add(sq_n)
    return chain, reached

class IllegalMove(Exception): pass

def place_stone(color, board, sq_c):
    return board[:sq_c] + color + board[sq_c+1:]

def bulk_place_stones(color, board, stones):
    byteboard = bytearray(board, encoding='ascii') 
    color = ord(color)
    for fstone in stones:
        byteboard[fstone] = color
    return byteboard.decode('ascii') 

def maybe_capture_stones(board, sq_c):
    chain, reached = find_reached(board, sq_c)
    if not any(board[sq_r] == EMPTY for sq_r in reached):
        board = bulk_place_stones(EMPTY, board, chain)
        return board, chain
    else:
        return board, []

def play_move_incomplete(board, sq_c, color):
    if board[sq_c] != EMPTY:
        raise IllegalMove
    board = place_stone(color, board, sq_c)

    opp_color = swap_colors(color)
    opp_stones = []
    my_stones = []
    for sq_n in NEIGHBORS[sq_c]:
        if board[sq_n] == color:
            my_stones.append(sq_n)
        elif board[sq_n] == opp_color:
            opp_stones.append(sq_n)

    for sq_s in opp_stones:
        board, _ = maybe_capture_stones(board, sq_s)

    for sq_s in my_stones:
        board, _ = maybe_capture_stones(board, sq_s)

    return board

def possible_ko(board, sq_c):
    'Check if sq_c is surrounded on all sides by 1 color, and return that color'
    if board[sq_c] != EMPTY: return None
    neighbor_colors = { board[ sq_n ] for sq_n in NEIGHBORS[sq_c]}
    if len(neighbor_colors) == 1 and not EMPTY in neighbor_colors:
        return list(neighbor_colors)[0]
    else:
        return None

class Game():
    def __init__(self, board = EMPTY_BOARD, ko = None, turn = 0, moves = []):
        self.turn = turn
        self.ko = ko
        self.board= board
        self.moves = moves

    def get_board(self):
        '''return board as array'''
        enc = {BLACK: 255, WHITE: -255, EMPTY: 0}
        return [enc[s] for s in self.board]

    def __str__(self):
        return '\n'.join(wrap(self.board, N))
    
    def play_move(self, c = None):
        '''play move from self.moves. If a coordinate is given that is played instead.'''
        color = turn_color(self.turn)
        if c != None:
            try:
                self.moves[self.turn] = c
            except IndexError:
                self.moves.append(c)
        else:
            if self.turn > len(self.moves):
                print("No more moves to play.")
            c = self.moves[self.turn]

        sq_c = squash(c)
        if sq_c == self.ko:
            raise IllegalMove(f"{self}\n Move at %{c} illegally retakes ko.") 

        if self.board[sq_c] != EMPTY:
            raise IllegalMove(f"{self}\n Stone exists at {c}.")

        possible_ko_color = possible_ko(self.board, sq_c)
        new_board = place_stone(color, self.board, sq_c)

        opp_color = swap_colors(color)
        opp_stones = []
        my_stones = []
        for sq_n in NEIGHBORS[sq_c]:
            if new_board[sq_n] == color:
                my_stones.append(sq_n)
            elif new_board[sq_n] == opp_color:
                opp_stones.append(sq_n)

        opp_captured = []
        for sq_s in opp_stones:
            new_board, captured = maybe_capture_stones(new_board, sq_s)
            opp_captured += captured

        # Check for suicide
        new_board, captured = maybe_capture_stones(new_board, sq_c)
        if captured:
            raise IllegalMove("\n{self}\n Move at {c} is suicide.")

        if len(opp_captured) == 1 and possible_ko_color == opp_color:
            new_ko = opp_captured[0] 
        else:
            new_ko = None

        self.board = new_board
        self.ko = new_ko
        self.turn += 1 

    def score(self):
        board = self.board
        while EMPTY in board:
            empty = board.index(EMPTY)
            empties, borders = find_reached(board, empty)
            possible_border_color = board[list(borders)[0]]
            if all(board[sq_b] == possible_border_color for sq_b in borders):
                board = bulk_place_stones(possible_border_color, board, empties)
            else:
                board = bulk_place_stones('?', board, empties)
        return board.count(BLACK) - board.count(WHITE)

    def get_liberties(self):
        board = self.board
        liberties = bytearray(N*N)
        for color in (WHITE, BLACK):
            while color in board:
                sq_c = board.index(color)
                stones, borders = find_reached(board, sq_c)
                num_libs = len([fb for fb in borders if board[fb] == EMPTY])
                for fs in stones:
                    liberties[fs] = num_libs
                board = bulk_place_stones('?', board, stones)
        return list(liberties)

