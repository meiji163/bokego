import itertools
from textwrap import wrap
N = 9 
WHITE, BLACK, EMPTY = 'O', 'X', '.'
EMPTY_BOARD = EMPTY*(N**2) 

class Game():
    def __init__(self, board = EMPTY_BOARD, ko = None, turn = 0, moves = []):
        self.turn = turn
        self.ko = ko
        self.board= board
        self.moves = moves

    def get_board(self):
        '''return board as array'''
        enc = {BLACK: 1, WHITE: -1, EMPTY: 0}
        return [enc[s] for s in self.board]

    def __str__(self):
        return '\n'.join(wrap(self.board, N))

    def play_move(self, c = None, testing = False):
        '''play move from self.moves. If a coordinate is given that is played instead.'''
        if c == None:
            if self.turn >= len(self.moves):
                print("No more moves to play.")
                return
            c = self.moves[self.turn]
        sq_c = squash(c)
        
        if sq_c == self.ko:
            raise IllegalMove(f"\n{self}\n Move at {c} illegally retakes ko.")
        if self.board[sq_c] != EMPTY:
            raise IllegalMove(f"\n{self}\n There is already a stone at {c}")
        color = (WHITE if self.turn%2 ==1 else BLACK) 
        possible_ko_color = possible_ko(self.board, sq_c)
        new_board = place_stone(color, self.board, sq_c)

        opp_color = (BLACK if color == WHITE else WHITE) 
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
            raise IllegalMove(" \n{self}\n Move at {c} is suicide.")

        if len(opp_captured) == 1 and possible_ko_color == opp_color:
            new_ko = opp_captured[0] 
        else:
            new_ko = None

        if not testing: 
            self.board = new_board
            self.ko = new_ko
            self.turn += 1 

    def is_legal(self, c):
        try:
            self.play_move(c, testing = True)
            return True
        except IllegalMove:
            return False

    def score(self):
        '''returns Bs score minus Ws score'''
        board = self.board
        while EMPTY in board:
            empty = board.index(EMPTY)
            empties, borders = flood_fill(board, empty)
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
                stones, borders = flood_fill(board, sq_c)
                num_libs = len([sq_b for sq_b in borders if board[sq_b] == EMPTY])
                for sq_s in stones:
                    liberties[sq_s] = num_libs
                board = bulk_place_stones('?', board, stones)
        return list(liberties)

#squash converts coordinate pair 0 <= x,y < N  to single integer 0 <= n < N^2
def squash(c):
    return N * c[0] + c[1]

def unsquash(sq_c):
    return divmod(sq_c, N)

def is_on_board(c):
    return c[0] % N == c[0] and c[1] % N == c[1]

def get_valid_neighbors(sq_c):
    x, y = unsquash(sq_c)
    possible_neighbors = ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
    return [squash(n) for n in possible_neighbors if is_on_board(n)]

NEIGHBORS = [get_valid_neighbors(sq_c) for sq_c in range(N*N)]

def flood_fill(board, sq_c):
    '''Flood fill to find the connected component containing sq_c and its boundary'''
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

def get_stone_lib(board, c):
    sq_c = squash(c)
    stones, borders = flood_fill(board, sq_c)
    return len([sq_b for sq_b in borders if board[sq_b] == EMPTY])

class IllegalMove(Exception): pass

#Helper functions
def place_stone(color, board, sq_c):
    return board[:sq_c] + color + board[sq_c+1:]

def bulk_place_stones(color, board, stones):
    byteboard = bytearray(board, encoding='ascii') 
    color = ord(color)
    for fstone in stones:
        byteboard[fstone] = color
    return byteboard.decode('ascii') 

def maybe_capture_stones(board, sq_c):
    chain, reached = flood_fill(board, sq_c)
    if not any(board[sq_r] == EMPTY for sq_r in reached):
        board = bulk_place_stones(EMPTY, board, chain)
        return board, chain
    else:
        return board, []

def play_move_incomplete(board, sq_c, color):
    if board[sq_c] != EMPTY:
        raise IllegalMove
    board = place_stone(color, board, sq_c)

    opp_color = WHITE if color == BLACK else WHITE
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

