import re
import itertools
from textwrap import wrap
N = 9 
WHITE, BLACK, EMPTY = 'o', '*', '.'
EMPTY_BOARD = EMPTY*(N**2) 

class Game():
    '''go.Game: a class to represent a go game. The board is represented as a length N^2 string
    using "squashed coordinates" 0,1,...,N^2-1.
    optional parameters: 
        board: str -- initialize a board position
        ko: int -- the position of the current ko
        turn: int -- the current turn number
        moves: list -- the list of moves played
        sgf: str -- path to an sgf to initialize from
        '''
    def __init__(self, board = EMPTY_BOARD, ko = None, turn = 0, moves = [], sgf = None):
        self.turn = turn
        self.ko = ko
        self.board= board
        if sgf:
            self.moves = self.get_moves(sgf)
        else:
            self.moves = moves
        self.enc = {BLACK: 1, WHITE: -1, EMPTY: 0}


    def __str__(self):
        return "\t  " +' '.join([str(i) for i in range(N)]) +"\n" \
            + '\n'.join(['\t'+str(i)+' '+ ' '.join( self.board[N*i:N*(i+1)]) for i in range(N)])

    def __len__(self):
        return len(self.moves)

    def get_board(self):
        return [self.enc[s] for s in self.board]

    def play_move(self, sq_c = None, testing = False):
        '''play move from self.moves. If a coordinate is given that is played instead.'''
        if sq_c == None:
            if self.turn >= len(self):
                print("No more moves to play.")
                return
            sq_c = self.moves[self.turn]
        
        if sq_c == self.ko:
            raise IllegalMove(f"\n{self}\n Move at {sq_c} illegally retakes ko.")
        if self.board[sq_c] != EMPTY:
            raise IllegalMove(f"\n{self}\n There is already a stone at {sq_c}")
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
            raise IllegalMove(" \n{self}\n Move at {sq_c} is suicide.")

        if len(opp_captured) == 1 and possible_ko_color == opp_color:
            new_ko = opp_captured[0] 
        else:
            new_ko = None

        if not testing: 
            self.board = new_board
            self.ko = new_ko
            self.turn += 1 

    def is_legal(self, sq_c):
        try:
            self.play_move(sq_c, testing = True)
            return True
        except IllegalMove:
            return False

    def score(self, komi = 5.5):
        '''Calculated using Chinese rules'''
        board = self.board
        while EMPTY in board:
            empty = board.index(EMPTY)
            empties, borders = flood_fill(board, empty)
            bd_list = [board[sq_b] for sq_b in borders]
            if bd_list.count(BLACK) > bd_list.count(WHITE):
                border_color = BLACK
            else:
                border_color = WHITE
            board = bulk_place_stones(border_color, board, borders)
            board = bulk_place_stones(border_color, board, empties)
        return board.count(BLACK), board.count(WHITE) + komi

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

    @staticmethod
    def get_moves(sgf):
        with open(sgf, 'r') as f:
            match = re.findall(r"[BW]\[(\w*)\]", f.read())
        mvs = []
        for mv in match:
            if len(mv)!= 2:
                break
            else: 
                mvs.append(9*(ord(mv[0])-97) + ord(mv[1])-97 )
        return mvs

#squash converts coordinate pair 0 <= x,y < N  to single integer 0 <= n < N^2
def squash(c):
    return N * c[0] + c[1]

def unsquash(sq_c):
    if type(sq_c) == list:
        return [divmod(sq_b, N) for sq_b in sq_c]
    else:
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

def get_stone_lib(board, sq_c):
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

