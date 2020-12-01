import re
import os
import random
import itertools

N = 9 #board size
WHITE, BLACK, EMPTY = 'O', 'X', '.'
EMPTY_BOARD = EMPTY*(N**2) 
ENC = {BLACK: 1, WHITE: -1, EMPTY: 0}
PASS = -1

class IllegalMove(Exception):
    def __init__(self, **kwargs):
        super().__init__()
        self.game = kwargs.get('game')
        self.rule_type = kwargs.get('rule_type')
        self.move = None
        if 'sq_c' in kwargs:
            self.move = unsquash(kwargs['sq_c'], alph = True)
        elif 'alph_c' in kwargs:
            self.move = kwargs['alph_c'] 
        elif 'c' in kwargs:
            c = kwargs['c']
            self.move = unsquash(9*c[0]+c[1], alph = True)

    def __str__(self):
        if rule_type == "ko":
            msg = f"\n{game}\n Move at {self.move} illegally retakes ko."
        elif rule_type == "suicide":
            msg = f"\n{game}\n Move at {self.move} is suicide."
        elif rule_type == "off_board":
            msg = f"Move is not on board"
        elif rule_type == "not_empty":
            msg = f"\n{game}\n There is already a stone at {self.move}"
        else:
            msg = ''
        return msg

class Game():
    '''go.Game: a class to represent a go game.
    Board coordinates are `squashed` (0 -- N**2).
    
    attributes:
        board: length N**2 string of WHITE, BLACK, and EMPTY representing board
        ko: the coordinate of the current ko, None if no ko
        turn: turn number (starting from 0) 
        moves: list of moves played 
        sgf: path to an sgf file to initialize moves from
        hash: stored 64 bit Zobrist hash of game state
        '''
    hash_table = [[ random.getrandbits(64) for _ in range(N*N)] for _ in range(4)]

    def __init__(self, board = EMPTY_BOARD,
                ko = None, last_move = None, 
                turn = 0, moves = None, 
                komi = 5.5, sgf = None):
        self.turn = turn
        self.ko = ko
        self.board= board
        self.komi = komi
        self.last_move = last_move
        self.hash = None
        if sgf:
            self.moves = get_moves(sgf)
        else:
            self.moves = moves
        self._libs = None

    def __str__(self):
        out = self.board
        if N == 9: #mark flower points
            for i in [20,24,40,56,60]:
                if out[i] == EMPTY:
                    out = place_stone('+', out, i)
        return "\t  " +' '.join(["ABCDEFGHJKLMNOPQRST"[i] for i in range(N)]) +"\n" \
                + '\n'.join(['\t'+str(i + 1)+' '
                + ' '.join( out[N*i:N*(i+1)]) for i in range(N)])
    
    def __hash__(self):
        if self.hash == None:
            self.hash = zobrist_hash(self.board, self.ko, self.last_move, Game.hash_table)
        return self.hash

    def __len__(self):
        if self.moves:
            return len(self.moves)
        return 0

    def __repr__(self):
        return repr( (self.board, self.ko, self.last_move, self.turn) )
        
    def to_numpy(self):
        '''Convert board to (N,N) numpy array'''
        try:
            import numpy as np
        except ImportError:
            print("Numpy not found")
            return
        return np.array([ENC[sq_c] for sq_c in self.board]).reshape(N,N)

    def play_pass(self):
        if self.hash != None:
            if self.ko != None:
                self.hash ^= Game.hash_table[turn%2][self.ko]
            if self.last_move != PASS and self.last_move != None:
                self.hash ^= Game.hash_table[turn%2][self.last_move]
        if not self.moves:
            self.moves = [PASS]
        else:
            self.moves.append(PASS)
        self.turn += 1
        self.ko = None
        self.last_move = PASS 

    def play_move(self, sq_c = None, testing = False):
        '''Play move at sq_c. If no coordinate is given a move is played from self.moves.
        optional: 
            testing: stop game state from being modified (default False)''' 
        if sq_c is None:
            if self.turn >= len(self):
                print("No moves to play.")
                return
            sq_c = self.moves[self.turn]
        elif sq_c == PASS:
            self.play_pass()
            return
        elif sq_c == self.ko:
            raise IllegalMove(game = self, rule_type = "ko", sq_c = sq_c)
        elif self.board[sq_c] != EMPTY:
            raise IllegalMove(game = self, rule_type = "not_empty", sq_c = sq_c)
        color = (WHITE if self.turn%2 ==1 else BLACK) 
        opp_color = (BLACK if color == WHITE else WHITE) 
        possible_ko_color = possible_ko(self.board, sq_c)
        new_board = place_stone(color, self.board, sq_c)
        new_board, opp_captured = get_caps(new_board, sq_c, color)
        if len(opp_captured) == 1 and possible_ko_color == opp_color:
            new_ko = opp_captured[0] 
        else:
            new_ko = None
        # Check for suicide
        new_board, captured = maybe_capture_stones(new_board, sq_c)
        if captured:
            raise IllegalMove(game = self, rule_type = "suicide", sq_c = sq_c)
        if testing: return

        if not self.moves:
            self.moves = [sq_c]
        else:
            self.moves.append(sq_c)

        if self.hash:
            #update the Zobrist hash
            self.hash ^= Game.hash_table[self.turn%2][sq_c]
            if self.ko != None:
                self.hash ^= Game.hash_table[2][self.ko]
            if new_ko != None:
                self.hash ^= Game.hash_table[2][new_ko]
            if opp_captured:
                for sq_b in opp_captured:
                    self.hash ^= Game.hash_table[(self.turn + 1)%2][sq_b]
            if self.last_move != PASS and self.last_move != None:
                self.hash ^= Game.hash_table[(self.turn + 1)%2][self.last_move]
            self.hash ^= Game.hash_table[3][sq_c]

        self.board = new_board
        self.last_move = sq_c
        self.ko = new_ko
        self.turn += 1 

    def is_legal(self, sq_c):
        try:
            self.play_move(sq_c, testing = True)
            return True
        except IllegalMove:
            return False

    def score(self):
        '''Calculated using Tromp-Taylor rules 
        (only accurate assuming dead groups are captured)'''
        board = self.board
        while EMPTY in board:
            empty = board.index(EMPTY)
            empties, borders = flood_fill(board, empty)
            bd_list = [board[sq_b] for sq_b in borders]
            if BLACK in bd_list and not WHITE in bd_list: 
                border_color = BLACK
            elif WHITE in bd_list and not BLACK in bd_list:
                border_color = WHITE
            else:
                border_color = '?'
            board = bulk_place_stones(border_color, board, borders)
            board = bulk_place_stones(border_color, board, empties)
        return board.count(BLACK) - (board.count(WHITE) + self.komi)

    def get_liberties(self):
        board = self.board
        if self._libs is None:
            self._libs = bytearray(N*N)
            for color in (WHITE, BLACK):
                while color in board:
                    sq_c = board.index(color)
                    num_libs, stones = get_stone_lib(board, sq_c, return_grp = True)
                    for sq_s in stones:
                        self._libs[sq_s] = num_libs
                    board = bulk_place_stones('?', board, stones)

        elif self.last_move != PASS and self._libs[self.last_move] == 0: 
        #liberties from previous state exist and not yet updated
            seen = set()
            color = board[self.last_move]
            for sq_b in NEIGHBORS[self.last_move]:
                if not board[sq_b] == EMPTY and sq_b not in seen: 
                    num_libs, stones = get_stone_lib(board, sq_b, return_grp = True)
                    for sq_s in stones:
                        self._libs[sq_s] = num_libs
                    seen |= stones 
            if len(seen) == 0:
                self._libs[sq_s] = 1    
        return list(self._libs)

#Helper functions 

def squash(c, alph = False):
    '''Converts coordinate pair to single integer 0 -- N^2.
    If a list of coordinates are passed, convert each element.
    optional:
        alph: squashes a letter-number coordinate '''
    if isinstance(c, list):
        return list(map(squash, c))
    if alph:
        #Letters skip I
        y = 8 if c[0] == 'J' else ord(c[0]) - 65 
        c = ( int(c[1]) - 1,  y)
    return N * c[0] + c[1]

def unsquash(sq_c, alph = False):
    if isinstance(sq_c, list): 
        return list(map(unsquash), sq_c) 
    else:
        c = divmod(sq_c, N)
        if alph:
            letr = 'J' if c[1] == 8 else chr(c[1] + 65)
            return letr + str(c[0] +1)
        return c

def is_on_board(c):
    return c[0] % N == c[0] and c[1] % N == c[1]

NEIGHBORS = [squash(list( filter(is_on_board, [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]))) \
                for x in range(N) for y in range(N)] 
DIAGONALS = [squash(list( filter(is_on_board, [(x+1,y+1), (x+1, y-1), (x-1, y-1), (x-1, y-1)]))) \
                for x in range(N) for y in range(N)] 

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

def get_stone_lib(board, sq_c, return_grp = False):
    '''get liberties of stone at coordinate sq_c 
    optional:
        return_grp: return the coordinates of stones in the group containing sq_c'''
    if board[sq_c] not in (WHITE, BLACK):
        return 0
    stones, borders = flood_fill(board, sq_c)
    num_libs = len([sq_b for sq_b in borders if board[sq_b] == EMPTY])
    if return_grp:
        return num_libs, stones
    return num_libs

def get_caps(board, sq_c, color):
        opp_color = BLACK if color == WHITE else WHITE
        opp_stones = []
        my_stones = []
        for sq_n in NEIGHBORS[sq_c]:
            if board[sq_n] == color:
                my_stones.append(sq_n)
            elif board[sq_n] == opp_color:
                opp_stones.append(sq_n)
        opp_captured = []
        for sq_s in opp_stones:
            new_board, captured = maybe_capture_stones(board, sq_s)
            opp_captured += captured
        new_board = bulk_place_stones(EMPTY, board, opp_captured)
        return new_board, opp_captured

def place_stone(color, board, sq_c):
    return board[:sq_c] + color + board[sq_c+1:]

def bulk_place_stones(color, board, stones):
    byteboard = bytearray(board, encoding='ascii') 
    color = ord(color)
    for fstone in stones:
        byteboard[fstone] = color
    return byteboard.decode('ascii') 

def maybe_capture_stones(board, sq_c):
    '''Check if group at sq_c is captured.
    Return board with stones captured, and list of captured coordinates.'''
    chain, reached = flood_fill(board, sq_c)
    if not any(board[sq_r] == EMPTY for sq_r in reached):
        board = bulk_place_stones(EMPTY, board, chain)
        return board, chain
    else:
        return board, []

def play_move_incomplete(board, sq_c, color):
    '''Place `color` stone at sq_c on board and capture stones.
    Doesn't check for ko or suicide.'''
    if board[sq_c] != EMPTY:
        raise IllegalMove(game = self, rule_type = "not_empty", sq_c = sq_c)
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
    '''Check if sq_c is surrounded by one color, and return that color'''
    if board[sq_c] != EMPTY: return None
    neighbor_colors = { board[sq_n] for sq_n in NEIGHBORS[sq_c]}
    if len(neighbor_colors) == 1 and not EMPTY in neighbor_colors:
        return list(neighbor_colors)[0]
    else:
        return None

def possible_eye(board, sq_c):
    '''Check if sq_c is (one point) eye and return the color of the eye'''
    color = possible_ko(board, sq_c)
    if color is None:
        return None
    diagonal_faults = 0
    diagonals = DIAGONALS[sq_c]
    if len(diagonals) < 4:
        diagonal_faults += 1
    for d in diagonals:
        if not board[d] in (color, EMPTY):
            diagonal_faults += 1
    if diagonal_faults > 1:
        return None
    else:
        return color

def zobrist_hash(board, ko, last_move, hash_table): 
    ''' Compute the Zobrist hash of the current game state defined by
    board, ko, and last_move using hash_table
    args:
        hash_table: a 4xN**2 array populated with random ints''' 
    out = 0 
    for sq_c in range(N*N):
        if board[sq_c] == BLACK:
            out ^= hash_table[0][sq_c]
        elif board[sq_c] == WHITE:
            out ^= hash_table[1][sq_c]
    if ko != None:
        out ^= hash_table[2][ko]
    if last_move != None and last_move != -1:
        out ^= hash_table[3][last_move]
    return out 

def get_moves(sgf):
    if not os.path.exists(sgf):
        raise IOError(f"Can't open sgf '{sgf}'")
    with open(sgf, 'r') as f:
        match = re.findall(r";[BW]\[(\w*)\]", f.read())
    mvs = []
    for mv in match:
        if len(mv) == 0:
            mvs.append(PASS)
        else: 
            mvs.append(9*(ord(mv[0])-97) + ord(mv[1])-97 )
    return mvs
