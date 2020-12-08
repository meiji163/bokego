'''
This module contains a Game class that represents a game of go (baduk/weiqi) and several utilities.

Coordinates:
    For an an NxN board 
        *Coordinate pair is a tuple c = (x,y) where 0 <= x,y < N
        *Squashed coordinate is an int sq_c where 0 <= sq_c < N**2, 
            obtained from coordinate pair by (x,y) --> N*x + y
        *Alpha-num coordinate is a string with a letter followed by a number.
            Follows western convention: letter A - T excluding I, number 1 - N.
    Methods in this module use squashed coordinates. 
    Use the squash and unsquash functions to convert between coordinate types.
'''
import re
import os
from random import getrandbits
from datetime import date 
from shutil import which
from subprocess import Popen, PIPE
from tempfile import gettempdir

N = 9 #board sizes: 9, 13, 19 
WHITE, BLACK, EMPTY, FLOWER = 'O', 'X', '.', '+'
EMPTY_BOARD = EMPTY*(N**2) 

PASS = -1
RESIGN = -2

FLOWERS9 = (20,60,24,56,40)
FLOWERS13 = (42, 126, 48, 120, 84) 
FLOWERS19 = (60, 300, 72, 288, 66, 174, 180, 186, 294 )

class Game():
    '''A class to represent a go game on a NxN board. 
    One instance is intended to play through a game once (no backtracking).

    To record moves as they are played, initialize with moves = []

    args:
        sgf (str): path to sgf to load moves from
        boardi (str): length N**2 string of WHITE, BLACK, and EMPTY representing board
        ko (int): the squashed coord of the current ko, None if no ko
        turn (int): turn number (starting from 0) 
        moves (list): list of squashed coords of moves played 
        komi (float): the komi (default 5.5 for 9x9)
    '''

    _hash_table = [[ getrandbits(64) for _ in range(N*N)] for _ in range(3)]
    _flip = getrandbits(64) 

    def __init__(self, board = EMPTY_BOARD, 
                ko = None, last_move = None, 
                turn = 0, moves = None, 
                komi = 5.5, sgf = None):
        self.sgf = sgf
        if self.sgf:
            self.moves = get_moves(sgf) 
        else:
            self.moves = moves 
        self.last_move = last_move 
        self.turn = turn
        self.ko = ko
        self.board= board
        self.komi = komi
        self._hash = None
        self._libs = None

    def __str__(self):
        out = self.board
        #mark flower points
        f = tuple()
        if N == 9:
            f = FLOWERS9            
        elif N == 13:
            f = FLOWERS13
        elif N == 19:
            f = FLOWERS19
        for i in f:
            if out[i] == EMPTY:
                out = place_stone(FLOWER, out, i)
        def add_space(i): return '  ' if i<9 else ' '
        return "\t   " +' '.join(["ABCDEFGHJKLMNOPQRST"[i] for i in range(N)]) +"\n" \
                + '\n'.join(['\t'+str(i + 1)+ add_space(i) 
                + ' '.join( out[N*i:N*(i+1)]) for i in range(N)])
    
    def __hash__(self):
        if self._hash is None:
            self._hash = self.zobrist_hash()
        return self._hash

    def __len__(self):
        if self.moves:
            return len(self.moves)
        return 0

    def __repr__(self):
        return repr( (self.board, self.ko, self.last_move) )
        
    def to_numpy(self):
        '''Convert board to (N,N) numpy array'''
        try:
            import numpy as np
        except ImportError:
            print("Numpy not found")
            return
        encode = {BLACK: 1, WHITE: -1, EMPTY: 0}
        return np.array([encode[sq_c] for sq_c in self.board], dtype = np.int8).reshape(N,N)

    def play_pass(self):
        if self._hash != None:
            if self.ko != None:
                self._hash ^= Game._hash_table[self.turn%2][self.ko]
            self._hash ^= Game._flip
            #if self.last_move != PASS and self.last_move != None:
            #    self._hash ^= Game._hash_table[self.turn%2][self.last_move]

        if self.moves != None:
            self.moves.append(PASS)
        self.turn += 1
        self.ko = None
        self.last_move = PASS 

    def play_move(self, sq_c = None, testing = False):
        '''Play move at coord for current player. 
        If no coordinate is given a move is played from moves list.
        optional: 
            testing: for testing legal moves
        ''' 
        if sq_c is None:
            if self.turn >= len(self):
                print("No moves to play.")
                return
            sq_c = self.moves[self.turn]
        elif sq_c == PASS:
            self.play_pass()
            return
        elif sq_c == self.ko:
            raise IllegalMove(self, rule_type = "ko", sq_c = sq_c)
        elif self.board[sq_c] != EMPTY:
            raise IllegalMove(self, rule_type = "not_empty", sq_c = sq_c)

        color = (WHITE if self.turn%2 ==1 else BLACK) 
        opp_color = (BLACK if color == WHITE else WHITE) 

        #check for ko
        possible_ko_color = possible_ko(self.board, sq_c)
        new_board = place_stone(color, self.board, sq_c)
        new_board, opp_captured = get_caps(new_board, sq_c, color)
        if len(opp_captured) == 1 and possible_ko_color == opp_color:
            new_ko = opp_captured[0] 
        else:
            new_ko = None

        #check for suicide
        new_board, captured = maybe_capture_stones(new_board, sq_c)
        if captured:
            raise IllegalMove(self, rule_type = "suicide", sq_c = sq_c)
        if testing: return

        self.get_liberties()
        if self.moves != None and self.sgf is None:
            self.moves.append(sq_c)

        if self._hash != None:
            #update the Zobrist hash
            self._hash ^= Game._hash_table[self.turn%2][sq_c]
            if self.ko != None:
                self._hash ^= Game._hash_table[2][self.ko]
            if new_ko != None:
                self._hash ^= Game._hash_table[2][new_ko]
            if opp_captured:
                for sq_b in opp_captured:
                    self._hash ^= Game._hash_table[(self.turn + 1)%2][sq_b]
            self._hash ^= Game._flip
            #if self.last_move != PASS and self.last_move != None:
            #    self._hash ^= Game._hash_table[(self.turn + 1)%2][self.last_move]
            #self._hash ^= Game._hash_table[3][sq_c]

        self.board = new_board
        self.last_move = sq_c
        self.ko = new_ko
        self.turn += 1 

    def is_legal(self, sq_c):
        if sq_c == PASS:
            return True
        if self.board[sq_c] != EMPTY:
            return False
        empties = 0
        for sq_b in NEIGHBORS[sq_c]:
            if empties > 1:
                return True
            if self.board[sq_b] == EMPTY:
                empties += 1
        #last check: suicide or ko
        try:
            self.play_move(sq_c, testing = True)
            return True
        except IllegalMove:
            return False

    def score(self):
        '''Return black's score minus white's score calculated using Tromp-Taylor rules 
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
        '''Get list of liberties for all coords. EMPTY coords have 0.'''
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
        #liberties from previous state exist and not yet updated
        elif self.last_move != PASS and self.last_move != None\
                and self._libs[self.last_move] == 0: 
            seen = set()
            color = board[self.last_move]
            for sq_b in NEIGHBORS[self.last_move] + [self.last_move]:
                if board[sq_b] != EMPTY and sq_b not in seen: 
                    num_libs, stones = get_stone_lib(board, sq_b, return_grp = True)
                    for sq_s in stones:
                        self._libs[sq_s] = num_libs
                    seen |= stones 
        return list(self._libs)
    
    def get_legal_moves(self):
        '''Return a list of all legal moves besides PASS'''
        legal = set()
        board = self.board
        #find empties with flood fill
        while EMPTY in board:
            sq_c = board.index(EMPTY)
            empties, _  = flood_fill(board, sq_c)
            board = bulk_place_stones('?', board, empties)
            if len(empties) > 1:
                legal |= empties 
            else:
                #sq_c is possibly illegal
                if self.is_legal(sq_c):
                    legal.add(sq_c) 
        return list(legal)
    
    def zobrist_hash(self): 
        ''' Compute the Zobrist hash of the current game state defined by
        the board, ko state, and turn'''
        out = 0 
        for sq_c in range(N*N):
            if self.board[sq_c] == BLACK:
                out ^= Game._hash_table[0][sq_c]
            elif self.board[sq_c] == WHITE:
                out ^= Game._hash_table[1][sq_c]
        if self.ko != None:
            out ^= Game._hash_table[2][ko]
        if self.turn%2 == 1:
            out ^= Game._flip
        #if last_move != None and last_move != -1:
        #    out ^= hash_table[3][last_move]
        return out 

class IllegalMove(Exception):
    '''A class to handle illegal moves.
    args:
        game: go.Game object
    kwargs:
        rule_type: type of illegal move - ko, suicide, not_empty, off_board
        c: tuple coordinate of illegal move
        sq_c: squashed coordinate of illegal move
        alph_c: alphanumeric coordinate of illegal move
        show_legal: if True, print all legal moves (default False)
    ''' 
    def __init__(self, game, **kwargs):
        super(IllegalMove, self).__init__()
        self.game = game
        self.rule_type = kwargs.get('rule_type')
        self.move = None
        self.show_legal = kwargs.get('show_legal', False)

        if 'sq_c' in kwargs:
            self.move = unsquash(kwargs['sq_c'])
        elif 'alph_c' in kwargs:
            self.move = kwargs['alph_c'] 
        elif 'c' in kwargs:
            c = kwargs['c']
            self.move = unsquash(9*c[0]+c[1])

    def __str__(self):
        if self.rule_type == "ko":
            msg = f"\n{self.game}\n Move at {self.move} illegally retakes ko."
        elif self.rule_type == "suicide":
            msg = f"\n{self.game}\n Move at {self.move} is suicide."
        elif self.rule_type == "off_board":
            msg = f"Move is not on board"
        elif self.rule_type == "not_empty":
            msg = f"\n{self.game}\n There is already a stone at {self.move}"
        else:
            msg = ''
        if self.show_legal:
            msg +=  "\nlegal moves: " \
                    + ', '.join(unsquash(self.game.get_legal_moves()))
        return msg

#Helper functions 
def squash(c):
    '''Converts coord pair (x,y) or an alpha-numeric coord to a squashed coord.
    If list of coordinates are passed, converts each element.'''
    if isinstance(c, list):
        return list(map(squash, c))
    elif isinstance(c, str):
        c = c.upper()
        if c == "PASS":
            return go.PASS

        m = re.match(r"([A-T])(\d+)", c)
        if m is None:
            raise ValueError
        #Letters skip I
        let, num = m[1], m[2]
        if let < 'J':
            y = ord(let) - 65
        elif let == 'J':
            y = 8
        else:
            y = ord(let) - 66
        return N*(int(num)-1) + y
    else:
        return N*c[0] + c[1]

def unsquash(sq_c, alph = True):
    '''Converts squashed coord to alpha-num coord.
    optional:
        alph: if False, unsquash to a coord pair instead'''
    if isinstance(sq_c, list): 
        return list(map(lambda x: unsquash(x, alph), sq_c))
    elif sq_c == PASS:
        return "PASS"
    else:
        c = divmod(sq_c, N)
        if alph:
            if c[1] < 8:
                letr = chr(c[1] + 65)
            elif c[1] == 8:
                letr = 'J'
            else:
                letr = chr(c[1] +66)
            return letr + str(c[0] +1)
        return c

def is_on_board(c):
    return c[0] % N == c[0] and c[1] % N == c[1]

NEIGHBORS = [squash(list( filter(is_on_board, [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]))) \
                for x in range(N) for y in range(N)] 
DIAGONALS = [squash(list( filter(is_on_board, [(x+1,y+1), (x+1, y-1), (x-1, y-1), (x-1, y-1)]))) \
                for x in range(N) for y in range(N)] 

def flood_fill(board, sq_c):
    '''Floodfill board starting from coord. Returns set containing
    the filled coords and set containing the border coords'''
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
    '''Check if coord is (one point) eye and return the color of the eye'''
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

def get_stones(board):
    black = set()
    white = set()
    for sq_c in range(go.N**2):
        if board[sq_c] == BLACK:
            black.add(sq_c)
        elif board[sq_c] == WHITE:
            white.add(sq_c)
    return black, white

#sgf utilities

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

def gnu_score(game: Game):
    '''Scores the game using gnugo opened in a subprocess.
    Returns 1 if black won, -1 if white won'''
    gnugo_path = which("gnugo")
    if gnugo_path is None:
        return
    temp = os.path.join(gettempdir(), f"{os.getpid()}.sgf")
    write_board_sgf(game, temp) 
    p =Popen([gnugo_path , "--komi", "5.5", "--mode", "gtp", "--chinese-rules", "-l", temp], \
                    stdin = PIPE, stdout = PIPE)
    p.stdin.write("final_score\n".encode('utf-8'))
    p.stdin.flush()
    rec = p.stdout.readline().decode('utf-8').strip('\n')
    p.communicate("quit\n".encode('utf-8'))
    os.remove(temp)

    res = re.search("[BW]\+.+",rec)
    if res:
        return 1 if 'B' in res[0] else -1

def write_sgf(moves, out_path, **kwargs): 
    '''
    Creates sgf for move sequence. Writes to file and returns sgf string
    args:
        moves (list): move sequence in squashed coords
        out_path (str): the file to write to
    kwargs:
        komi (float): the komi (default 5.5)
        B (str): name of black player
        W (str): name of white player
        result (str): result of game (e.g. "B+2.5")
        handicap (int): number of handicap stones (default 0)
    '''
    B = kwargs.get('B', '')
    W = kwargs.get('W', '')
    handi = kwargs.get("handicap",0)
    komi = kwargs.get("komi", 5.5)
    result = kwargs.get('result', '') 

    today = date.today()
    out = f"(;GM[1]HA[{handi}]RU[Chinese]DT[{today}]"
    if B and W:
        out += f"PB[{B}]PW[{W}]"
    if result:
        out += f"RE[{result}]"
    out += f"SZ[{N}]KM[{komi}]\n"
    turn = "B"
    for mv in moves:
        if mv == PASS:
            out += f";{turn}[]\n"
        else:
            x, y = chr(mv//9 + 97), chr(mv%9 +97)
            out += f";{turn}[{x}{y}]\n"
        turn = "W" if turn == "B" else "B" 
    out += ")" 
    with open(out_path, 'w') as f:
        f.write(out)
    return out

def write_board_sgf(game, out_path): 
    '''write board to sgf (use when move sequence not available)'''
    out = f"(;GM[1]RU[Chinese]HA[0]SZ[{N}]KM[{game.komi}]\n"
    W = "AW"
    B = "AB"
    black, white = get_stones(game.board)
    for i in range(N): 
        mv = game.board[i]
        if mv == 'X':
            x, y = chr(i//9 + 97), chr(i%9 +97)
            B += f"[{x}{y}]" 
        elif mv == 'O':
            x, y = chr(i//9 + 97), chr(i%9 +97)
            W += f"[{x}{y}]" 
    turn = 'B' if game.turn%2 == 0 else 'W' 
    out += B + '\n' + W + f"PL[{turn}])"
    with open(out_path, 'w') as f:
        f.write(out)
