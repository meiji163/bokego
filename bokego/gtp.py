import bokego.go as go
from bokego.mcts import MCTS, Go_MCTS
import re
import os
import sys
import shlex
from itertools import cycle
from threading import Thread
from collections.abc import Generator
from collections import defaultdict
from subprocess import Popen, PIPE, TimeoutExpired
import multiprocessing as mp
from time import sleep
from timeit import default_timer

class GTP(MCTS):
    '''
    Wraps mcts.MCTS to interface via Go Text Protocol 2; 
    see `gnu.org/software/gnugo/gnugo_19.html` for usage.

    Also defines pass, resign, and time usage strategies. 

    kwargs:
        pondering (bool): if True, do rollouts while waiting for input (default True)
        time_lim (float): max number of seconds to spend per move (default 20.0)
        n_rollouts (int): max number of rollouts to do per move (no time limit)
        connection (Connection): connection for io. Uses stdin/stdout by default.
    Attributes:
        surrender (bool): True if current player should resign, else False
        
    '''
    colors = ("black", "b", "w", "white")

    #analyze command is for Sabaki (https://sabaki.yichuanshen.de/) 
    #pondering is a custom command
    commands = ("name","boardsize", "clear_board", "komi",\
                    "play", "genmove", "reg_genmove","final_score",\
                     "quit", "version", "showboard", "clear_cache",\
                     "last_move", "move_history", "undo", "help",\
                     "known_command", "protocol_version", "list_commands",\
                     "set_fixed_handicap", "printsgf", "loadsgf",\
                     "analyze", "pondering")

    #commands for Forest
    _forest_cmds = ("sync")

    def __init__(self, root,
                policy_net ,
                value_net = None,
                **kwargs):
        self.time_lim = kwargs.pop("time_lim", 20.0)
        self.n_rollouts = kwargs.pop("n_rollouts", None)
        self.pondering = kwargs.pop("pondering", True)
        self._conn = kwargs.pop("connection", None)
        super(GTP, self).__init__(root, policy_net, value_net, **kwargs)

        self.running = False 
        self._move_history = []
        self._last_root = None 
        self._undid = False
        self._input = [None] 

    def start(self): 
        self.running = True

        #mainloop
        while self.running:
            self.get_input()

            while self._input[0] is None:
                if self.pondering:
                    self.rollout(10)
                else:
                    sleep(0.5)
            out = self.send(self._input[0])

            #for analyze commands
            if isinstance(out, Generator):
                self.get_input()
                while self._input[0] is None:
                    try:
                        print(next(out), end= '')
                        sys.stdout.flush()
                    except StopIteration:
                        break
                out = self.send(self._input[0])

            if self._conn != None:
                self._conn.send(out)
            else:
                print(out, end = '')
                sys.stdout.flush()
        
    def stop(self):
        self._input[0] = "quit" 
        self.running = False

    def get_input(self): 
        self._input[0] = None
        def wait():
            if self._conn is not None:
                while not self._conn.poll():
                    sleep(0.5)
                self._input[0] = self._conn.recv()
            else:
                self._input[0] = input()
        in_thread = Thread(target = wait)
        in_thread.start()

    def send(self, cmd: str):
        '''Sends GTP command and returns response as a string, 
        or a Generator object for analyze commands'''
        if not self.running or not cmd:
            return
        valid = False
        out = ''
        cmd = cmd.lower().split()
        cmd_id = '' #optional command ID
        if re.match('\d+', cmd[0]):
            cmd_id = cmd[0]
            cmd = cmd[1:]

        this_turn = self.root.turn

        if cmd[0] not in GTP.commands:
            out = f"unknown command '{cmd[0]}'"

        elif cmd[0] == "protocol_version":
            out = "2"
            valid = True

        elif cmd[0] == "version":
            out = "0.3"
            valid = True

        elif cmd[0] == "known_command":
            if len(cmd) == 2:
                out = "true" if cmd[1] in GTP.commands else "false"
                valid = True

        elif cmd[0] == "boardsize":
            if len(cmd) != 2 or cmd[1] != '9':
                out = "boke only plays on 9x9 board"
            else:
                valid = True

        elif cmd[0] == "clear_board":
            self.set_root(Go_MCTS())
            valid = True

        elif cmd[0] == "komi":
            if len(cmd) <2: 
                out = "usage: komi <num-komi>"
            else:
                try:
                    self.root.komi = float(cmd[1])
                    valid = True
                except:
                    out = "invalid komi value"

        elif cmd[0] == "play":
            if len(cmd) <3 or not cmd[1] in GTP.colors: 
                out = "usage: play <color> <vertex>"
            elif cmd[2] == "resign":
                valid = True
                self.running = False
            else:
                try:
                    mv = go.squash(cmd[2])
                except:
                    out = "invalid coordinate"
                
                if out == "": 
                    turn = 0 if 'b' in cmd[1] else 1
                    if turn != this_turn%2: 
                        #one color playing two moves in a row
                        #increments turn by 2
                        new = self.root.make_move(go.PASS)
                        if new.is_legal(mv):
                            self._last_root = self.root
                            self.set_root(new.make_move(mv))
                            self._move_history.append(mv)
                            self._undid = False
                            valid = True
                        else:
                            out = "illegal move"
                    else:
                        #alternating play
                        try:
                            self.input_move(mv)
                            valid = True
                        except go.IllegalMove:
                            out = "illegal move"

        elif cmd[0] == "showboard":
            out = "\n" + str(self.root)
            valid = True

        elif cmd[0] in ("genmove", "reg_genmove"):
            if len(cmd) != 2 or not cmd[1] in GTP.colors:
                out = f"usage: {cmd[0]} <color>"
            else:
                turn = 0 if 'b' in cmd[1] else 1
                if turn != this_turn%2: 
                    self.input_move(go.PASS)
                    self._undid = True 
                resign = False if cmd[0] == "reg_genmove" else None
                mv = self.genmove(resign) 
                if mv == go.RESIGN:
                    out = "resign"
                    self.running = False 
                else:
                    out = go.unsquash(mv)
                valid = True

        elif cmd[0] == "undo":
            #only one undo allowed
            if self._undid or self._last_root is None:
                out = "cannot undo"
            else:
                self.set_root(self._last_root)
                self._move_history.pop()
                self._last_root = None
                self._undid = True
                valid = True

        elif cmd[0] == "last_move":
            mv = self.root.last_move
            last_col = "black " if this_turn%2 == 1 else "white "
            if mv is None:
                out = "no previous move known"
            else:
                out = last_col + go.unsquash(mv)
                valid = True

        elif cmd[0] == "name":
            out = "boke"
            valid = True

        elif cmd[0] == "quit":
            self.running = False
            valid = True

        elif cmd[0] in ("help", "list_commands"):
            out = '\n'.join(GTP.commands)
            valid = True

        elif cmd[0] == "clear_cache":
            self._prune()
            self._undid = True
            MCTS._val_cache = dict()
            MCTS._dist_cache = dict()
            MCTS._fts_cache = dict()

        elif cmd[0] == "final_score":
            score = self.root.score()
            if abs(score) < 1e-4:
                out = f"0"
            elif score>0:
                out = f"B+{score}"
            else:
                out = f"W+{-score}"
            valid = True

        elif cmd[0] == "move_history":
            out = '\n'.join(go.unsquash(self._move_history))
            valid = True
        
        elif cmd[0] == "set_fixed_handicap":
            if len(cmd) != 2 or not cmd[1].isnumeric():
                out = "usage: set_fixed_handicap <num-handicaps>"
            elif self.root.board != go.EMPTY_BOARD:
                out = "board is not empty"
            #handicaps for 9x9 game
            elif not 1 < int(cmd[1]) <=5:
                out = "invalid number of handicaps"
            else:
                handicaps = go.FLOWERS9[:int(cmd[1])]
                new_board = go.bulk_place_stones(go.BLACK, go.EMPTY_BOARD, handicaps)
                self.set_root(Go_MCTS(board = new_board, turn = 1))
                out = ' '.join(go.unsquash(list(handicaps)))
                valid = True

        elif cmd[0] == "printsgf":
            if len(cmd) == 2:
                outpath = cmd[1]
            else:
                outpath = os.path.join(os.getcwd(), "bokego.sgf")
            out = go.write_sgf(self._move_history, outpath)
            valid = True

        elif cmd[0] == "loadsgf":
            if len(cmd) != 3 or not cmd[2].isnumeric():
                out = "usage: loadsgf <path-to-sgf> <move-number>"
            else:
                try:
                    sgf_mvs = go.get_moves(cmd[1]) 
                    mv_num = int(cmd[2]) -1
                    for mv in sgf_mvs:
                        self.input_move(mv) 
                    out = "black" if mv_num%2 == 0 else "white"
                    valid = True
                except IOError as e:
                    out = str(e)
                except IndexError:
                    out = "invalid move number"
                except go.IllegalMove:
                    out = "illegal move in sgf"

        elif cmd[0] == "analyze":
            if len(cmd)!= 3 or cmd[1] not in GTP.colors or not cmd[2].isnumeric():
                out = "usage: analyze <color> <interval>"
            else:
                turn = 0 if 'b' in cmd[1] else 1
                if turn != this_turn%2: 
                    out = f"it is not {cmd[1]}'s turn"
                else:
                    return self.analyze(int(cmd[2]))

        elif cmd[0] == "pondering":
            if len(cmd)!= 2 or cmd[1] not in ("on", "off"):
                out = "usage: pondering <on/off>"
            else:
                self.pondering = True if cmd[1] == "on" else False
                valid = True

        if valid:
            return f"={cmd_id} {out}\n\n"
        else:
            return f"?{cmd_id} {out}\n\n"
        
    def input_move(self, sq_c):
        node = self.root.make_move(sq_c)
        self._last_root = self.root 
        self.set_root(node)
        self._move_history.append(sq_c)
        self._undid = False

    @property 
    def surrender(self):
        return self.winrate() is not None and\
                self.winrate() < 0.1 and self.root.turn > 50

    def genmove(self, resign = None):
        '''Generate move for current player. Returns squashed coordinate.
        optional:
            resign (bool): resign the game if true, else keep playing.
                           If not specified, uses the surrender condition'''
        if resign is not None:
            condition = resign
        else:
            condition = self.surrender 
        if condition:
            self.running = False
            return go.RESIGN

        if self.time_lim:
            self.timed_rollout(self.time_lim)
        elif self.n_rollouts:
            self.rollout(self.n_rollouts)            
        self._last_root = self.root
        child = self.choose()
        mv = child.last_move
        self._move_history.append(mv)
        self._undid = False
        return mv 
    
    def timed_rollout(self, time, 
                     analyze_dict = None):
        t_0 = default_timer()
        while default_timer() < t_0 + time: 
            self.rollout(analyze_dict = analyze_dict)
    
    def analyze(self, interval, k = 3):    
        '''Yield rollout information (visits, winrates, prior, variations)
        updated at regular intervals until input is recieved
        args:
            interval (int): the update interval in centiseconds
            k (int): show the top k moves (default 3)'''
        variations = dict()
        yield "= \n"
        while True: 
            self.timed_rollout( interval/200.0, 
                                analyze_dict = variations )
            best_mvs = sorted(variations.keys(), 
                        key = lambda n: self.N[n])
            if self._input[0] is not None:
                break
            out = ""
            for n in best_mvs[-k:]:
                mv = go.unsquash(n.last_move)
                variation = go.unsquash( [m.last_move for m in variations[n] ]) 
                prior = self.root.dist.probs[n.last_move]
                out += f"info move {mv} visits {self.N[n]} "\
                       f"winrate {10000*n.winrate:.0f} "\
                       f"prior {10000*prior:.0f} "\
                      "pv " + " ".join(variation) + " "
            yield out + "\n"

#TODO
#write Forest
class Forest(MCTS):
    '''Manages several MCTS trees in separate processes.'''
    def __init__(self, num_trees, root,
                policy_net, value_net,
                **kwargs):
        super(Forest, self).__init__(root, policy_net, value_net, **kwargs)
        self.trees = []
        self.conns = [] 
        self.num_trees = num_trees
        self.root = root

        policy_net.share_memory()
        value_net.share_memory()

        for n in range(num_trees):
            master_conn, tree_conn = mp.Pipe() 
            self.conns.append(master_conn)
            kwargs["connection"] = tree_conn
            p = Process(target = _tree_worker,
                            args = (n, self.root, policy_net, value_net),
                            kwargs = kwargs) 
            self.trees.append(p)
        
    def forest_choose(self):
        '''Get combined visits and choose child'''
        pass             

    def sync_forest(self):
         pass

    def start(self):
        for p in self.trees:
            p.start() 

        while True:
            cmd = input()

    def send_all(self, data):
        for conn in self.conns:
            conn.send(data)

def _tree_worker(id_num, root, policy_net, value_net, **kwargs):
    np.random.seed()
    #tree is listening on connection
    tree = GTP(root, policy_net, value_net, **kwargs)
    tree.start()    

class GTPprocess():
    '''Runs a GTP engine in a subprocess. 
    args:
        label: id for subprocess
        cmd (str): shell command that starts the GTP engine    
        verbose (bool): print everything received and sent (default False)
    '''
    def __init__(self, label, cmd, verbose = False):
        cmd = shlex.split(cmd)
        self.verb = verbose
        self.id = label
        self._name = None
        self.subproc = Popen(cmd, stdin=PIPE, stdout=PIPE)
        
        try:
            gtp_version = self.send("0 protocol_version\n")
            assert gtp_version[:2] == '=0', "invalid response"
            assert gtp_version.split(" =\n") == "2", "wrong protocol version"
        except Exception as e:
            self.close()
            raise e
        print(f"Process {self.id} created with {self.name}")

    def send(self, data: str):
        if self.verb:
            print(f"sending {self.id}: {data}")
        #can only PIPE bytes objects
        self.subproc.stdin.write(data.encode('utf-8'))
        self.subproc.stdin.flush()
        result = ""
        while True:
            data = self.subproc.stdout.readline()
            if data is None:
                break
            result += data.decode('utf-8')
        if self.verb:
            print(f"received: {result}")
        return result

    def close(self):
        print(f"Closing process {self.id}")
        try:
            self.subproc.communicate("quit\n".encode('utf-8'), timeout = 10)
        except TimeoutExpired:
            self.subproc.kill()

    @property
    def name(self):
        if self._name is None:
            self._name = self.subproc.send("name\n").strip(" =\n")
        return self._name

    def version(self):
        return self.subproc.send("version\n").strip(" =\n")

    def known(self, cmd):
        known = self.subproc.send(f"known_command {cmd}\n").strip(" =\n")
        if known.lower() == "true":
            return True
        return False

    def boardsize(self, boardsize):
        return self.subproc.send(f"boardsize {boardsize}\n").strip(" =\n")

    def komi(self, komi):
        return self.subproc.send(f"komi {komi}\n").strip(" =\n")

    def clear_board(self):
        self.subproc.send("clear_board\n")

    def genmove(self, color):
        return self.subproc.send( f"genmove {color}\n").strip(" =\n")

    def showboard(self):
        self.subproc.send("showboard\n")

    def play(self, color, move):
        self.subproc.send(f"play {color} {move}\n")

    def final_score(self):
        return self.subproc.send("final_score\n").strip(" =\n")


def GTP_match(B_cmd, W_cmd, sgf_path = None):
    '''Play a game between two GTP engines.
    Return True if black wins and False if white wins
    args:
        B_cmd (str): shell command to start black GTP engine
        W_cmd (str): shell command to start white GTP engine
        sgf_path (str): optional path to write sgf to'''

    white = GTPprocess("white", B_cmd, verbose = True)
    black = GTPprocess("black", W_cmd, verbose = True)

    black.name()
    black.version()
    white.name()
    white.version()

    black.boardsize(9)
    white.boardsize(9)

    black.komi(5.5)
    white.komi(5.5)

    black.clear_board()
    white.clear_board()

    first_pass = False

    def abort():
        print("No move recieved. Exiting")
        return 

    moves = []
    while True:
        vertex = black.genmove("black").upper()
        if not vertex:
            abort()
        if vertex == "PASS":
            if first_pass:
                break
            else:
                first_pass = True
        else:
            first_pass = False
            moves.append(go.squash(vertex))

        white.play("black", vertex)
        white.showboard()
        vertex = white.genmove("white")

        if not vertex:
            abort()
        elif vertex == "PASS":
            if first_pass:
                break
            else:
                first_pass = True
        else:
            first_pass = False
            moves.append(go.squash(vertex))

        white.showboard()
        black.play("white", vertex)
    
    score = black.final_score() 
    go.write_sgf(moves, sgf_path, 
                B = black.name(), 
                W = white.name(), 
                result = score)

    black.close()
    white.close()
    return 'B' in score

