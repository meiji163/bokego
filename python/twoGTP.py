from subprocess import Popen, PIPE
import go
from time import sleep
import re
class GTPprocess(object):

    def __init__(self, label, args):
        self.label = label
        self.subprocess = Popen(args, stdin=PIPE, stdout=PIPE)
        print("{} subprocess created".format(label))

    def send(self, data):
        print("sending {}: {}".format(self.label, data))
        #PIPE can only send bytes objects
        self.subprocess.stdin.write(data.encode('utf-8'))
        self.subprocess.stdin.flush()
        result = ""
        while True:
            data = self.subprocess.stdout.readline()
            if data == b'\n':
                break
            result += data.decode('utf-8')
        print("received: {}".format(result))
        return result

    def close(self):
        print("quitting {} subprocess".format(self.label))
        self.subprocess.communicate("quit\n".encode('utf-8'))


class GTPplyr(object):

    def __init__(self, color, args):
        self.color = color 
        self.gtp_subprocess = GTPprocess(color, args)

    def name(self):
        self.gtp_subprocess.send("name\n")

    def version(self):
        self.gtp_subprocess.send("version\n")

    def boardsize(self, boardsize):
        self.gtp_subprocess.send("boardsize {}\n".format(boardsize))

    def komi(self, komi):
        self.gtp_subprocess.send("komi {}\n".format(komi))

    def clear_board(self):
        self.gtp_subprocess.send("clear_board\n")

    def genmove(self, color):
        message = self.gtp_subprocess.send( "genmove {}\n".format(color))
        mv = re.search('[ABCDEFGHJ]\d', message)
        if mv:
            return mv[0]
        else:
            return self.gtp_subprocess.close()

    def showboard(self):
        self.gtp_subprocess.send("showboard\n")

    def play(self, color, move):
        self.gtp_subprocess.send("play {} {}\n".format(color,move))

    def final_score(self):
        self.gtp_subprocess.send("final_score\n")

    def close(self):
        self.gtp_subprocess.close()

def write_to_sgf(out_path, moves, score):
    if score > 0:
        res = f"RE[B+{score}]"
    else:
        res = f"RE[W+{-score}]"
    print(res)
    out = "(\n;PB[GnuGo]PW[Boke]" + res + "HA[0]KM[5.5]SZ[9]GM[1]\n"
    turn = 0
    for mv in moves: 
        sq_c = go.squash(mv, alph = True)
        plr =('B' if turn == 0 else 'W')
        out += ";" + plr + "[" + chr(sq_c//9 +97) + chr(sq_c%9 +97) + "]\n"
        turn = 1 - turn
    out += ")"
    with open(out_path, 'w') as f:
        f.write(out)

    

GNUGO = ["gnugo", "--chinese-rules","--mode", "gtp"]
GNUGO_MCTS = ["gnugo", "--chinese-rules", "--mode", "gtp","--monte-carlo"]
BOKE_B = ["python", "bokePlay.py", "--mode", "gtp", "-r", "200", "-c", "B"]
BOKE_W = ["python", "bokePlay.py", "--mode", "gtp", "-r", "200"]

moves = []
white = GTPplyr("white", BOKE_W)
black = GTPplyr("black", GNUGO)

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

moves = []
while True:
    vertex = black.genmove("black")
    if vertex == "PASS":
        break
        if first_pass:
            break
        else:
            first_pass = True
    else:
        first_pass = False
        moves.append(vertex)
    white.play("black", vertex)
    white.showboard()

    vertex = white.genmove("white")
    if vertex == "PASS":
        break
        if first_pass:
            break
        else:
            first_pass = True
    else:
        first_pass = False
        moves.append(vertex)

    white.showboard()
    black.play("white", vertex)

write_to_sgf("boke_gnugo_6.sgf",moves, black.final_score())
black.final_score()
white.final_score()

black.close()
white.close()
