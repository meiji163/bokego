from subprocess import Popen, PIPE
import go
from selfPlay import write_sgf
from time import sleep
import re

class GTPprocess(object):
    '''A class that manages io to GTP subprocess'''
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
        return self.gtp_subprocess.send("name\n").strip(" =\n")

    def version(self):
        return self.gtp_subprocess.send("version\n").strip(" =\n")

    def boardsize(self, boardsize):
        return self.gtp_subprocess.send("boardsize {}\n".format(boardsize)).strip(" =\n")

    def komi(self, komi):
        return self.gtp_subprocess.send("komi {}\n".format(komi)).strip(" =\n")

    def clear_board(self):
        self.gtp_subprocess.send("clear_board\n")

    def genmove(self, color):
        message = self.gtp_subprocess.send( "genmove {}\n".format(color))
        mv = re.search('[ABCDEFGHJ]\d', message)
        if mv:
            return mv[0]
        pass_mv = re.search('PASS', message)
        if pass_mv:
            return pass_mv[0]
        return "Error: " + message 

    def showboard(self):
        self.gtp_subprocess.send("showboard\n")

    def play(self, color, move):
        self.gtp_subprocess.send("play {} {}\n".format(color,move))

    def final_score(self):
        return self.gtp_subprocess.send("final_score\n").strip(" =\n")

    def close(self):
        self.gtp_subprocess.close()



def gtpGame(B_PLYR, W_PLYR, sgf_path):
    '''Play a game between B_PLYR and W_PLYR and record the game in sgf_path.
    Return True if black wins and False if white wins'''

    white = GTPplyr("white", W_PLYR)
    black = GTPplyr("black", B_PLYR)

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
            if first_pass:
                break
            else:
                first_pass = True
        else:
            first_pass = False
            moves.append(vertex)

        white.showboard()
        black.play("white", vertex)
    
    score = black.final_score() if black.name() == "GNU Go" else white.final_score()

    write_sgf(go.squash(moves, alph = True) , sgf_path, B= black.name() , W=white.name(), result = score)
    black.close()
    white.close()
    return 'B' in score

GNUGO = ["gnugo", "--chinese-rules","--mode", "gtp"]
GNUGO_MCTS = ["gnugo", "--chinese-rules", "--mode", "gtp","--monte-carlo"]
BOKE_B = ["python", "bokePlay.py", "--mode", "gtp", "-c", "B"]
BOKE_W = ["python", "bokePlay.py", "--mode", "gtp"]

if __name__ == "__main__":
    wins = 0
    #for n in range(1,6):
    #    wins += gtpGame(BOKE_B, GNUGO, f"boke_gnugo_{n}.sgf")
    gtpGame(GNUGO, BOKE_W, f"boke_gnugo_10.sgf")
