from subprocess import Popen, PIPE
import go
from selfPlay import write_sgf
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
        pass_mv = re.search('PASS', message)
        if pass_mv:
            return pass_mv[0]
        return "Error: " + message 

    def showboard(self):
        self.gtp_subprocess.send("showboard\n")

    def play(self, color, move):
        self.gtp_subprocess.send("play {} {}\n".format(color,move))

    def final_score(self):
        self.gtp_subprocess.send("final_score\n")

    def close(self):
        self.gtp_subprocess.close()

    

GNUGO = ["gnugo", "--chinese-rules","--mode", "gtp"]
GNUGO_MCTS = ["gnugo", "--chinese-rules", "--mode", "gtp","--monte-carlo"]
BOKE_B = ["python", "bokePlay.py", "--mode", "gtp", "-r", "100", "-c", "B"]
BOKE_W = ["python", "bokePlay.py", "--mode", "gtp", "-r", "200"]


if __name__ == "__main__":
    white = GTPplyr("white", GNUGO)
    black = GTPplyr("black", BOKE_B)

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

    score = float(re.findall("[BW]\+.+", white.final_score())[0])
    if score > 0:
        res = f"B+{score}"
    else:
        res = f"W+{-score}"
    write_sgf(moves, "boke_gnugo_6.sgf", B="Boke", W="GNUGO", result = res)
    black.final_score()
    white.final_score()
    black.close()
    white.close()
