import go
from sys import exit, stdout
from itertools import cycle
from os import system, name
from bokePolicy import PolicyNet, policy_predict
from go_mcts import *
from mcts import MCTS, Node
from threading import Thread
from torch import load, device, no_grad 
import argparse
from time import sleep

parser = argparse.ArgumentParser(description = "Play against Boke")
parser.add_argument("-p", metavar="PATH", type = str, dest = 'p', help = "path to model", default = "v0.5/RL_policy_3.pt")
parser.add_argument("-c", type = str, action = 'store', choices = ['W','B'], dest = 'c', help = "Boke's color", default = 'W')
parser.add_argument("--selfplay", action = 'store_true', dest = 'selfplay', help = 'self play', default = False)
parser.add_argument("-r", nargs = 1, metavar="ROLLOUTS", action = 'store', type = int, default = 25, dest = 'r', help = "number of rollouts per move")

args = parser.parse_args()

def get_input(in_ref):
    in_ref[0] = input("Your Move: ")

def clear():
    system('cls' if name == 'nt' else 'clear')

done = False #toggle thinking
def loading():
    for c in cycle(['|', '/', '-', '\\']):
        if done:
            break
        stdout.write('\rThinking ' + c)
        stdout.flush()
        sleep(0.1)

if  __name__ == "__main__":
    NUM_ROLLOUTS = args.r[0]
    device = device("cuda" if torch.cuda.is_available() else "cpu")
    pi = PolicyNet()
    checkpt = load(args.p, map_location = device)
    pi.load_state_dict(checkpt["model_state_dict"])
    pi.eval()
    board = Go_MCTS(policy = pi)
    tree = MCTS(exploration_weight = 5)

    with no_grad():
        if args.selfplay:
            for _ in range(40):
                print(g)
                g.play_move(policy_sample(pi, g))
                sleep(1)
            exit()


        if args.c == 'B' and board.turn == 0: 
            tree.do_rollout(board, NUM_ROLLOUTS)
            board = tree.choose(board)
        
        in_ref = [None]
        while(True):
            clear()
            print(board)

            #thread waiting for opponent's move
            thread = Thread(target=get_input, args = (in_ref,))
            thread.start()

            #rollouts while waiting
            while in_ref[0] == None:
                tree.do_rollout(board)
            
            uin = in_ref[0]
            if uin == 'h':
                moves = policy_predict(pi, board, device, k = 4).indices.tolist()
                print("Boke's top moves: " + ','.join([chr(mv//9+65) + str(mv%9) for mv in moves]))
                sleep(3)
            elif uin == 'q':
                exit() 
            else:
                try:
                    done = False
                    sq_c = 9*(ord(uin[0])-65) + int(uin[1]) - 1
                    board = board.make_move(sq_c)
                    clear()
                    print(board)
                    t = Thread(target = loading)
                    t.start() 
                    tree.do_rollout(board,NUM_ROLLOUTS)
                    done = True
                    board = tree.choose(board)
                except: 
                    print("Enter coordinate (e.g. D5) to play a move\nEnter h to show hint\nEnter q to quit" )
                    sleep(3)
            if board.terminal:
                exit()
            in_ref[0] = None
