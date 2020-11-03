import go
import os
from bokePolicy import PolicyNet 
from go_mcts import *
from mcts import MCTS, Node
from threading import Thread
import torch
import argparse
from time import sleep

parser = argparse.ArgumentParser(description = "Play against Boke Go")
parser.add_argument("--path", metavar="MODEL", type = str, nargs = 1, help = "path to model", default = "v0.5/RL_policy_3.pt")
parser.add_argument("--color", metavar = "COLOR", type = str, nargs = 1, choices = ['W','B'], default = 'W', help = "Boke's color")
parser.add_argument("--selfplay", action = 'store_true', dest = 'selfplay', help = 'self play', default = False)
parser.add_argument("-r", type = int, nargs = 1, help = "number of rollouts/move", default = 100)
args = parser.parse_args()

def get_input(in_ref):
    in_ref[0] = input("Your Move: ")

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

if  __name__ == "__main__":
   
    NUM_ROLLOUTS = args.r 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pi = PolicyNet()
    checkpt = torch.load(args.path, map_location = device)
    pi.load_state_dict(checkpt["model_state_dict"])
    pi.eval()
    board = Go_MCTS(policy = pi)
    tree = MCTS(exploration_weight = 1)

    
    if args.selfplay:
        for _ in range(40):
            print(g)
            g.play_move(policy_sample(pi, g))
            sleep(1)


    if args.color == 'B' and board.turn == 0:
        tree.do_rollout(board,NUM_ROLLOUTS)
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
        if uin == 'p':
            probs, moves = policy_predict(pi, g, device, k = 5)
            print(go.unsquash(moves.tolist()))   
            print(probs.tolist())
        elif uin == 'q':
            break
        else:
            try:
                sq_c = 9*(ord(uin[0])-65) + int(uin[1]) - 1
                board = board.make_move(sq_c)
                clear()
                print(board)
                tree.do_rollout(board,NUM_ROLLOUTS)
                board = tree.choose(board)
            except: 
                print("Enter coordinate (e.g. D5) to play a move\nPress p to show policy prediction\nPress q to quit" )

        in_ref[0] = None
