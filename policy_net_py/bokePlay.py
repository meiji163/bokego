import go
from bokePolicy import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from time import sleep

if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Boke Policy Prediction")
    parser.add_argument("path", metavar="MODEL", type = str, nargs = 1, help = "path to model")
    parser.add_argument("--sgf", metavar="SGF", type = str, nargs = 1, help = "path to sgf")
    parser.add_argument("--selfplay", action = 'store_true', dest = 'selfplay', help = 'self play')
    parser.set_defaults(selfplay = False)
    args = parser.parse_args()
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pi = PolicyNet()
    checkpt = torch.load(args.path[0], map_location = device)
    pi.load_state_dict(checkpt["model_state_dict"])
    pi.eval()
    
    if args.sgf:
        g = go.Game(sgf = args.sgf[0])
    else:
        g = go.Game()
    
    if args.selfplay:
        for _ in range(40):
            print(g)
            g.play_move(policy_sample(pi, g))
            sleep(1)

    while(True):
        print(g)
        if args.sgf:
            uin = input("   \t- press n to see next move in sgf\n\
        - press p to show prediction\n\
        - press q to quit\n")
        else:
            uin = input("\t Enter Move: ")
        if args.sgf and uin == 'n':
            g.play_move()
        elif uin == 'p':
            probs, moves = policy_predict(pi, g, device, k = 5)
            print(go.unsquash(moves.tolist()))   
            print(probs.tolist())
        elif uin == 'q':
            break
        elif uin == 'u':
            g.undo(2)
        else:
            try:
                g.play_move(go.squash(tuple([int(i) for i in uin.split(' ') ])) )
                mv = policy_sample(pi,g)
                while not g.is_legal(mv):
                    mv = policy_sample(pi,g)
                print(g)
                sleep(1)
                g.play_move(policy_sample(pi, g))
            except:
                print("\t - Enter coordinate \"x y\" to play a move\n\
                - Press p to show prediction\n\
                - Press q to quit" )

