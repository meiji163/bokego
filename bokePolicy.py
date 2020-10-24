import go
from parse_sgf import get_moves
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # 3x3 Convolution: 9x9 --> 5x5x(30 features) --> 3x3x30
        # + 3 fully connected layers
        self.conv = nn.Conv2d(1,30,3)
        self.pool = nn.MaxPool2d((2,2))
        self.l1 = nn.Linear(30*3*3, 200)
        self.l2 = nn.Linear(200, 120)
        self.l3 = nn.Linear(120, 81)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def policy_predict(policy, board, device = "cpu"):
    board = np.array(board).reshape((9,9))
    board = torch.Tensor(board).unsqueeze(0).unsqueeze(0).float()
    predicts = torch.topk(policy(board).squeeze(0), 3)
    return [go.unsquash(sq_c) for sq_c in predicts[1].tolist()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "BokeGo Policy Prediction")
    parser.add_argument("path", metavar="path", type = str, nargs = 1, help = "path to model")
    parser.add_argument("--sgf", metavar="sgf", type = str, nargs = 1, help = "path to sgf")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pi = Policy()
    pi.load_state_dict(torch.load(args.path[0], map_location=device))
    pi.eval()
    
    if args.sgf != None:
        print(args.sgf)
        mvs = get_moves(args.sgf[0])
        g = go.Game(moves = mvs)

    else:
        g = go.Game()

    uin = ""
    while(uin != 'q'):
        print(g)
        uin = input("\t- press p to show prediction\n\
        - enter coordinate to play move\n\
        - press q to quit\n")
        if uin == 'p':
            print(policy_predict(pi, g.get_board(), device))
        else:
            g.play_move(tuple([int(i) for i in uin.split(' ')]))

