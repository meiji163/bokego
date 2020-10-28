import go
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

SCALE = 10.0

class PolicyNet(nn.Module):
    def __init__(self, scale = 1):
        super(PolicyNet, self).__init__()
        '''7 9x9 input features
        5x5 convolution: 9x9 -> 9x9
        3x3 convolution: 9x9 -> 9x9
        3x3 convolution: 9x9 -> 7x7
        3 fully connected hidden layers with dropout
        output distribution over coords 0-81'''
        self.conv = nn.Sequential(
                nn.Conv2d(7,10,5, padding = 2),
                nn.ReLU(),
                nn.Conv2d(10,16,3, padding =1),
                nn.ReLU(),
                nn.Conv2d(16,32,3),
                nn.ReLU())
        self.lin = nn.Sequential(
                nn.Linear(32*7*7, 1024 , bias = False),
                nn.Dropout(0.6),
                nn.ReLU(),
                nn.Linear(1024, 600, bias = False),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(600, 200, bias = False),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(200, 81)
                )
        self.scale = scale #scalar for the data

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.num_flat_features(x))
        return self.lin(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NinebyNineGames(Dataset):
    def __init__(self, path, transform = None, scale = 1):
        '''read boards csv from path. 
        transform is a tuple (n,m), n = 0,1,2,3 and m = 0,1  
        which applies a rotation of 90*n clockwise and m reflections to the data.
        This generates the full symmetry group of the square 
        '''
        self.boards = pd.read_csv(path)
        self.path = path
        self.transform = transform
        self.scale = scale

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board, ko, turn, move = self.boards.iloc[idx]
        ko = (None if ko == "None" else int(ko))
        g = go.Game(board = board, turn = turn, ko = ko)
        
        if self.transform: 
                for _ in range(self.transform[0]):
                    #rotates 90 deg clockwise
                    move = (move*9 + 8 - move//9)%81
                for _ in range(self.transform[1]):
                    x, y = divmod(move, 9)
                    move = 9*y + x
        return features(g, scale = self.scale, transform = self.transform).float(), move

def features(game: go.Game, scale = 1, transform = None):
    ''' go.Game --> (7,9,9) torch.Tensor
        layer: feature
        0: player stones
        1: opponent stones
        2: empty
        3: turn
        4: legal
        5: liberties
        6: liberties after playing'''
    empty = np.array(game.get_board()).reshape(9,9)
    plyr = empty.copy()
    oppt = empty.copy()
    empty[empty == 0] = 2 
    empty[empty != 2] = 0

    turn_num = (1 if game.turn%2 == 0 else -1)
    color = (go.BLACK if turn_num == 1 else go.WHITE)
    plyr[plyr == -turn_num] = 0
    oppt[oppt == turn_num] = 0
    plyr *= turn_num 
    oppt *= -turn_num

    if color == 1:
        turn = np.ones((9,9), dtype = float)
    else:
        turn = np.zeros((9,9), dtype = float)

    legal = np.array([game.is_legal(sq_c) for sq_c in range(81)]).reshape(9,9)
    libs = np.array(game.get_liberties()).reshape(9,9)
    libs_after = np.array([go.get_stone_lib(go.place_stone(color, game.board,\
            sq_c), sq_c) for sq_c in range(81)]).reshape(9,9)
    fts = np.stack([plyr, oppt, empty, turn, legal, libs, libs_after])

    if transform:
        out = np.rot90(fts, k = (3*transform[0])%4 , axes = (1,2))
        if transform[1]:
            return scale*torch.from_numpy(np.ascontiguousarray(np.transpose(out, axes = (0,2,1))))
        else:
            return scale*torch.from_numpy(np.ascontiguousarray(out))
    return scale*torch.from_numpy(fts)

def policy_predict(policy: PolicyNet, game: go.Game , device = "cpu"):
    fts = features(game, policy.scale).unsqueeze(0).float()
    predicts = torch.topk(F.softmax(policy(fts), dim = 1).squeeze(0), 5)
    return predicts 


