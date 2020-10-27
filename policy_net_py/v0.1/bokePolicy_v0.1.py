import go
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

SCALE = 100

class PolicyNet(nn.Module):
    def __init__(self, scale = 1):
        super(PolicyNet, self).__init__()
        '''6 9x9 input features
        5x5 convolution: 9x9 -> 9x9
        3x3 convolution: 9x9 -> 9x9
        1 fully connected hidden layers
        output distribution over coords 0-81'''
        self.conv1 = nn.Conv2d(6, 10, 5, padding = 2)
        self.conv2 = nn.Conv2d(10, 15, 3, padding = 1) 
        self.l1 = nn.Linear(15*9*9, 256, bias = False)
        self.l2 = nn.Linear(256, 128, bias = False)
        self.l3 = nn.Linear(128 , 81, bias = False)
        self.scale = scale #scalar for the data

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
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

class NinebyNineGames(Dataset):
    def __init__(self, path, transform = None, scale = 1):
        '''read boards csv from path. 
        transform = "rot90" rotates by 90 deg clockwise,
        transform = "reflect" reflects along main diagonal''' 
        self.boards = pd.read_csv(path)
        self.path = path
        self.transform = transform
        self.scale = scale
    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board, ko, move = self.boards.iloc[idx]
        ko = (None if ko == "None" else int(ko))
        g = go.Game(board = board, ko = ko)

        if self.transform == "rot90":
            #rotates 90 deg clockwise
            move = (move*9 + 8 - move//9)%81
        elif self.transform == "reflect":
            #reflect along the main diagonal
            x, y = divmod(move, 9)
            move = 9*y + x

        return features(g, scale = self.scale, transform = self.transform), move

def features(game: go.Game, scale = 1, transform = None):
    ''' go.Game --> (6,9,9) torch.Tensor
        layer: feature
        0: black stones
        1: white stones
        2: empty 
        3: legal
        4: liberties
        5: liberties after playing'''
    empty = np.array(game.get_board()).reshape(9,9)
    black = empty.copy()
    white = empty.copy()
    black[black == -1] = 0
    white[white == 1] = 0
    white *= -1
    empty[empty == 0] = 2 
    empty[empty != 2] = 0
    legal = np.array([game.is_legal(sq_c) for sq_c in range(81)]).reshape(9,9)
    libs = np.array(game.get_liberties()).reshape(9,9)
    color = (go.BLACK if game.turn%2 == 0 else go.WHITE)
    libs_after = np.array([go.get_stone_lib(go.place_stone(color, game.board,\
            sq_c), sq_c) for sq_c in range(81)]).reshape(9,9)
    fts = np.stack([black, white, empty, legal, libs, libs_after])
    if transform == "reflect":
        return scale*torch.from_numpy(np.transpose(fts, axes = (0,2,1)))
    elif transform == "rot90":
        return scale*torch.from_numpy(np.ascontiguousarray(np.rot90(fts, k = 3, axes = (1,2))))
    return scale*torch.from_numpy(fts)

def policy_predict(policy: PolicyNet, game: go.Game , device = "cpu"):
    fts = features(game, policy.scale).unsqueeze(0)
    predicts = torch.topk(F.softmax(policy(fts), dim = 1).squeeze(0), 5)
    return predicts 


