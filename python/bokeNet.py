import go
import os
from math import sqrt
import numpy as np
import pandas as pd
import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

SOFT = nn.Softmax(dim = 1)

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        '''27 9x9 input features
        1 5x5 convolution: 9x9 -> 9x9
        5 3x3 convolution: 9x9 -> 9x9
        1 1x1 convolution with untied bias: 9x9 -> 81
        output distribution over coords 0-81'''
        self.conv = nn.Sequential(
                nn.Conv2d(27,64,5, padding = 2),
                nn.ReLU(),
                nn.Conv2d(64,128,3, padding =1),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.ReLU(),
                Conv2dUntiedBias(9,9,128,1,1))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 81)
        return x 

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ValueNet(nn.Module):
    '''27 9x9 input features
    1 5x5 convolution: 9x9 -> 9x9
    4 3x3 convolutions: 9x9 -> 9x9
    1 1x1 convolution: 9x9 -> 9x9
    2 fully connected layers
    output value (between -1 and 1)
    '''
    def __init__(self):
        super(ValueNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(27,64,5, padding = 2),
            nn.ReLU(),
            nn.Conv2d(64, 128,3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128,3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1))
        self.lin1 = nn.Linear(128*9*9, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(-1, 128*9*9) 
        x = self.lin1(x)
        return x.view(-1, 1) 
        

class NinebyNineGames(Dataset):
    def __init__(self, path):
        '''read boards csv from path.'''
        cols = pd.read_csv(path, nrows = 0).columns
        self.boards = pd.read_csv(path, converters = {col: self.convert_type for col in cols}, low_memory = False)
        self.path = path

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board, ko, turn, last, res = self.boards.iloc[idx]
        g = go.Game(board = board, turn = turn, ko = ko, last_move = last)
        #if (res == 'B' and turn%2 == 0) or (res == 'W' and turn%2 == 1):
        #reward = 1 if current player wins the game, -1 if he loses.
        #    reward = 1.0
        #else:
        #    reward = -1.0
        return features(g), res 

    @staticmethod
    def convert_type(x):
        if x == "None":
            return None
        if x.isnumeric():
            return int(x)
        elif x == "-1": 
            return -1
        else:
            return str(x)
        

def features(game: go.Game):
    ''' go.Game --> (27,9,9) torch.Tensor
        Compute the input features from the board state
        
        9x9 layer index: feature
        ------------------------
        0: player stones        
            1 if coord has player's stones, else 0
        1: opponent stones      
            1 if coord has opponent's stone, else 0
        2: empty                
            1 if coord is empty, else 0
        3: turn                 
            all 1's if it is B's turn, all 0's if it is W's turn
        4: last move
            1 if coord was last move, else 0
        5: legal                
            1 if coord is legal move for player, else 0
        6-12: liberties         
            n if stone at coord has n liberties, else 0
            layer 5 has coords with 1 liberty
            layer 6 has coords with 2 liberties
            ...
            layer 11 has coords with >6 liberties
        13-19: liberties after playing
            n if coord is a legal move and player's stone has n liberties after playing, else 0
            liberties are separated the same way as 5-11
        20-26: number of captures
            n if playing at coord would capture n opponent stones, else 0
            number of captures are separated the same way as 5-11
        '''
    plyr = np.array(game.get_board()).reshape(1,9,9)
    oppt = np.copy(plyr) 
    turn_num = (1 if game.turn%2 == 0 else -1)
    color = (go.BLACK if turn_num == 1 else go.WHITE)
    plyr[plyr == -turn_num] = 0
    oppt[oppt == turn_num] = 0
    plyr *= turn_num 
    oppt *= -turn_num
    empty = np.invert((plyr + oppt).astype(bool)).astype(float)
    if color == go.BLACK:
        turn = np.ones((1,9,9), dtype = float)
    else:
        turn = np.zeros((1,9,9), dtype = float)
    last_mv = np.zeros(81, dtype = float)
    if isinstance(game.last_move, int) and game.last_move >= 0:
        last_mv[game.last_move] = 1.0
    last_mv = last_mv.reshape(1,9,9)
    legal = np.array([game.is_legal(sq_c) for sq_c in range(81)], dtype = float) #very slow
    libs = np.array(game.get_liberties(), dtype = float).reshape(9,9)
    libs_after = np.zeros(81, dtype = float)
    caps = np.zeros(81, dtype = float)
    for sq_c in np.nonzero(legal)[0]:
        new_board, opp_captured = go.get_caps(go.place_stone(color, game.board,sq_c), sq_c, color)
        if opp_captured:
            libs_after[sq_c] = go.get_stone_lib(new_board, sq_c)
            caps[sq_c] = len(opp_captured)
        else:
            libs_after[sq_c] = go.get_stone_lib(go.place_stone(color, game.board, sq_c), sq_c) 
            
    libs_after = libs_after.reshape(9,9)
    caps = caps.reshape(9,9)
    legal = legal.reshape(1,9,9)
    fts = np.vstack( [plyr, oppt, empty, turn, last_mv, legal,\
            separate(libs) , separate(libs_after) , separate(caps)])
    return torch.from_numpy(fts).float()

def separate(arr):
    out = np.zeros((7,9,9), dtype = float)
    for i in range(6):
        out[i, arr == i+1] = i+1
    out[6, arr >6] = 7
    return out 

def policy_dist(policy: PolicyNet, game: go.Game , device = "cpu"):
    '''Return torch.distribution.Categorial distribution over coordinates'''
    fts = features(game).unsqueeze(0)
    fts = fts.to(device)
    probs = SOFT(policy(fts)).squeeze(0)
    dist = Categorical(probs)
    return dist

def value(v: ValueNet, game: go.Game, device = torch.device("cpu")):
    fts = features(game).unsqueeze(0).to(device)
    return v(fts).item()

def policy_sample(policy: PolicyNet, game: go.Game, device = "cpu"):
    '''sample a move from policy distribution. Use policy_dist
    for multiple samplings'''
    fts = features(game).unsqueeze(0)
    fts = fts.to(device)
    probs = SOFT(policy(fts)).squeeze(0)
    m = Categorical(probs)
    return m.sample()


class Conv2dUntiedBias(nn.Module):
    def __init__(self, height, width, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2dUntiedBias, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels, height, width))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        output = F.conv2d(input, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)
        # add untied bias
        output += self.bias.unsqueeze(0).repeat(input.size(0), 1, 1, 1)
        return output

