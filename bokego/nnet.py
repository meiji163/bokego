import bokego.go as go
import os
from math import sqrt
from tqdm import trange
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

#v0.3 Policy Net
class PolicyNet(nn.Module):
    '''(27,9,9) torch.Tensor --> (81) torch.Tensor
    Takes input from features(game: go.Game). 
    The softmax of the output is the prior distribution over moves (0--80)
    
    Layers:
        1 5x5 convolution: 9x9 -> 9x9
        6 3x3 convolution: 9x9 -> 9x9
        1 1x1 convolution with untied bias: 9x9 -> 9x9
    '''
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(27,128,5, padding = 2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding =1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                Conv2dUntiedBias(9,9,128,1,1))
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 81)
        return x 

class ValueNet(nn.Module):
    '''(27,9,9) torch.Tensor --> (1) toch.Tensor
    Takes input from features(game: go.Game). 
    The output is the expected value of the game from current player's perspective; 
    win = 1, lose = -1
    
    Layers:
        1 5x5 convolution: 9x9 -> 9x9
        6 3x3 convolutions: 9x9 -> 9x9
        1 convolution with untied bias: 9x9 -> 9x9
        2 fully connected layers 81 -> 64 -> 1
    '''
    def __init__(self):
        super(ValueNet, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(27,128,5, padding = 2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding =1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128,3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                Conv2dUntiedBias(9,9,128,1,1))
        self.lin1 = nn.Linear(81,64)
        self.lin2 = nn.Linear(64,1)
        self.bn = nn.BatchNorm2d(1) 
        self.lin_bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def load_policy_dict(self, policy_dict):
        '''load convolution weights from a PolicyNet state dict'''
        new_dict = self.state_dict()
        new_dict.update(policy_dict)
        self.load_state_dict(new_dict)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(-1, 81)
        x = self.relu(self.lin_bn(self.lin1(x)))
        return self.tanh(self.lin2(x))

#v0.2 Policy Net 
class PolicyNet_v2(nn.Module):
    def __init__(self):
        super(PolicyNet_v2, self).__init__()
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

class Conv2dUntiedBias(nn.Module):
    def __init__(self, height, width, 
                in_channels, out_channels, 
                kernel_size, stride=1, 
                padding=0, dilation=1, 
                groups=1):
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
    plyr = np.expand_dims(game.to_numpy(), 0)
    oppt = np.copy(plyr) 
    turn_num = (1 if game.turn%2 == 0 else -1)
    color = (go.BLACK if turn_num == 1 else go.WHITE)
    plyr[plyr != turn_num] = 0
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

    legal_list = game.get_legal_moves()
    legal = np.zeros(81)
    legal[legal_list] = 1

    libs = np.array(game.get_liberties()).reshape(9,9)
    libs_after = np.zeros(81)
    caps = np.zeros(81)
    
    for sq_c in legal_list: 
        new_board, opp_captured = go.get_caps(go.place_stone(color, game.board,sq_c), sq_c, color)
        if opp_captured:
            libs_after[sq_c] = go.get_stone_lib(new_board, sq_c)
            caps[sq_c] = len(opp_captured)
        else:
            libs_after[sq_c] = go.get_stone_lib(go.place_stone(color, game.board, sq_c), sq_c) 
            
    libs_after = libs_after.reshape(9,9)
    caps = caps.reshape(9,9)
    legal = legal.reshape(1,9,9)

    def separate(arr):
        out = np.zeros((7,9,9), dtype = float)
        for i in range(6):
            out[i, arr == i+1] = i+1
        out[6, arr >6] = 7
        return out 

    fts = np.vstack( [plyr, oppt, empty, turn, last_mv, legal,\
            separate(libs) , separate(libs_after) , separate(caps)])
    return torch.from_numpy(fts).float()


def policy_dist(policy: PolicyNet,
                game: go.Game,
                device = torch.device("cpu"),
                fts: torch.Tensor=None):
    '''Return torch.distribution.Categorial distribution over coordinates'''
    if fts is None:
        fts = features(game)
    fts = fts.unsqueeze(0).to(device)
    probs = SOFT(policy(fts)).squeeze(0)
    dist = Categorical(probs)
    return dist

def value(v: ValueNet,
          game: go.Game,
          device = torch.device("cpu"),
          fts: torch.Tensor=None):
    if fts is None:
        fts = features(game)
    fts = fts.unsqueeze(0).to(device)
    return v(fts).item()

def policy_sample(policy: PolicyNet,
                  game: go.Game,
                  device = torch.device("cpu"),
                  fts: torch.Tensor=None):
    '''sample a move from policy distribution. Use policy_dist
    for multiple samplings'''
    if fts is None:
        fts = features(game)
    fts = fts.unsqueeze(0).to(device)
    probs = SOFT(policy(fts)).squeeze(0)
    m = Categorical(probs)
    return m.sample()

class NinebyNineGames(Dataset):
    def __init__(self, path, **kwargs):
        '''Load and process data for training policy or value net.
        args:
            path: path to csv or numpy zipped file (.npz) 
                  csv must have columns (board, ko, turn, move) or (board, ko, turn, val).
                  npz must have "features" with size (n,27,9,9) 
                  and either "vals" or "moves" with size (n, 1)
        kwargs:
            out_path: path to save input/target tensors (default "./data.npz") 
            transform: callable to apply to the data
        '''
        self.out_path = kwargs.get("out_path", os.path.join(os.getcwd(), "data"))
        self.vals = False
        if path.endswith(".csv"):
            self.inputs, self.targets = process_csv(path, out_path)
        elif path.endswith(".npz"):
            load = np.load(path)
            self.inputs = torch.from_numpy(load["features"]).float()
            if "vals" in load.files:
                self.targets = load["vals"]
                self.vals = True
            else:
                self.targets = load["moves"]
            self.targets = torch.from_numpy(self.targets).float()
            assert self.inputs[0,].shape == (27,9,9) 
            assert self.targets[0,].shape == (1,)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform:
            if self.vals:
                return self.transform(self.inputs[idx]), self.targets[idx]
            else:
                return self.transform(self.inputs[idx], self.targets[idx])
        return self.inputs[idx], self.targets[idx]

def rand_refl(features, moves = None):
    do_refl = randint(1)
    if do_refl:
        features = torch.transpose(features,2,3)
        if moves != None:
            x,y = moves//9, moves%9
            moves = 9*y + x
    if moves is None:
        return features
    return features, moves 
         
def rand_rot(features, moves = None):
    do_rot = randint(3) 
    if do_rot:
        features = torch.rot90(features, do_rot,[3,2])
        if moves != None:
            moves = (moves*9+8-moves//9)%81 
    if moves is None:
        return features
    return features, moves
        
def compose(*args):
    def compose_two(f,g):
        return lambda x: f(g(x))
    return reduce(compose, arg, lambda x: x)

def process_csv(path, npz_name):
    cols = pd.read_csv(path, nrows = 0).columns
    convert = {col: lambda x: eval(x) for col in cols}
    convert["board"] = lambda x: x
    boards = pd.read_csv(path, converters = convert, low_memory = False)
    print(f"Processing features from {path}...")
    fts = np.zeros(shape = (len(boards),27,9,9), dtype = np.int8)
    targets = np.zeros(shape = (len(boards),1), dtype = np.int8)
    for i in trange(len(boards)):
        board, ko, last, target = boards.iloc[i]
        g = go.Game(board, ko, last)
        g.turn = 1 if g.board[last] == go.BLACK else 0
        fts[i] = features(g)
        if cols[-1] == "val":
            targets[i] = -1 if target else 1
        elif cols[-1] == "move":
            targets[i] = target
    np.savez_compressed(npz_name, features = fts, targets = targets)
     
    
