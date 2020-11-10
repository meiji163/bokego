import go
import os
import re
import time
from numpy.random import randint
from copy import copy
from bokeNet import *
from subprocess import Popen, PIPE
import multiprocessing as mp
import torch
from torch.distributions.categorical import Categorical

def playout(game, pi_1, pi_2, device = "cpu", get_dist = []):
    '''Play game between policies pi_1 and pi_2, with pi_1 playing black and pi_2 playing white.''' 
    dists = []
    while True:
        if game.turn > 70:
            break
        if game.turn in get_dist:
            dist = policy_dist(pi_1, game, device = device)
            mv = dist.sample()
            dists.append(dists)
        mv1 = legal_sample(pi_1, game, device)
        if not mv1:
            break 
        else:
            game.play_move(mv1.item())
        mv2 = legal_sample(pi_2, game, device)        
        if not mv2:
           break 
        else:
            game.play_move(mv2.item())
    return game 

def legal_sample(pi, game, device):
    move = policy_sample(pi, game, device)
    tries = 0
    while not game.is_legal(move.item()):
        move = policy_sample(pi, game, device )
        tries += 1
        if tries > 1000:
            print(go.unsquash(move.item(), alph = True))
            return
    return move

def write_sgf(game, black, white, out_path):
    out = f"(;GM[1]RU[Chinese]SZ[9]KM[5.5]PW[{white}]PB[{black}]\n"
    turn = "B"
    for mv in game.moves:
        x, y = chr(mv//9 + 97), chr(mv%9 +97)
        out += f";{turn}[{x}{y}]\n"
        turn = "W" if turn == "B" else "B"
    out += ")"
    with open(out_path, 'w') as f:
        f.write(out)

def gnu_score(sgf):
    p =Popen(["gnugo", "--mode", "gtp", "--chinese-rules", "-l", sgf], stdin = PIPE, stdout = PIPE)
    p.stdin.write("final_score\n".encode('utf-8'))
    p.stdin.flush()
    rec = p.stdout.readline().decode('utf-8').strip('\n')
    p.communicate("quit\n".encode('utf-8'))
    score = re.search("[BW]\+.+",rec)
    if score:
        return score[0]
    return 

def selfPlayGame(pi, num_games, device):
    for n in range(num_games):
        g = go.Game(moves = [], turn = 0, last_move = None)
        playout(g, pi, pi, device)        
        out_path = f"sp_{n}_{os.getpid()}.sgf"
        write_sgf(g, "boke", "boke", out_path)
        
if __name__ == "__main__":
    pi = PolicyNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pi.to(device)
    checkpt = torch.load("v0.2/policy_v0.2_2020-11-07_1.pt",map_location = device) 
    pi.load_state_dict(checkpt["model_state_dict"])
    optimizer = torch.optim.Adam(pi.parameters(), lr = 0.001)
    pi_opp = pi.copy()
    games = 100
    rand = randint(2, 35, 4)
    for i in range(games):
        #play a game and update based on random move index
        #opponent policy plays black
        g = go.Game(moves = [])
        g.play_move(
        g.play_move(mv)
        with torch.no_grad():
            res = playout(pi_opp, pi ,g) 
        reward = (-1 if res else 1) #+1 if W won, -1 if W lost

        #Policy gradient descent
        optimizer.zero_grad()
        loss = -m.log_prob(mv) * reward 
        loss.backward()
        optimizer.step()
