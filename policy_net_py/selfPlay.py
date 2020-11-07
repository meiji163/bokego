import go
import os
import time
from copy import copy
from bokePolicy import *
import multiprocessing as mp
import torch
from torch.distributions.categorical import Categorical

def playout(pi_1, pi_2, device):
    '''Play game between policies pi_1 and pi_2, with pi_1 playing black and pi_2 playing white.''' 
    game = go.Game()
    while True:
        mv1 = legal_sample(pi_1, game, device)
        if not mv1:
            break 
        else:
            game.play_move(mv1)
        mv2 = legal_sample(pi_2, game, device)        
        if not mv2:
           break 
        else:
            game.play_move(mv2)
    return game 

def legal_sample(pi, game, device):
    move = policy_sample(pi, game, device)
    tries = 0
    while not game.is_legal(move):
        move = policy_sample(pi, game, device )
        tries += 1
        if tries > 100:
            return
    return move

def write_sgf(game, black, white, result, out_path):
    out = f"(;GM[1]RU[Chinese]SZ[9]KM[5.5]PW[{white}]PB[{black}]RE[{result}]\n"
    turn = "B"
    for mv in game.moves:
        x, y = chr(mv//9 + 97), chr(mv%9 +97)
        out += f";{turn}[{x}{y}]\n"
        turn = "W" if turn == "B" else "B"
    out += ")"
    with open(out_path, 'w') as f:
        f.write(out)

def selfplay(pi, n_games, device ):
    for i in range(n_games):
        game = playout(pi,pi, device) 
        score = game.score()
        if -20 < score < 20:
            if score > 0:
                res = f"B+{score}"
            else:
                res = f"W+{-score}"
            out_path = str(i) + "_" + str(os.getpid()) + ".sgf"
            write_sgf(game, "boke", "boke", res, out_path)
        
        
if __name__ == "__main__":
    pi = PolicyNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pi.to(device)
    checkpt = torch.load("v0.2/policy_v0.2_2020-11-07_1.pt",map_location = device) 
    pi.load_state_dict(checkpt["model_state_dict"])
    mp.set_start_method("spawn")
    with torch.no_grad(): 
        processes = []
        for i in range(16):
            p = mp.Process(target = selfplay, args = (pi, 100, device,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
    #optimizer = torch.optim.Adam(pi.parameters(), lr = 0.001)

    #games = 100
    #for i in range(games):
    #    g = go.Game()
    #    for _ in range(35):

    #        #opponent policy plays black
    #        g.play_move(sample(pi_opp,g))
    #        mv, m = sample(pi, g, return_dist = True)
    #        g.play_move(mv)
    #        with torch.no_grad():
    #            res = playout(pi_opp, pi ,g) 
    #        reward = (-1 if res else 1) #+1 if W won, -1 if W lost

    #        #Policy gradient descent
    #        optimizer.zero_grad()
    #        loss = -m.log_prob(mv) * reward 
    #        loss.backward()
    #        optimizer.step()
    

    
