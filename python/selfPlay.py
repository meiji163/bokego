import go
import os
import re
import time
from tqdm import trange
from numpy.random import randint
from copy import deepcopy
from bokeNet import *
from subprocess import Popen, PIPE
import multiprocessing as mp
import torch
from torch.distributions.categorical import Categorical

DEV= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def playout(game, pi_1, pi_2, device = DEV):
    '''Play game between policies pi_1 and pi_2, with pi_1 playing black and pi_2 playing white.''' 
    while True:
        if game.turn > 79:
            break
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

def legal_sample(pi, game, device = DEV):
    move = policy_sample(pi, game, device)
    tries = 0
    while not game.is_legal(move.item()):
        move = policy_sample(pi, game, device )
        tries += 1
        if tries > 1500:
            #print(go.unsquash(move.item(), alph = True))
            return
    return move

def write_sgf(game, out_path):
    out = f"(;GM[1]RU[Chinese]SZ[9]KM[5.5]\n"
    turn = "B"
    for mv in game.moves:
        x, y = chr(mv//9 + 97), chr(mv%9 +97)
        out += f";{turn}[{x}{y}]\n"
        turn = "W" if turn == "B" else "B" 
    out += ")" 
    with open(out_path, 'w') as f:
        f.write(out)

def gnu_score(game):
    #temp = os.environ["TMPDIR"] + str(os.getpid()) + ".sgf"
    temp = "temp/" + str(os.getpid())+".sgf"
    write_sgf(game, temp) 
    p =Popen(["gnugo", "--komi", "5.5", "--mode", "gtp", "--chinese-rules", "-l", temp], \
                    stdin = PIPE, stdout = PIPE)
    p.stdin.write("final_score\n".encode('utf-8'))
    p.stdin.flush()
    rec = p.stdout.readline().decode('utf-8').strip('\n')
    p.communicate("quit\n".encode('utf-8'))
    res = re.search("[BW]\+.+",rec)
    if res:
        return 1 if 'B' in res[0] else 0 
    return

def self_play(pi1, pi2, num_games, device = DEV):
    games = []
    results = []
    for n in range(num_games):
        g = go.Game()
        playout(g, pi1, pi2, device)        
        games.append(g.moves)
        results.append(gnu_score(g))
    return games, results

def reinforce(pi, pi_opp, batch_size, n_itrs = 1, device = DEV):
    optimizer = torch.optim.Adam(pi.parameters(), lr = 1e-4)
    winrates = []
    for itr in trange(n_itrs):
        wins = 0 
        games, results = self_play(pi, pi_opp, batch_size, device = DEV)
        for i in range(batch_size):
            loss = 0.0
            g = go.Game(moves = games[i])
            for j in range(0,len(g),2):
                dist = policy_dist(pi, g, DEV)
                mv = torch.tensor(g.moves[j]).to(device)
                loss += -dist.log_prob(mv)
                try:
                    g.play_move()
                    g.play_move()
                except go.IllegalMove:
                    break
            #training policy plays BLACK
            wins += results[i]
            loss *= results[i] 
        if itr>9:
            avg_win = sum(winrates[-10:])/(batch_size*10)
            print(f"Winrate: {avg_win}")
        winrates.append(wins)
        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(winrates)
                
if __name__ == "__main__":
    pi = PolicyNet()
    checkpt = torch.load("v0.2/RL_policy_1.pt",map_location = DEV) 
    pi.load_state_dict(checkpt["model_state_dict"])
    pi_opp = deepcopy(pi)
    pi.to(DEV)
    pi_opp.to(DEV)
    pi.train()
    pi_opp.eval()
 
    reinforce(pi, pi_opp, 16, 300, DEV)
    out_path = os.getcwd() + "/RL_policy_3.pt"
    torch.save({"model_state_dict":pi.state_dict()}, out_path)
