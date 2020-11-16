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
        if mv1 is None:
            break 
        else:
            game.play_move(mv1.item())
        mv2 = legal_sample(pi_2, game, device)        
        if mv2 is None:
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
            return
    return move

def write_sgf(moves, out_path, B = None, W = None, result = None):
    '''Write minimal sgf for moves list to outpath'''
    out = f"(;GM[1]RU[Chinese]"
    if B and W:
        out += f"PB[{B}]PW[{W}]"
    if result:
        out += f"RE[{result}]"
    out += "SZ[9]KM[5.5]\n"
    turn = "B"
    for mv in moves:
        x, y = chr(mv//9 + 97), chr(mv%9 +97)
        out += f";{turn}[{x}{y}]\n"
        turn = "W" if turn == "B" else "B" 
    out += ")" 
    with open(out_path, 'w') as f:
        f.write(out)

def gnu_score(game):
    '''Scores the game using gnugo opened in a subprocess'''
    temp = "/tmp/" + str(os.getpid())+".sgf" #put temp directory here
    write_sgf(game.moves, temp) 
    p =Popen(["gnugo", "--komi", "5.5", "--mode", "gtp", "--chinese-rules", "-l", temp], \
                    stdin = PIPE, stdout = PIPE)
    p.stdin.write("final_score\n".encode('utf-8'))
    p.stdin.flush()
    rec = p.stdout.readline().decode('utf-8').strip('\n')
    p.communicate("quit\n".encode('utf-8'))
    os.remove(temp)
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

def reinforce(pi, pi_opp, optimizer, batch_size = 32, n_itrs = 1, device = DEV):
    '''Implement the REINFORCE policy gradient descent algorithm'''
    winlist = []
    for itr in trange(n_itrs):
        wins = 0 
        games, results = self_play(pi, pi_opp, batch_size, device = DEV)
        for i in range(batch_size):
            loss = 0.0
            g = go.Game(moves = games[i])
            #training policy plays black 
            reward = 1 if results[i] == 1 else -1
            for j in range(0,len(g),2):
                dist = policy_dist(pi, g, DEV)
                mv = torch.tensor(g.moves[j]).to(device)
                loss += -dist.log_prob(mv)*reward
                try:
                    g.play_move()
                    g.play_move()
                except go.IllegalMove:
                    break
            wins += results[i]

        winlist.append(wins)
        if len(winlist)%11 == 10:
            avg_win = sum(winlist[-10:])/(batch_size*10)
            print(f"Winrate: {avg_win}")

        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":

    mp.set_start_method("spawn")
    epochs = 10
    pi = PolicyNet()
    optimizer = torch.optim.Adam(pi.parameters())
    checkpt = torch.load("v0.2/RL_policy_13.pt",map_location = DEV) 
    optimizer.load_state_dict(checkpt["optimizer_state_dict"])

    for state in optimizer.state.values():
        for k, t in state.items():
            if torch.is_tensor(t):
                state[k] = t.cuda()

    pi.load_state_dict(checkpt["model_state_dict"])
    pi.to(DEV)
    pi.train()
    pi.share_memory()
    n_opps = 23 

    for _ in range(epochs):
        pi_opp = PolicyNet()
        #choose a random opponent from the previous policies
        n = randint(1,n_opps)
        opp_checkpt = torch.load(f"v0.2/RL_policy_{n}.pt", map_location = DEV)
        print(f"Playing against Policy {n}")
        pi_opp.load_state_dict(opp_checkpt["model_state_dict"])
        pi_opp.to(DEV)
        pi_opp.eval()
        
        processes = []
        for _ in range(8):
            p = mp.Process( target = reinforce, args = (pi, pi_opp, optimizer, 16, 60))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

        n_opps += 1
        out_path = os.getcwd() + f"/v0.2/RL_policy_{n_opps}.pt"
        torch.save({"model_state_dict":pi.state_dict(), "optimizer_state_dict":optimizer.state_dict()}, out_path)
