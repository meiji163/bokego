#!/usr/bin/env python
import bokego.go as go
from bokego.nnet import PolicyNet, policy_sample, policy_dist
import os
import re
import argparse
from glob import glob
from tqdm import trange
from numpy.random import randint
from copy import deepcopy
import torch.multiprocessing as mp
import torch
from torch.distributions.categorical import Categorical

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_TURNS = 70

def playout(game: go.Game, pi_1, pi_2, device = DEVICE):
    '''Playout game between policies pi_1 and pi_2, 
       with pi_1 playing first''' 
    while True:
        if game.turn > MAX_TURNS:
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

def legal_sample(pi, game: go.Game, device = DEVICE):
    move = policy_sample(pi, game, device)
    tries = 0
    color = go.BLACK if game.turn%2 == 0 else go.WHITE
    k = 0
    while not game.is_legal(move.item()):
        if k == 0:
            moves = torch.topk(policy_dist(pi, game, device).probs, k = 81).indices
        elif k > 80:
            return
        move = moves[k]
        k += 1
    return move

def self_play(pi1, pi2, num_games, device = DEVICE):
    games = []
    results = []
    for n in range(num_games):
        g = go.Game(moves = [])
        playout(g, pi1, pi2, device)        
        games.append(g.moves)
        results.append(go.gnu_score(g))
    return games, results

def reinforce(pi, pi_opp, optimizer, train_color, **kwargs):
    '''Implements the REINFORCE policy gradient descent algorithm using selfplay
    args:
        pi: training PolicyNet
        pi_opp: opponent PolicyNet
        optimizer: torch.optimizer for pi
        train_color: color pi plays -- "black" or "white"
    kwargs:
        n_itrs: number of iterations to train (default 60)
        bs: batch size of each iteration (default 16)
        device: torch.device for pi and pi_opp 
        stats: list to write winrate stats to
        id: identifier for process
    '''
    n_itrs = kwargs.get("n_itrs", 60)
    bs = kwargs.get("bs", 16)
    device = kwargs.get("device", DEV)
    stats = kwargs.get("stats")
    idn = kwargs.get("id", '')

    winlist = []
    for itr in trange(n_itrs):
        if train_color == "black":
            games, results = self_play(pi, pi_opp, bs, device = DEVICE)
        elif train_color == "white":
            games, results = self_play(pi_opp, pi, bs, device = DEVICE)
        else:
            raise ValueError("train_color must be black or white")

        wins = 0 
        for i in range(bs):
            loss = 0.0 
            g = go.Game(moves = games[i])
            if len(g) < MAX_TURNS - 5:
                print(len(g))
            reward = -results[i] if train_color == "white" else results[i]
            #replay the game to calculate the loss
            if train_color == "white":
                g.play_move()
            for j in range(g.turn,len(g),2):
                dist = policy_dist(pi, g, DEV)
                mv = torch.tensor(g.moves[j]).to(device)
                loss += -dist.log_prob(mv)
                try:
                    g.play_move()
                    g.play_move()
                except go.IllegalMove:
                    break

            if train_color == "black":
                wins += results[i]
            else:
                wins += not results[i]
        winlist.append(wins)

        if len(winlist)>0 and len(winlist)%10 == 0:
            avg_win = sum(winlist[-10:])/(bs*10)
            print(f"Winrate ({train_color}{id}): {avg_win:.2f}")

        loss *= reward
        loss /= bs 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    stats.extend(winlist)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Script for self-play training")
    parser.add_argument("-e", help = "number of epochs", metavar = "E", type = int, dest = 'e', default = 1)
    parser.add_argument("-b", help = "batch size", metavar = "B", type = int, dest = 'b', default = 16)
    parser.add_argument("-n", help = "number of iterations per epoch", metavar = "N", type = int, dest = 'n', default = 64)
    parser.add_argument("-f", help = "file to write stats to", metavar = "PATH", type = str, 
                        dest = 'f', default = os.path.join(os.getcwd(), "RL_stats.txt"))
    parser.add_argument("-w", help = "path to look for weights", metavar = "PATH", type = str, dest = 'w',
                        default = os.path.abspath(os.path.join("..", "bokego", "data", "weights")))
    args = parser.parse_args()

    mp.set_start_method("spawn")
    pi = PolicyNet()
    optimizer = torch.optim.AdamW(pi.parameters(), lr = 1e-5)

    policy_paths = glob( os.path.join( args.w, "policy_*.pt"))
    n_opps = len(policy_paths) - 1
    print(f"Opponent pool size: {n_opps}")

    checkpt = torch.load(os.path.join(args.w, f"policy_{n_opps}.pt"), 
                          map_location = DEVICE) 

    #move optimizer states to GPU
    if "optimizer_state_dict" in checkpt and DEVICE.type == "cuda":
        optimizer.load_state_dict(checkpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, t in state.items():
                if torch.is_tensor(t):
                    state[k] = t.cuda()

    pi.load_state_dict(checkpt["model_state_dict"])
    pi.to(DEVICE)
    pi.train()
    pi.share_memory()

    for epoch in range(args.e):
        print(f"Epoch: {epoch +1}") 
        pi_opp = PolicyNet()

        #choose a random opponent from the previous policies
        #play policy_0 (SL policy) first
        opp_id = 0 
        while not os.path.exists(os.path.join(args.w, f"policy_{opp_id}.pt")):
            print(f"Policy {opp_id} not found")
            opp_id = randint(n_opps+1)
 
        opp_checkpt = torch.load( os.path.join( args.w, f"policy_{opp_id}.pt"),
                                    map_location = DEVICE)
        print(f"Playing against Policy {opp_id}")
        pi_opp.load_state_dict(opp_checkpt["model_state_dict"])
        pi_opp.to(DEVICE)
        pi_opp.eval()
        
        n_workers = mp.cpu_count()
        processes = []
        manager = mp.Manager()
        stat_list = manager.list()

        #half of workers train black, half train white
        for i in range(n_workers//2):
            keywords = {"id": i, "n_itrs": args.n, "bs": args.b, "stats": stat_list}
            p_b = mp.Process(target = reinforce, 
                            args = (pi, pi_opp, optimizer, "black"), 
                            kwargs = keywords)
            keywords["id"] += 1
            p_w = mp.Process(target = reinforce, 
                            args = (pi, pi_opp, optimizer, "white"), 
                            kwargs = keywords)
            p_b.start()
            p_w.start()
            processes.append(p_b)
            processes.append(p_w)
        for p in processes:
            p.join()
        
        with open(args.f, 'a+') as f:
            f.write(f"Policy {n_opps} vs. Policy {opp_id}\n")
            f.write(f"Batch Size: {args.b}, Iterations: {args.n}\n")
            f.write(','.join([str(w) for w in stat_list]) + '\n')
            
        n_opps += 1
        out_path = os.path.join( args.w, f"policy_{n_opps}.pt") 
        torch.save({"model_state_dict":pi.state_dict(), "optimizer_state_dict":optimizer.state_dict()}, out_path)
