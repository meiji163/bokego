import go
import os
import re
import argparse
from glob import glob
from tqdm import trange
from numpy.random import randint
from copy import deepcopy
from bokeNet import PolicyNet, policy_sample, policy_dist
from subprocess import Popen, PIPE
import multiprocessing as mp
import torch
from torch.distributions.categorical import Categorical

DEV= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_TURNS = 85

def playout(game: go.Game, pi_1, pi_2, device = DEV):
    '''Playout game between policies pi_1 and pi_2, with pi_1 playing black and pi_2 playing white.''' 
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

def legal_sample(pi, game: go.Game, device = DEV):
    move = policy_sample(pi, game, device)
    tries = 0
    while not game.is_legal(move.item()) or go.possible_eye(game.board, move.item()):
        move = policy_sample(pi, game, device )
        tries += 1
        if tries > 10:
            #abort if policy selects no reasonable move
            return
    return move

def write_board_sgf(game: go.Game, out_path):
    '''write board to sgf (move sequence not available)'''
    out = "(;GM[1]RU[Chinese]HA[0]SZ[9]KM[5.5]\n"
    W = "AW"
    B = "AB"
    for i in range(81): 
        mv = game.board[i]
        if mv == 'X':
            x, y = chr(i//9 + 97), chr(i%9 +97)
            B += f"[{x}{y}]" 
        elif mv == 'O':
            x, y = chr(i//9 + 97), chr(i%9 +97)
            W += f"[{x}{y}]" 
    turn = 'B' if game.turn%2 == 0 else 'W' 
    out += B + '\n' + W + f"PL[{turn}])"
    with open(out_path, 'w') as f:
        f.write(out)

def write_sgf(moves, out_path, **kwargs): 
    '''Write minimal sgf for moves list
    args:
        moves: list of moves in squashed coordinates (0 - 80), pass = -1
        out_path: path to write to
    kwargs:
        B: name of black player
        W: name of white player
        result: result of game (e.g. "B+2.5")
        '''
    B = kwargs.get('B', '')
    W = kwargs.get('W', '')
    result = kwargs.get('result', '') 
    out = f"(;GM[1]RU[Chinese]"
    if B and W:
        out += f"PB[{B}]PW[{W}]"
    if result:
        out += f"RE[{result}]"
    out += "SZ[9]KM[5.5]\n"
    turn = "B"
    for mv in moves:
        if mv == -1:
            out += f";{turn}[]\n"
        else:
            x, y = chr(mv//9 + 97), chr(mv%9 +97)
            out += f";{turn}[{x}{y}]\n"
        turn = "W" if turn == "B" else "B" 
    out += ")" 
    with open(out_path, 'w') as f:
        f.write(out)

def gnu_score(game):
    '''Scores the game using gnugo opened in a subprocess.
    Return 1 if black won, 0 if white won'''
    temp = "/tmp/" + str(os.getpid())+".sgf" #put temp directory here
    write_board_sgf(game, temp) 
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
        device: torch.device for pi and pi_opp (default DEV) 
        stats: list to write winrate stats to
        '''
    n_itrs = kwargs.get("n_itrs", 60)
    bs = kwargs.get("bs", 16)
    device = kwargs.get("device", DEV)
    stats = kwargs.get("stats")

    winlist = []
    for itr in trange(n_itrs):
        if train_color == "black":
            games, results = self_play(pi, pi_opp, bs, device = DEV)
        elif train_color == "white":
            games, results = self_play(pi_opp, pi, bs, device = DEV)
        else:
            raise ValueError("train_color must be black or white")

        wins = 0 
        for i in range(bs):
            loss = 0.0
            g = go.Game(moves = games[i])
            reward = 1 if (results[i] and train_color == "black")\
                        or (not results[i] and train_color == "white")  else -1
            #replay the game to calculate the loss
            if train_color == "white":
                g.play_move()
            for j in range(g.turn,len(g),2):
                dist = policy_dist(pi, g, DEV)
                mv = torch.tensor(g.moves[j]).to(device)
                loss += -dist.log_prob(mv)*reward
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

        if len(winlist)>0 and len(winlist)%12 == 0:
            avg_win = sum(winlist[-10:])/(bs*10)
            print(f"Winrate ({train_color}): {avg_win:.2f}")

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
    parser.add_argument("-f", help = "file to write stats to", metavar = "PATH", type = str, dest = 'f', default = "v0.2/RL_stats.txt")
    args = parser.parse_args()

    mp.set_start_method("spawn")
    pi = PolicyNet()
    optimizer = torch.optim.Adam(pi.parameters())

    policy_paths = glob("v0.2/RL_policy_*.pt")
    n_opps = len(policy_paths) - 1
    print(f"Opponent pool size: {n_opps}")
    checkpt = torch.load(f"v0.2/RL_policy_{n_opps}.pt",map_location = DEV) 
    optimizer.load_state_dict(checkpt["optimizer_state_dict"])
    #move optimizer states to GPU
    for state in optimizer.state.values():
        for k, t in state.items():
            if torch.is_tensor(t):
                state[k] = t.cuda()
    pi.load_state_dict(checkpt["model_state_dict"])
    pi.to(DEV)
    pi.train()
    pi.share_memory()

    for epoch in range(args.e):
        print(f"Epoch: {epoch +1}") 
        pi_opp = PolicyNet()
        #choose a random opponent from the previous policies
        opp_id = randint(n_opps+1)
        opp_checkpt = torch.load(f"v0.2/RL_policy_{opp_id}.pt", map_location = DEV)
        print(f"Playing against Policy {opp_id}")
        pi_opp.load_state_dict(opp_checkpt["model_state_dict"])
        pi_opp.to(DEV)
        pi_opp.eval()
        
        n_workers = mp.cpu_count()
        processes = []
        manager = mp.Manager()
        stat_list = manager.list()

        #half of workers train black, half train white
        for _ in range(n_workers//2):
            keywords = {"n_itrs": args.n, "bs": args.b, "stats": stat_list}
            p_b = mp.Process(target = reinforce, args = (pi, pi_opp, optimizer, "black"), kwargs = keywords)
            p_w = mp.Process(target = reinforce, args = (pi, pi_opp, optimizer, "white"), kwargs = keywords)
            p_b.start()
            p_w.start()
            processes.append(p_b)
            processes.append(p_w)
        for p in processes:
            p.join()
        
        with open(args.f, 'a+') as f:
            f.write(f"\n{args.b*args.n} games v. Policy {opp_id}")
            f.write(','.join([str(w) for w in stat_list]))
            
        n_opps += 1
        out_path = os.getcwd() + f"/v0.2/RL_policy_{n_opps}.pt"
        torch.save({"model_state_dict":pi.state_dict(), "optimizer_state_dict":optimizer.state_dict()}, out_path)
