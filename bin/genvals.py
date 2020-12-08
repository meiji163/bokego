from os import path
import torch.multiprocessing as mp
import bokego.go as go
from bokego.nnet import policy_sample, PolicyNet
from selfplay import legal_sample
import torch
from tqdm import trange
import numpy as np
import argparse

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TURNS = 90

ENC = {go.EMPTY: 0, go.BLACK: 1, go.WHITE: -1}
DEC = {0: go.EMPTY, 1: go.BLACK, -1: go.WHITE}

def gen_game(pi_SL, pi_RL, data_list, n_games = 1):
    for _ in trange(n_games):
        np.random.seed()
        g = go.Game()
        r = np.random.randint(70,90)
        for _ in range(r):
            mv = legal_sample(pi_SL, g, DEV)
            if mv is None:
                return
            g.play_move(mv)
        rand_mv = np.random.randint(81)
        while not g.is_legal(rand_mv):
            rand_mv = np.random.randint(81)
        g.play_move(rand_mv)
        board, move, ko = g.board, rand_mv, g.ko

        while g.turn < MAX_TURNS:
            mv = legal_sample(pi_RL, g, DEV)     
            if mv is None:
                break
            g.play_move(mv)
        result = go.gnu_score(g)

        # 1 if player who played rand move won, else 0
        val = 1 if (result and r%2 == 0) or ( not result and r%2 == 1) else 0
        data_list.append( (board, ko, move, val))

def data_str(board, ko, last, result):
    return ','.join([board, str(ko), str(last), str(result)]) +'\n'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "generate data for value net")
    parser.add_argument("-n", metavar = "N", type = int, dest = 'n', help = "number of games to generate in thousands", default = 1)
    parser.add_argument("-o", metavar = "PATH", type = str, dest = 'o', help = "output file", required = True) 
    args = parser.parse_args()

    pi_SL = PolicyNet_v2()
    pi_RL = PolicyNet_v2()
    pi_SL, pi_RL = pi_SL.to(DEV), pi_RL.to(DEV)
    cpt_SL = torch.load("v0.2/RL_policy_0.pt", map_location = DEV)
    cpt_RL = torch.load("v0.2/RL_policy_50.pt", map_location = DEV)
    pi_SL.load_state_dict(cpt_SL["model_state_dict"] )
    pi_RL.load_state_dict(cpt_RL["model_state_dict"])
    mp.set_start_method("spawn")
    manager = mp.Manager()
    data_list = manager.list()
    processes = []
    n_workers = mp.cpu_count() 
    for _ in range(args.n):
        for _ in range(n_workers):
            p = mp.Process(target = gen_game, args = (pi_SL, pi_RL, data_list, 500))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    
        with open(args.o, 'a+') as f:
            if not path.exists(args.o):
                f.write("board,last,ko,val\n")
            for board, ko, last, val in data_list:
                for _ in range(4):
                    f.write(data_str(board, ko, last, val))
                    f.write(data_str(refl(board), refl(ko), refl(last), val))
                    board, ko, last = rot(board), rot(ko), rot(last)
         
