import go
from path import os
import multiprocessing as mp
from bokeNet import policy_sample, PolicyNet
from selfplay import legal_sample, gnu_score
from tqdm import trange
import torch
import numpy as np

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TURNS = 90 

ENC = {go.EMPTY: 0, go.BLACK: 1, go.WHITE: -1}
DEC = {0: go.EMPTY, 1: go.BLACK, -1: go.WHITE}

def gen_game(pi_SL, pi_RL, data_list, n_games = 1):
    for _ in trange(n_games):
        g = go.Game()
        np.random.seed()
        r = np.random.randint(60,80)
        for _ in range(r):
            mv = legal_sample(pi_SL, g, DEV)
            if mv is None:
                return
            g.play_move(mv)
            #print(g)
        rand_mv = np.random.randint(81)
        while not g.is_legal(rand_mv):
            rand_mv = np.random.randint(81)
        #print(f"Random move: {go.unsquash(rand_mv, alph = True)}"
        g.play_move(rand_mv)
        #print(g)
        board, move, ko = g.board, rand_mv, g.ko

        while g.turn < MAX_TURNS:
            mv = legal_sample(pi_RL, g, DEV)     
            if mv is None:
                break
            g.play_move(mv)
            #print(g)
        result = gnu_score(g)
        #print(f"result : {result}")
        # 1 if player who played rand move won, else 0
        val = 1 if (result and r%2 == 0) or ( not result and r%2 == 1) else 0
        data_list.append( (board, ko, move, val))

def data_str(board, ko, last, result):
    return ','.join([board, str(ko), str(last), str(result)]) +'\n'

def rot(b):
    if b is None:
        return
    if isinstance(b, str):
        A = np.rot90(np.array([ENC[c] for c in b]).reshape(9,9), k = 3).reshape(81).tolist()
        return ''.join([ DEC[n] for n in A])
    elif isinstance(b, int):
        return (b*9 +8 - b//9)%81

def refl(b):
    if b is None:
        return
    if isinstance(b, str):
        A = (np.array([ENC[c] for c in b]).reshape(9,9).T).reshape(81).tolist()
        return ''.join([DEC[n] for n in A])
    elif isinstance(b, int):
        x, y = divmod(b, 9)
        return 9*y + x


if __name__ == "__main__":
    pi_SL = PolicyNet()
    pi_RL = PolicyNet()
    pi_SL, pi_RL = pi_SL.to(DEV), pi_RL.to(DEV)
    cpt_SL = torch.load("v0.2/RL_policy_0.pt", map_location = DEV)
    cpt_RL = torch.load("v0.2/RL_policy_50.pt", map_location = DEV)
    pi_SL.load_state_dict(cpt_SL["model_state_dict"] )
    pi_RL.load_state_dict(cpt_RL["model_state_dict"])
    mp.set_start_method("spawn")
    manager = mp.Manager()
    data_list = manager.list()
    n_workers = mp.cpu_count()  
    for _ in range(5):
        processes = []
        for _ in range(2*n_workers):
            p = mp.Process(target = gen_game, args = (pi_SL, pi_RL, data_list, 500))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        with open("score_me.csv", 'a+') as f:
            #f.write("board,last,ko,val\n")
            for board, ko, last, val in data_list:
                for _ in range(4):
                    f.write(data_str(board, ko, last, val))
                    f.write(data_str(refl(board), refl(ko), refl(last), val))
                    board, ko, last = rot(board), rot(ko), rot(last)
         
