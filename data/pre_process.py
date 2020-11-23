#!/usr/bin/python3
import sys
import os
import re
import argparse
import numpy as np
from tqdm import tqdm 
#path to go.py  
sys.path.append(r"/home/jupyter/BokeGo/python")
import go


parser = argparse.ArgumentParser()
parser.add_argument("-i", type = str, required = True, metavar = "INPATH", nargs = 1, help = "input directory")
parser.add_argument("-o", type = str, metavar = "OUTPATH", required = True, nargs = 1, help = "output directory")
args = parser.parse_args()

def pre_process(root_dir, target_dir):
    sgf_files = [ s for s in os.scandir(root_dir) if s.path.endswith(".sgf")]
    with open(target_dir, 'w') as f:
        f.write("board,ko,turn,last,move\n")
        for sgf in sgf_files: 
            #result = get_result(sgf)
            #if not result:
            #    continue
            mvs = get_moves(sgf)
            if len(mvs) < 10:
                continue
            g = go.Game(moves = [], turn =0 , last_move = None)
            #rand = np.random.randint(2, len(mvs),size = 5)
            for i in range(len(mvs)-1):
                if mvs[i] != -1:
                    last = None if i == 0 else g.last_move
                    board, ko, move = g.board, g.ko, mvs[i]
                    for k in range(4):
                        f.write(data_str(board, ko, i, last, move))
                        f.write(data_str(refl(board), refl(ko), i, refl(last), refl(move)))
                        board, ko, last , move= rot(board), rot(ko), rot(last), rot(move)
                g.play_move(mvs[i])
        print(go.unsquash(g.moves))
enc  = {go.EMPTY : 0, go.BLACK: 1, go.WHITE: -1}
dec = {0: go.EMPTY, 1: go.BLACK, -1: go.WHITE}

def data_str(board, ko , mv_num, last, move):
    return ','.join([board, str(ko), str(mv_num), str(last), str(move)+'\n']) 

def rot(b):
    #rotates 90 deg clockwise
    if b is None:
        return
    if isinstance(b, str):
        A = np.array([enc[c] for c in b]).reshape((9,9))
        B = np.rot90(A, k = 3).reshape(81).tolist()
        return ''.join([ dec[n] for n in B])
    elif isinstance(b, int):
        if b == -1:
            return -1
        return (b*9 + 8 - b//9)%81
    
def refl(b):
    if b is None:
        return
    if isinstance(b, str):
        A = np.array([enc[c] for c in b]).reshape((9,9))
        B = (A.T).reshape(81).tolist()
        return ''.join([dec[n] for n in B])
    elif isinstance(b, int):
        if b == -1:
            return -1
        x, y = divmod(b, 9)
        b = 9*y + x
        return b

def get_result(sgf):
    with open(sgf, 'r') as f:
        match = re.findall(r"RE\[(.*)\]", f.read())
    if match: 
        return 'B' if 'B' in match[0] else 'W'
    else:
        return None 

def get_moves(sgf):
    with open(sgf, 'r') as f:
        match = re.findall(r"\[([a-z]*)\]", f.read())
    mvs = []
    for mv in match:
        if len(mv) == 0:
            mvs.append(-1) 
        elif len(mv) == 2: 
            mvs.append(9*(ord(mv[0])-97) + ord(mv[1])-97 )
    return mvs

if __name__ == "__main__":
    pre_process(args.i[0], args.o[0])
