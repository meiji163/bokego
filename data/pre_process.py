#!/usr/bin/python3
import sys
import os
import re
import numpy as np
#path to go.py  
sys.path.append(r"/home/meiji163/Documents/BokeGo/policy_net_py")
import go

def pre_process(root_dir, target_dir):
    out = os.path.join(target_dir, "boards_aug.csv")
    sgf_files = [ s for s in os.scandir(root_dir) if s.path.endswith(".sgf") ]
    with open(out, 'w') as f:
        f.write("board,ko,turn,next\n")
        for sgf in sgf_files:
            mvs = get_moves(sgf)
            g = go.Game(moves = mvs)
            for i in range(len(mvs)):
                board, ko, move = g.board, g.ko, g.moves[i]
                for k in range(4):
                    f.write(data_str(board, ko, i, move))
                    f.write(data_str(refl(board), refl(ko), i, refl(move)))
                    board, ko, move = rot(board), rot(ko), rot(move)
                g.play_move()

enc  = {go.EMPTY : 0, go.BLACK: 1, go.WHITE: -1}
dec = {0: go.EMPTY, 1: go.BLACK, -1: go.WHITE}

def data_str(board, ko , mv_num, move):
    return ','.join([board, str(ko), str(mv_num), str(move) + '\n']) 

def rot(b):
    #rotates 90 deg clockwise
    if isinstance(b, str):
        A = np.array([enc[c] for c in b]).reshape((9,9))
        B = np.rot90(A, k = 3).reshape(81).tolist()
        return ''.join([ dec[n] for n in B])
    elif isinstance(b, int):
        return (b*9 + 8 - b//9)%81
    
def refl(b):
    if isinstance(b, str):
        A = np.array([enc[c] for c in b]).reshape((9,9))
        B = (A.T).reshape(81).tolist()
        return ''.join([dec[n] for n in B])
    elif isinstance(b, int):
        x, y = divmod(b, 9)
        b = 9*y + x
        return b

def get_result(sgf):
    with open(sgf, 'r') as f:
        match = re.match(r"RE\[()\]")
    if match:
        return match.group(1)
    else:
        return "No result"

def get_moves(sgf):
    with open(sgf, 'r') as f:
        match = re.findall(r"[BW]\[(\w*)\]", f.read())
    mvs = []
    for mv in match:
        if len(mv)!= 2:
            break
        else: 
            mvs.append(9*(ord(mv[0])-97) + ord(mv[1])-97 )
    return mvs

if __name__ == "__main__":
    pre_process("data", ".")
