#!/usr/bin/env python
''' 
Loop through all sgfs in a directory and convert each board state 
to csv (for training data)

Usage: python sgf_to_csv.py <directory> <output_file>"
'''

import os
import glob
import sys
from tqdm import tqdm

import bokego.go as go

if __name__ == "__main__":
    if len(sys.argv) != 3: 
        print(f"usage: python {sys.argv[0]} <directory> <output_file>")
        sys.exit(1)
    directory, out_file = sys.argv[1:]
    sgf_files = glob.glob(directory + "/*.sgf")
    
    if len(sgf_files) == 0:
        print("No SGF files found")
        sys.exit(1)

    print(f"Processing {len(sgf_files)} SGFs")


    with open(out_file, 'w+') as out:
        out.write("board,ko,turn,move\n")
        for sgf in tqdm(sgf_files):
            g = go.Game(sgf=sgf)
            for i in range( len(g)-2 ):
                mv = g.moves[i]
                if mv == go.PASS:
                    g.play_pass()
                    continue

                if g.ko is None:
                    out.write(f"{g.board},,{g.turn},{mv}\n")
                else:
                    out.write(f"{g.board},{g.ko},{g.turn},{mv}\n")
                g.play_move()
