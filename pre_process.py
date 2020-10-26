import os
import re
import go

def pre_process(root_dir, target_dir):
    out = os.path.join(target_dir, "boards.csv")
    sgf_files = [ s for s in os.scandir(root_dir) if s.path.endswith(".sgf") ]
    with open(out, 'w') as f:
        f.write("board,ko,next\n")
        for sgf in sgf_files:
            mvs = get_moves(sgf)
            g = go.Game(moves = mvs)
            for i in range(len(mvs)):
                f.write(','.join([g.board,str(g.ko), str(go.squash(g.moves[i]))]) + '\n')
                g.play_move()
        
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
            mvs.append((ord(mv[0])-97, ord(mv[1])-97) )
    return mvs

if __name__ == "__main__":
    pre_process("data", ".")
