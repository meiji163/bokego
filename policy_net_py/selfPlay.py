import go
from bokePolicy import *
import torch
from torch.distributions.categorical import Categorical

def self_play(pi_1, pi_2):
    '''Play game between policies pi_1 and pi_2, with pi_1 playing black and pi_2 playing white.
    Return 1 if pi_1 wins and -1 if pi_2 wins.'''
    g = go.Game()
    for _ in range(35):
        g.play_move(sample(pi_1, g))
        g.play_move(sample(pi_2, g))
    bScore, wScore = g.score(komi = 5.5)
    print(g)
    print( bScore, wScore )
    if bScore > wScore:
        return 1
    else:
        return -1

def sample(policy, game):
    fts = features(game, policy.scale).unsqueeze(0)
    probs = F.softmax(policy(fts), dim = 1)
    m = Categorical(probs)
    mv = m.sample().item()
    while not game.is_legal(mv):
        mv = m.sample().item()
    return mv 

def update_policy():
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    pi = PolicyNet()
    checkpt = torch.load("v0.5/policy_v0.5_2020-10-29_1.pt", map_location = torch.device("cpu"))
    pi.load_state_dict(checkpt["model_state_dict"])
    for _ in range(5):
        self_play(pi, pi)

