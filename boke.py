#!/usr/bin/env python
import argparse
import os
import sys
import torch
from bokego.gtp import GTP
from bokego.nnet import PolicyNet, ValueNet, PolicyNet_v2
from bokego.mcts import Go_MCTS 
from bokego import PKG_PATH 

torch.set_grad_enabled(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "BokeGo v0.3 GTP Engine")
    parser.add_argument("-t", metavar="SEC", type = float, dest = 't',
                        help = "time limit in seconds for each move", default = 10.0)
    parser.add_argument("-r", dest = 'r', type = int, help = "number of rollouts per move")
    parser.add_argument("-p", metavar="PATH", type = str, dest = 'p', help = "path to policy weights",
                        default = os.path.join( "data", "weights", "policy_0.pt"))
    parser.add_argument("-v", metavar="PATH", type = str, dest = 'v', help = "path to value weights", 
                        default = os.path.join("data","weights", "value_1.pt"))
    parser.add_argument("-g", "--gpu", action = "store_true", 
                        help = "Accelerate NN forward by GPU.")
    parser.add_argument("--simulate", action = "store_true",
                        help = "enable simulations to end of game (slow)")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")

    pi = PolicyNet()
    p_weight = torch.load(args.p, map_location= device)
    pi.load_state_dict(p_weight["model_state_dict"])
    pi.eval()

    val = ValueNet()
    v_weight = torch.load(args.v, map_location= device)
    val.load_state_dict(v_weight["model_state_dict"])
    val.eval()

    gtp = GTP(Go_MCTS(), 
                policy_net=pi, value_net=val, 
                no_sim = not args.simulate,
                time_lim = args.t,
                device = device)
    gtp.start()
