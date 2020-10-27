#!/usr/bin/python3
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bokePolicy import PolicyNet, NinebyNineGames, SCALE
from datetime import date 
import argparse 

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description = "training script for Boke policy")
    parser.add_argument("-d", metavar="DATA", type = str, nargs=1, help = "path to csv", required = True)
    parser.add_argument("-c", metavar="CHECKPOINT", type = str, nargs = 1, help = "path to saved torch model")
    parser.add_argument("-e", metavar="EPOCHS", type = int, nargs =1, help = "number of epochs", default = 1)
    args = parser.parse_args() 
    
    print("Loading data...")
    data = NinebyNineGames(args.d[0], transform = "rot90", scale = SCALE)
    dataloader = DataLoader(data, batch_size = 128, shuffle = True, num_workers = 4)
    print("Number of board positions: {}".format(len(data)))

    pi = PolicyNet(scale = SCALE)
    pi.cuda()
    err = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    optimizer = torch.optim.Adam(pi.parameters())
    if args.c:
        print("Loading checkpoint...")
        checkpt = torch.load(args.c[0], map_location = device)
        pi.load_state_dict(checkpt["model_state_dict"] )
        optimizer.load_state_dict(checkpt["optimizer_state_dict"]) 
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        pi.train()
     
    epochs = args.e[0] 
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader,0)):
            inputs, moves = data
            inputs, moves = inputs.to(device), moves.to(device)
            
            optimizer.zero_grad()
            outputs = pi(inputs)
            
            #backprop
            loss = err(outputs, moves)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            if i%2000 == 1999:
                print(" Loss: ", running_loss)
                running_loss = 0.0
        out_path = r"/home/jupyter/BokeGo/policy_net_py/" + "policy_train_" + str(date.today()) + "_2.pt"  
        torch.save({"model_state_dict": pi.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, out_path)
    
