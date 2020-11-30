import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bokeNet import PolicyNet, ValueNet, NinebyNineGames 
from datetime import date 
import argparse 

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description = "Supervise learning training script")
    parser.add_argument("-d", metavar="CSV", type = str, help = "path to data csv", required = True, dest = 'd')
    parser.add_argument("-c", metavar="CHECKPOINT", type = str, help = "path to saved torch model", dest = 'c')
    parser.add_argument("-e", metavar="EPOCHS", type = int, help = "number of epochs", default = 1, dest = 'e')
    parser.add_argument("-v", metavar="CSV", type = str, help = "paths to validation csv", dest = 'v')
    args = parser.parse_args() 
    print("Loading data...")
    data = NinebyNineGames(args.d)
    dataloader = DataLoader(data, batch_size = 128, shuffle = True, num_workers = 8)
    if args.v:
        validation_set = NinebyNineGames(args.v)
        validloader = DataLoader(validation_set, batch_size = 128, shuffle = True, num_workers = 4)
    print("Number of board positions: {}".format(len(data)))

    #v = ValueNet()
    #v.cuda()
    #v.train()
    #err = nn.MSELoss()
    pi = PolicyNet()
    pi.cuda()
    err = nn.CrossEntropyLoss()      
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    optimizer = torch.optim.AdamW(pi.parameters())
    if args.c:
        print("Loading checkpoint...")
        checkpt = torch.load(args.c, map_location = device)
        v.load_state_dict(checkpt["model_state_dict"] )
        optimizer.load_state_dict(checkpt["optimizer_state_dict"]) 
        epochs_trained = checkpt["epoch"]

        for state in optimizer.state.values():
            for k, t in state.items():
                if torch.is_tensor(t):
                    state[k] = t.cuda()
        #v.train()
    else:
        #policy = torch.load("v0.2/RL_policy_50.pt", map_location = device)
        #v.load_policy_dict(policy["model_state_dict"])
        epochs_trained = 0 

    pi.train()
    epochs = args.e 
    for epoch in range(epochs):
        losses = []
        valid_losses = []
        print("Epoch: {}".format(epochs_trained + 1))
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
            if i%1000 == 999:
                print(" Loss: {:.3f}".format(running_loss/1000.0))
                losses.append(running_loss/1000.0)
                running_loss = 0.0

                #validation loss
                if i%2000 == 999 and args.v:
                    v.eval()
                    with torch.no_grad():
                        valid_loss = 0.0
                        for j, v_data in enumerate(validloader, 0):
                            if j == 100:
                                break
                            inputs, moves = v_data
                            inputs, moves = inputs.to(device), moves.to(device)
                            outputs = v(inputs)
                            loss = err( outputs, moves)
                            valid_loss += loss
                        valid_losses.append(valid_loss/100.0)
                        print(" Validation Loss: {:.3f}".format(valid_loss/100.0))
                    v.train()
         
        epochs_trained += 1
        out_path = os.getcwd() + "/v0.3/policy_" + str(epochs_trained)+ ".pt"  
        torch.save({"model_state_dict": pi.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epochs_trained}, out_path)
        with open('policystats.txt', 'a+') as f:
            f.write(f"Epoch: {epochs_trained}\n")
            f.write(','.join([format(n, '.3f') for n in losses]) + '\n')
            f.write(','.join([format(n, '.3f') for n in valid_losses]) + '\n')
