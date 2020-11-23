#include <torch/nn.h>
#include "board.h"

#ifndef POLICY_NET_H
#define POLICY_NET_H

#define BORDER -1
#define EMPTY 0
#define BLACK 1
#define WHITE 2

struct PolicyNet : torch::nn::Module{
    torch::Tensor features(Board &ref_board){
        torch::Tensor input = torch::zeros({27,9,9});
        int lib, libAfterPlay, gsize, color, player, valid, turn, captSize;

        Board board = Board(9);
        board.loadState(ref_board.getBoardString(), ref_board.getTurn(), ref_board.getLastMove(), ref_board.getKo());

        if(board.getLastMove() != -1){
            std::pair<int, int> lastmove = board.convertCoord(board.getLastMove());
            //last move, 4
            input[4][lastmove.first - 1][lastmove.second - 1] = 1;
        }
        
        turn = board.getTurn() % 2;
        player = turn == 0 ? BLACK : WHITE;
        for(int i = 0; i<9; i++){
            for(int j = 0; j<9; j++){
                color = board.getPos(i+1,j+1);
                valid = board.PsuedoisValid(player, i+1 , j+1); 
                //position color, 0 - 2
                if(color == EMPTY){
                    input[2][i][j] = 1;
                }else if(player == color){
                    input[0][i][j] = 1;
                }else{
                    input[1][i][j] = 1;
                }
                //turns, 3
                if(player == BLACK){
                    input[3][i][j] = 1;
                }
                
                //lib, 6-12
                if(color != EMPTY){
                    lib = board.getLibAndSize(color, i+1, j+1).first;
                    lib = lib > 7 ? 7 : lib;
                    input[5 + lib][i][j] = lib;
                }
                
                //check this
                if(valid){
                    //valid, 5
                    input[5][i][j] = 1;
                    //lib after play, 13-19
                    libAfterPlay = board.getLibsAfterPlay(player, i+1, j+1);
                    libAfterPlay = libAfterPlay > 7 ? 7 : libAfterPlay;
                    input[12+ libAfterPlay][i][j] = libAfterPlay;
                }
                //20-26
                captSize = board.potentialCaptureSize(player, i+1, j+1);
                captSize = captSize > 7 ? 7 : captSize;
                if(captSize != 0){
                    input[19+captSize][i][j] = 1;
                }
            }
        }
        return input;
    }

    int scale;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr}, conv6{nullptr};
};

#endif