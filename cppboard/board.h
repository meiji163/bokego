#include <vector>
#ifndef BOARD_H
#define BOARD_H

class Board{
    public:
        Board(int n, const char* sgf = NULL);
        void disp();
        void loadState(int** state);
        void readSGF(const char* sgf);
        void resetVisit();
        bool play(int i, int j);
        void pass(int color);
        bool addStone(int color, int i, int j);
        bool isValidMove(int color, int i, int j);
        bool isAlive(int color, int i, int j);
        int removeStones(int color, int i, int j);
        int getTurn();
        int** getBoard();
        std::vector< std::pair<int,int> > getMoves();
        ~Board();
    private:
        int dim;
        int turn;
        int bCap;
        int wCap;
        bool wPass;
        bool bPass;
        int iKo;
        int jKo;
        bool isKo;
        bool end;
        std::vector< std::pair<int,int> > moves;
        int** board;
        bool** visited;
};

#endif 
