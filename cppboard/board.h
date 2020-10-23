#ifndef BOARD_H
#define BOARD_H

class Board{
    public:
        Board(int n);
        void disp();
        void loadState(int** state);
        void resetVisit();
        bool play(int i, int j);
        bool addStone(int color, int i, int j);
        bool isValidMove(int color, int i, int j);
        bool isAlive(int color, int i, int j);
        int removeStones(int color, int i, int j);
        int getTurn();
        int** getBoard();
        ~Board();
    private:
        int dim;
        int turn;
        int bCap;
        int wCap;
        int iKo;
        int jKo;
        bool isKo;
        int** board;
        bool** visited;
};

#endif 
