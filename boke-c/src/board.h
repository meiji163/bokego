#include <vector>
#include <string>
#ifndef BOARD_H
#define BOARD_H

class Board{
    public:
        Board(int n, const char* sgf = NULL);
        void disp();
        void readSGF(const char* sgf);
        void loadState(int** state);
        void loadState(std::string state);
        void loadState(std::string state, int turn, int lastMove = -1, int ko = -1);
        void createSGF(std::string file_path);

        void resetVisit();
        bool play(int color, int i, int j);
        bool play(int color, int mv);
        void pass(int color);
        bool addStone(int color, int i, int j);
        bool addStone(int color, int mv);
        bool isValidMove(int color, int i, int j);
        bool isValidMove(int color, int mv);
        bool PsuedoisValid(int color, int i, int j);
        bool isAlive(int color, int i, int j);
        bool isAlive(int color, int mv);
        bool checkAlive(int color, int i, int j);
        bool checkAlive(int color, int mv);
        int removeStones(int color, int i, int j);
        int removeStones(int color, int mv);

        int getPos(int i, int j);
        int getPos(int mv);
        int getTurn();
        int getLastMove();
        int** getBoard();
        std::string getBoardString();
        std::string stateToString();
        int getKo();

        std::pair<int, int> convertCoord(int coord);
        int convertCoord(int i, int j);

        //for policy network
        int getScore();
        int potentialCaptureSize(int color, int i, int j);
        std::pair<int, int> getLibAndSize(int color, int i, int j);
        std::pair<int, int> countLibAndSize(int color, int i, int j);
        //assume move is valid
        int getLibsAfterPlay(int color, int i, int j);
        int playPseudoMove(int color, int i, int j);
        bool fillOwnEye(int color, int i, int j);

        std::vector< std::pair<int,int> > getMoves();
        ~Board();
    private:
        int dim, turn, wCap, bCap, iKo, jKo, lastMove;
        bool wPass, bPass, isKo, end;
        std::string board_string;
        int** board;
        bool** visited;
};

#endif 
