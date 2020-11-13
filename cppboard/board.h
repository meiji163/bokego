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
        void loadState(std::string state, int turn, std::string ko="None");
        std::string stateToString();

        void resetVisit();
        bool play(int color, int i, int j);
        void pass(int color);
        bool addStone(int color, int i, int j);
        bool isValidMove(int color, int i, int j);
        bool isAlive(int color, int i, int j);
        bool checkAlive(int color, int i, int j);
        int removeStones(int color, int i, int j);

        int getPos(int i, int j);
        int getTurn();
        int** getBoard();

        //for policy network
        int potentialCaptureSize(int color, int i, int j);
        std::pair<int, int> getLibAndSize(int color, int i, int j);
        std::pair<int, int> countLibAndSize(int color, int i, int j);
        int getAllLiberties(int color);
        int getAllLibertiesHelper(int color, int i, int j, bool found);
        int getTurnsSince(int i, int j);

        //assume move is valid
        int getLibsAfterPlay(int color, int i, int j);
        int playPseudoMove(int color, int i, int j);
        void test();

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
        std::string board_string;
        int** board;
        bool** visited;
};

#endif 
