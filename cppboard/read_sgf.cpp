#include "board.h"
#include <regex>
#include <fstream>
#include <vector>
#include <iostream>
using namespace std;

vector <pair <int,int>> get_moves(string sgf){
    ifstream f(sgf);
    string line;
    smatch match;
    vector <pair<int,int>> moves;

    regex mv(";[BW]\\[(\\w\\w)\\]");
    regex res("RE\\[(\\w\\+\\w+)\\]");
    
    while(getline(f, line)){
        for (sregex_iterator it = sregex_iterator(line.begin(), line.end(), mv);
                it != sregex_iterator(); it++){
            match = *it;
            pair <int, int> move;
            move.first = match.str(1)[0]-96;
            move.second = int(match.str(1)[1])-96;
            moves.push_back(move);
            }
    }     
    return moves;
}

int main(){
    vector <pair <int,int>> mvs = get_moves("sample.sgf");
    Board board(9);

    for(vector<pair <int, int>>::iterator it = mvs.begin(); it!= mvs.end(); it++){
        pair <int, int> mv = *it;
        board.disp();
        cin.ignore();
        board.play(mv.first, mv.second);
    }
    return 0;
}
