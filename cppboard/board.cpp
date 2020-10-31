#include "board.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <regex>
#define BORDER -1
#define EMPTY 0
#define BLACK 1
#define WHITE 2


Board::Board(int n, const char* sgf ){ 
    this->dim = n;
    this->turn = 0;
    this->bCap = 0;
    this->wCap = 0;
    this->wPass = false;
    this->bPass = false;
    this->iKo = -1;
    this->jKo = -1;
    this->isKo = false;
    this->end = false;
    std::vector< std::pair<int,int> > mvs;
    this->moves = mvs; 
    this->board = new int*[n+2];
    this->visited = new bool*[n+2];
    for(int i = 0; i<n+2; i++){
        this->board[i] = new int[n+2];
        this->visited[i] = new bool[n+2];
        for(int j = 0; j<n+2; j++){
            if(i == 0 || i == n+1 || j == 0 || j == n+1){
                this->board[i][j] = BORDER;
            }else{
                this->board[i][j] = EMPTY;
            }
            this->visited[i][j] = false;
        }
    } 
    if(sgf != NULL)
            this->readSGF(sgf); 
}

void Board::readSGF(const char* sgf){ 
    std::ifstream f;
    std::string line;
    std::smatch match;
    std::regex mv(";[BW]\\[(\\w*)\\]");
    try{
        f.open(sgf);
        while(getline(f, line)){
            for (std::sregex_iterator it = std::sregex_iterator(line.begin(), line.end(), mv); it != std::sregex_iterator(); it++){
                match = *it;
                if (match.str(1) == ""){
                    std::pair <int, int> mv = {-1,-1};
                    this->moves.push_back(mv);
                }
                else{
                    std::pair <int, int> mv = {int(match.str(1)[0]-96), int(match.str(1)[1])-96};
                    this->moves.push_back(mv); 
                }
            }     
        }
    }
    catch (const std::ifstream::failure& e) {
        std::cerr << "Couldn't read sgf file";
    }
}

void Board::loadState(int** state){
    for(int i = 0; i<this->dim+2; i++){
        for(int j = 0; j<this->dim+2; j++){
            this->board[i][j] = state[i][j];
        }
    }
}

void Board::resetVisit(){
    for(int i = 0; i < this->dim+2; i++){
        for(int j =0; j < this->dim+2; j++){
           this->visited[i][j] = false;
        }
    }
}

void Board::disp(){
    int color;
    for(int i = 0; i<this->dim+2; i++){
        for(int j = 0; j<this->dim+2; j++){
            color = this->board[i][j];
            if(color == -1){
                std::cout << "X" << " ";
            }else if(color == 0){
                std::cout << "." << " ";
            }else if(color == 1){
                std::cout << "*" << " ";
            }else if(color == 2){
                std::cout << "O" << " ";
            }else{
                std::cout << color << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "Turn: " << this->turn << "| White: " << this->wCap << " | Black: " << this->bCap << std::endl;
    if (this->end)
        std::cout << "Game ended." << std::endl;
    std::cout << std::endl;
}

bool Board::play(int i, int j){
    if(this->addStone((this->turn %2) + 1, i,j)){
        this->turn++;
        std::pair<int,int> mv = {i,j};
        this->moves.push_back(mv);
        return true;
    }
    return false;
}

void Board::pass(int color){
    if (color == BLACK){
        if (wPass)
            this->end = true;
        this->bPass = true;
    }
    else if (color == WHITE){
        if (bPass)
            this->end = true;
        this->wPass = true;
    }
    this->turn++;
}

bool Board::isValidMove(int color, int i, int j){
    if(this->end)
        return false;
    if(i == -1 && j== -1)
        return true;

    if(this->board[i][j] != 0)
        return false;

    if(i == this->iKo && j == this->jKo)
        return false;
    
    int op_color = (color%2) + 1;

    bool take = false;
    bool self_atari = false;

    this->board[i][j] = color;
    self_atari = !isAlive(color, i, j);
    this->resetVisit();
    for(int k=-1; k<3; k++){
        int a = i + (k%2);
        int b = j + ((k-1)%2);
        if(this->board[a][b] == op_color){
            take = take | !isAlive(op_color, a, b); 
            this->resetVisit();
        }
    }
    this->board[i][j] = 0;
    
    //4 cases:
    // 1) self_atari = true, take = true, ko
    // 2) self_atari = false, take = true, then true
    // 3) self_atari = false, take = false, then true
    // 4) self_atari = true, take = false, then false
    if(self_atari){
        if(take){
            this->isKo = true;
            return true;
        }else{
            return false;
        }
    }else{
        this->iKo = -1;
        this->jKo = -1;
        return true;
    }
}

bool Board::addStone(int color, int i, int j){
    if(this->isValidMove(color, i, j)){
        if (i == -1 && j == -1){ 
            //pass = {-1,-1}
            this->pass(color);
            return true;
        }
        this->board[i][j] = color;

        for(int k=-1; k<3; k++){
            int a = i + (k%2);
            int b = j + ((k-1)%2);
            int neighbor_color = this->board[a][b];
            if(!isAlive(neighbor_color, a, b)){
                int points = removeStones(neighbor_color, a, b);
                if(points == 1 && this->isKo){
                    this->iKo = a;
                    this->jKo = b;
                    this->isKo = false;
                }
                if( color == BLACK){
                    this->bCap += points;
                }else{
                    this->wCap += points;
                }
            }
            this->resetVisit();
        }
        return true;
    }
    return false;
}

bool Board::isAlive(int color, int i, int j){
    if(color != BLACK && color != WHITE){
        return true;
    }
    if(this->board[i][j] == EMPTY){
        return true;
    }
    if(this->board[i][j] != color || visited[i][j]){
        return false;
    }
    this->visited[i][j] = true;
    bool check = false;
    for(int k=-1; k<3; k++){
        int a = i + (k%2);
        int b = j + ((k-1)%2);
        check = check || isAlive(color, a, b);
    }
    return check;
}

int Board::removeStones(int color, int i, int j){
    if(this->board[i][j] != color){
        return 0;
    }
    this->board[i][j] = 0;
    int dead_num = 1;
    for(int k=-1; k<3; k++){
        int a = i + (k%2);
        int b = j + ((k-1)%2);
        dead_num += removeStones(color, a, b);
    }
    return dead_num;
}

int Board::getTurn(){
    return this->turn;
}

int** Board::getBoard(){
    return this->board;
}

std::vector< std::pair<int,int> > Board::getMoves(){
    return this->moves;
}
Board::~Board(){
    for(int i =0; i<this->dim+2; i++){
        delete this->board[i];
        delete this->visited[i];
    }
    delete this->board;
    delete this->visited;
}
