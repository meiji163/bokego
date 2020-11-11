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
    this->turn = 0, this->bCap = 0;
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
    this->board_string = "";
    if(sgf != NULL) this->readSGF(sgf); 
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

void Board::loadState(std::string state){
    if(state.length() != this->dim*this->dim){
        std::runtime_error("Cannot load state");
    }

    this->board_string = state;
    char pos;
    int val;
    
    for(int i = 1; i<= this->dim; i++){
        for(int j = 1; j<=this->dim; j++){
            pos = state[9*(i-1)+j - 1];
            if(pos == 'X'){
                val = BLACK;
            }else if(pos == 'O'){
                val = WHITE;
            }else{
                val = EMPTY;
            }
            this->board[i][j] = val;
        }
    }
}

//load board state
void Board::loadState(int** state){
    for(int i = 0; i<this->dim+2; i++){
        for(int j = 0; j<this->dim+2; j++){
            this->board[i][j] = state[i][j];
        }
    }
}

void Board::loadState(std::string state, int turn, std::string ko){
    if(state.length() != this->dim*this->dim){
        std::runtime_error("Cannot load state");
    }

    this->board_string = state;
    this->turn = turn;
    if(ko != "None"){
        int num_ko = std::stoi(ko);
        this->iKo = (num_ko/this->dim) + 1;
        this->jKo = (num_ko%this->dim) + 1;
        this->isKo = true;
    }
    char pos;
    int val;
    
    for(int i = 1; i<= this->dim; i++){
        for(int j = 1; j<=this->dim; j++){
            pos = state[9*(i-1)+j - 1];
            if(pos == 'X'){
                val = BLACK;
            }else if(pos == 'O'){
                val = WHITE;
            }else{
                val = EMPTY;
            }
            this->board[i][j] = val;
        }
    }
}

std::string Board::stateToString(){
    std::string state = "";
    char pos;
    for(int i = 1; i<=this->dim; i++){
        for(int j = 1; j<=this->dim; j++){
            if(this->board[i][j] == EMPTY){
                pos = '.';
            }else if(this->board[i][j] == BLACK){
                pos = 'X';
            }else{
                pos = 'O';
            }
            state += pos;
        }
    }
    return state;
}

//sets visited matrix to all false
void Board::resetVisit(){
    for(int i = 0; i < this->dim+2; i++){
        for(int j =0; j < this->dim+2; j++){
           this->visited[i][j] = false;
        }
    }
}

//prints state of board
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

//plays stone at position and tracks turn and moves
bool Board::play(int color, int i, int j){
    if(this->addStone(color, i,j)){
        this->turn ++;
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

//checks if move is valid
//4 cases:
// 1) self_atari = true, take = true, ko
// 2) self_atari = false, take = true, then true
// 3) self_atari = false, take = false, then true
// 4) self_atari = true, take = false, then false
bool Board::isValidMove(int color, int i, int j){
    if(this->end) return false;
    if(i == -1 && j== -1) return true;
    if(this->board[i][j] != EMPTY) return false;
    if(i == this->iKo && j == this->jKo) return false;
    
    int op_color = (color%2) + 1;
    bool take = false;
    bool self_atari = false;

    this->board[i][j] = color;
    self_atari = !isAlive(color, i, j);
    for(int k=-1; k<3; k++){
        int a = i + (k%2);
        int b = j + ((k-1)%2);
        if(this->board[a][b] == op_color){
            take = take | !isAlive(op_color, a, b); 
        }
    }
    this->board[i][j] = EMPTY;

    if(self_atari && take){
        this->isKo = true;
        return true;
    }else if(self_atari){
        return false;
    }else{
        this->iKo = -1;
        this->jKo = -1;
        return true;
    }
}

//plays stone at position; handles captures and kos
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
            if(neighbor_color != BORDER && !isAlive(neighbor_color, a, b)){
                int points = removeStones(neighbor_color, a, b);
                if(points == 1 && this->isKo){
                    this->iKo = a;
                    this->jKo = b;
                    this->isKo = false;
                }
                if(color == BLACK){
                    this->bCap += points;
                }else{
                    this->wCap += points;
                }
            }
        }
        return true;
    }
    return false;
}

//return whether a group is alive or dead
bool Board::isAlive(int color, int i, int j){
    bool alive = checkAlive(color, i , j);
    this->resetVisit();
    return alive;
}

//isAlive helper function
bool Board::checkAlive(int color, int i, int j){
    if(this->board[i][j] == EMPTY) return true;
    if(this->board[i][j] != color || visited[i][j]) return false;
    
    this->visited[i][j] = true;
    bool check = false;
    for(int k=-1; k<3; k++){
        int a = i + (k%2);
        int b = j + ((k-1)%2);
        check = check || checkAlive(color, a, b);
    }
    return check;
}

//remove group and return number of stones removed
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

//return group's number of liberty and size
std::pair<int, int> Board::getLibAndSize(int color, int i, int j){
    if(this->board[i][j] == EMPTY){
        return std::make_pair(0,0);
    }
    std::pair<int,int> vals = this->countLibAndSize(color, i, j);
    this->resetVisit();
    return vals;
}

//getLibAndSize helper function
std::pair<int,int> Board::countLibAndSize(int color, int i, int j){
    std::pair<int, int> vals = std::make_pair(0, 0);
    if(this->board[i][j] == EMPTY && !visited[i][j]){
        vals.first++;
    }
    if(this->board[i][j] != color || visited[i][j]){
        visited[i][j] = true;
        return vals;
    }
    visited[i][j] = true;
    vals.second++;
    for(int k=-1; k<3; k++){
        int a = i + (k%2);
        int b = j + ((k-1)%2);
        std::pair<int, int> temp = countLibAndSize(color, a, b);
        vals.first += temp.first;
        vals.second += temp.second;
    }
    return vals;
}

//returns number of stones that would be captured if stone is played
int Board::potentialCaptureSize(int color, int i, int j){
    if(!this->isValidMove(color, i, j)){
        return 0;
    }
    this->board[i][j] = color;
    int points = 0;
    for(int k=-1; k<3; k++){
        int a = i + (k%2);
        int b = j + ((k-1)%2);
        int neighbor_color = this->board[a][b];
        if(neighbor_color != BORDER && !isAlive(neighbor_color, a, b)){
            points += getLibAndSize(neighbor_color, a, b).second;
        }
    }
    this->board[i][j] = EMPTY;
    return points;
}

int Board::getAllLiberties(int color){
    int lib = 0;
    for(int i = 1; i<= this->dim; i++){
        for(int j = 1; j<=this->dim; j++){
            
            lib += this->getAllLibertiesHelper(color, i, j, false);
            
        }
    }
    this->loadState(this->board_string);
    return lib;
}

int Board::getAllLibertiesHelper(int color, int i, int j, bool found){
    int pos = this->board[i][j];
    if(pos == EMPTY && found){
        return 1;
    }
    if(pos != color || (pos == EMPTY && !found) ){
        return 0;
    }
    this->board[i][j] = 9; //arbitary value
    int lib = 0;
    for(int k=-1; k<3; k++){
        int a = i + (k%2);
        int b = j + ((k-1)%2);
        lib += getAllLibertiesHelper(color, a, b, true);
    }
    return lib;
}

int Board::playPseudoMove(int color, int i, int j){
    this->board[i][j] = color;
    int points = 0;
    for(int k=-1; k<3; k++){
        int a = i + (k%2);
        int b = j + ((k-1)%2);
        int neighbor_color = this->board[a][b];
        if(neighbor_color != BORDER && !isAlive(neighbor_color, a, b)){
            points += removeStones(neighbor_color, a, b);
        }
    }
    return points;
}

int Board::getLibsAfterPlay(int color, int i, int j){
    this->playPseudoMove(color, i, j);
    int lib = this->getLibAndSize(color, i, j).first;
    this->loadState(this->board_string);
    return lib;
}

void Board::test(){
    for(int i = 1;i<=this->dim; i++){
        for(int j = 1; j<=this->dim; j++){
            std::cout << this->getLibsAfterPlay(BLACK, i, j) << " ";
        }
    }
}

// returns the number of turns since move was played (if played since 8 turns return 8)
int Board::getTurnsSince(int i, int j){
    int k = 0;
    while(k < this->moves.size() && k<8){
        if(this->moves[this->moves.size() - 1 - k].first == i && this->moves[this->moves.size() - 1 - k].second == j){
            return k + 1;
        }
        k++;
    }
    return 8;
}

int Board::getPos(int i, int j){
    return this->board[i][j];
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