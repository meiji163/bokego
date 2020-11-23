#include "board.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <regex>
#include <array>
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
    this->lastMove = -1;
    std::vector< std::pair<int,int> > mvs;
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
    for(int i =0; i<this->dim*this->dim;i ++){
        this->board_string += ".";
    }
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
                }
                else{
                    std::pair <int, int> mv = {int(match.str(1)[0]-96), int(match.str(1)[1])-96};
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

void Board::loadState(std::string state, int turn, int lastMove, int ko){
    if(state.length() != this->dim*this->dim){
        std::runtime_error("Cannot load state");
    }

    this->board_string = state;
    this->turn = turn;
    this->lastMove = lastMove;
    
    if(ko == -1){
        this->iKo = -1;
        this->jKo = -1;
        this->isKo = false;
    }else{
        std::tie(this->iKo, this->jKo) = convertCoord(ko);
        this->isKo = false;
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

int Board::getKo(){
    if(this->iKo == -1 && this->jKo == -1){
        return -1;
    }else{
        return convertCoord(this->iKo, this->jKo);
    }
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
            if(i == 0 && j == 0){
                std::cout << "X" << " ";
            }else if(color == -1){
               if(i == 0 && j <= this->dim){
                   std::cout << j << " ";
               }else if(j == 0 && i <= this->dim){
                   std::cout << i << " ";
               }
            }else if(color == 0){
                std::cout << "_" << " ";
            }else if(color == 1){
                std::cout << "X" << " ";
            }else if(color == 2){
                std::cout << "O" << " ";
            }else{
                std::cout << color << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "Turn: " << this->turn << "| White: " << this->wCap << " | Black: " << this->bCap << std::endl;
    std::cout << "Ko:" << this->iKo << " " << this->jKo << std::endl;
    if (this->end)
        std::cout << "Game ended." << std::endl;
    for(int i = 0; i<40; i++){
        std::cout << "-";
    }
    std::cout << std::endl;
}

//plays stone at position and tracks turn and moves
bool Board::play(int color, int i, int j){
    if(this->addStone(color, i,j)){
        this->turn ++;
        this->lastMove = convertCoord(i, j);
        this->board_string = this->stateToString();
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
    //4 cases:
    // 1) self_atari = true, take = true, ko
    // 2) self_atari = false, take = true, then true
    // 3) self_atari = false, take = false, then true
    // 4) self_atari = true, take = false, then false
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

bool Board::PsuedoisValid(int color, int i, int j){
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
    //4 cases:
    // 1) self_atari = true, take = true, ko
    // 2) self_atari = false, take = true, then true
    // 3) self_atari = false, take = false, then true
    // 4) self_atari = true, take = false, then false
    if(self_atari && take){
        return true;
    }else if(self_atari){
        return false;
    }else{
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
                }else{
                    this->iKo = -1;
                    this->jKo = -1;
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
    if(!this->PsuedoisValid(color, i, j)){
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

std::pair<int, int> Board::convertCoord(int coord){
    return std::make_pair((coord/this->dim) + 1, coord%this->dim + 1);
}

int Board::convertCoord(int i, int j){
    return (i-1)*9 + (j-1);
}

void Board::createSGF(std::string file_path){
    std::ofstream os;
    os.open(file_path);
    os << "(;GM[1]SZ[9]KM[7]RU[Chinese]\n";
    char p1,p2;
    for(int i = 1; i<=9; i++){
        for(int j =1; j<=9; j++){
            p1 = i + 96;
            p2 = j + 96;
            if(this->board[i][j] == WHITE){
                os << ";W[" << p1 << p2 << "]";
            }else if(this->board[i][j] == BLACK){
                os << ";B[" << p1 << p2 << "]";
            }
        }
    }
    os <<")";
    os.close();
}

std::string Board::getBoardString(){
    return this->board_string;
}

//overload functions
bool Board::play(int color, int mv){
    std::pair<int, int> crds = convertCoord(mv);
    return this->play(color, crds.first, crds.second);
}

bool Board::addStone(int color, int mv){
    std::pair<int, int> crds = convertCoord(mv);
    return this->addStone(color, crds.first, crds.second);
}

bool Board::isValidMove(int color, int mv){
    std::pair<int, int> crds = convertCoord(mv);
    return this->isValidMove(color, crds.first, crds.second);
}

bool Board::isAlive(int color, int mv){
    std::pair<int, int> crds = convertCoord(mv);
    return this->isAlive(color, crds.first, crds.second);
}

bool Board::checkAlive(int color, int mv){
    std::pair<int, int> crds = convertCoord(mv);
    return this->checkAlive(color, crds.first, crds.second);
}

int Board::removeStones(int color, int mv){
    std::pair<int, int> crds = convertCoord(mv);
    return this->removeStones(color, crds.first, crds.second);
}

int Board::getPos(int mv){
    std::pair<int, int> crds = convertCoord(mv);
    return this->getPos(crds.first, crds.second);
}

int Board::getScore(){
    this->createSGF("./temp.sgf");
    std::string cm = "gnugo --score --chinese-rules --komi 5.5 -l sgf temp.sgf";
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cm.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    std::remove("./temp.sgf");
    switch(result[0]){
        case 'B':
            return 1;
        case 'W':
            return 0;
        default:
            return -1;
    }
}

int Board::getPos(int i, int j){
    return this->board[i][j];
}

int Board::getTurn(){
    return this->turn;
}

int Board::getLastMove(){
    return this->lastMove;
}

int** Board::getBoard(){
    return this->board;
}

Board::~Board(){
    for(int i =0; i<this->dim+2; i++){
        delete this->board[i];
        delete this->visited[i];
    }
    delete this->board;
    delete this->visited;
}