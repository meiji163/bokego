#include<string>
#include<fstream>
#include<vector>
#include<tuple>
#include<sstream>
#include<iostream>
#include<chrono>
#include<thread>
#include "board.h"


#ifndef DATA_LOADER_H
#define DATA_LOADER_H

typedef std::chrono::time_point<std::chrono::high_resolution_clock> cTime;

class DataLoader{
    public:
        DataLoader();
        void mount(std::string file);
        void process(int min = 0, int max = -1);
        std::string getColumnNames();
        std::vector<std::string>* getSamples();
        int size();
    private:
        std::ifstream rs;
        std::string file;
        std::string column_names;
        std::vector<std::string> samples;
        bool isMounted;
};

#endif