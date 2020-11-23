#include "data_loader.h"

DataLoader::DataLoader(){
    this->isMounted = false;
    this->file = "";
    this->column_names = "No column names";
}

void DataLoader::mount(std::string file){
    rs.open(file);
    if(!this->rs.is_open()){
        throw std::runtime_error("file cannot be open");
    }
    this->file = file;
    this->isMounted = true;
}

void DataLoader::process(int min, int max){
    std::string line, buffer;
    if(!this->isMounted){
        throw std::runtime_error("file is not mounted");
    }
    //get column names
    getline(this->rs, line);
    this->column_names = line;

    for(int i = 0; i<min; i++){
        getline(this->rs, line);
    }

    auto start = std::chrono::high_resolution_clock::now();
    double count = 0, total = 0;
    while(getline(this->rs, line) && total != max){
        if(line != ""){
            if(count == 500000){
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                std::cout << "num of samples/s: " << count/(double)(duration.count()*0.000001) << std::endl;
                count = 0;
                start = end;
            }else{
                count++;
            }
            this->samples.push_back(line);
            total++;
        }
    }
}

std::string DataLoader::getColumnNames(){
    if(!this->isMounted){
        throw std::runtime_error("file is not mounted");
    }
    return this->column_names;
}

std::vector<std::string>* DataLoader::getSamples(){
    return &this->samples;
}

int DataLoader::size(){
    return this->samples.size();
}