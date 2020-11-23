#include <iostream>
#include <string>
#include <chrono> 
#include "board.h"
#include "policy_net.h"
#include "torch/script.h"
#include "data_loader.h"
#include "mcts.h"
#include<vector>
#include<fstream>
#include<random>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  Board game = Board(9);
  PolicyNet model = PolicyNet();

  MCTS ts = MCTS(20, 1.0, module);
  ts._root->_lastmove = -1;
  ts._root->_turn = 0;

  
  //auto start = std::chrono::high_resolution_clock::now();
  int turn = 0;
  while(true){
    int i, j;
    std::cin >> i >> j;

    ts.play(ts._root, 9*(i-1)+j-1);
    ts._root->print();
    if(turn <= 10){
      ts.rollout(ts._root, 20);
    }else{
      ts.rollout(ts._root, 1000);
    }
    turn++;
    ts.choose();
    ts._root->print();
  }
  //auto end = std::chrono::high_resolution_clock::now();

  //auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  //std::cout << "Time taken: " << duration.count() << std::endl;

  return 0;
}