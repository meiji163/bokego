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
  if (argc != 3) {
    std::cerr << "usage: app <path-to-SL-policy> <path-to-RL-policy>\n";
    return -1;
  }

  torch::jit::script::Module pi, pi_r;
  try {
    pi = torch::jit::load(argv[1]);
    pi_r = torch::jit::load(argv[2]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  Board game = Board(9);

  MCTS ts = MCTS(5.0, pi, pi_r);
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
      ts.rollout(ts._root, 200);
    }else{
      ts.rollout(ts._root, 500);
    }
    turn++;
    ts.choose();
    ts._root->print();
    std::cout << "Visits: " << ts._root->_visited << std::endl;
    std::cout << "Rewards: " << ts._root->_reward << std::endl;
  }
  //auto end = std::chrono::high_resolution_clock::now();

  //auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  //std::cout << "Time taken: " << duration.count() << std::endl;

  return 0;
}
