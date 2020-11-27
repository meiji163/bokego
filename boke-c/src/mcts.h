#include<string>
#include<vector>
#include"board.h"
#include"policy_net.h"
#include<torch/script.h>
#include<iostream>
#include<queue>
#include<chrono>
#include<map>
#include<random>
#include<mutex>
#include<thread>

#ifndef MCTS_H
#define MCTS_H

#define EMPTY_STATE "................................................................................."

#define DEBUG false

struct Node{
    Node(std::string state, int lastMove = -1, int ko = -1){
        _state = state;
        _visited = 0;
        _ko= ko;
        _lastmove= lastMove;
        _parent = nullptr;
        _turn = 0;
        _reward = 0;
    }

    void print(){
        for(int i =0; i<=9; i++){
            std::cout << i << " ";
        }
        std::cout << std::endl;
        int row = 1;
        for (int i = 0; i<81; i+=9){
            std::cout << row << " ";
            for(auto &n : _state.substr(i, 9)){
                std::cout << n << " ";
            }
            std::cout << std::endl;
            row++;
        }
        std::cout << "ko: " << _ko << std::endl;
        std::cout << "lastmove: " << _lastmove << std::endl;
        std::cout << "turn: " << _turn << std::endl;
    }

  	std::string _state;
    int _turn, _visited, _reward, _lastmove, _ko;
    
    std::vector<Node*> children;
    Node* _parent;
};

struct MCTS{
    int _max_turns, _expand_thresh, _branch_num;
    float _alpha; //exploration weight
    Node* _root;
    PolicyNet _network;
    torch::jit::script::Module _model, _rollout_model;
    std::map<std::string, std::vector<float>> _dist_map; //store policy distributions
    std::map<std::string, std::vector<float>> _dist_r_map; //store rollout policy distributions

    MCTS(float alpha, torch::jit::script::Module &pi, torch::jit::script::Module &pi_r, std::string state = EMPTY_STATE){
        //descend and select with pi, rollout pi_r
        _max_turns  = 75;
        _expand_thresh = 20;
        _branch_num = 10; 
        _alpha = alpha;
        _root = new Node(state);
        _network = PolicyNet();
        _model = pi;
        _rollout_model = pi_r;
    }
    
    Node* play(Node* node,int mv){
        Board game(9);
        game.loadState(node->_state, node->_turn, node->_lastmove, node->_ko);
        int player = (node->_turn)%2 + 1;
        game.play(player, mv);
        Node* c = new Node(game.getBoardString());
        c->_ko = game.getKo();
        c->_turn = game.getTurn();
        c->_lastmove = mv;
        this->_root = c;
        c->_parent = nullptr;
        return c;
    }

    void set_dist(Board &game, bool rollout = false){
        torch::Tensor x = _network.features(game);
        x = torch::unsqueeze(x, 0);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x);
        at::Tensor output;
        if (rollout){
            output = _rollout_model.forward(inputs).toTensor();
        }else{
            output = _model.forward(inputs).toTensor();
        }
        output = torch::softmax(output, 1);
        output = torch::squeeze(output, 0).contiguous();
        std::vector<float> probs(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
        std::string state = game.stateToString();
        if (rollout){
            _dist_r_map.insert(std::pair<std::string, std::vector<float>>(state, probs));
        }else{
            _dist_map.insert(std::pair<std::string, std::vector<float> >(state, probs));
        }
    }

    void set_dist(Node* node, bool rollout = false){
        Board game = Board(9);
        game.loadState(node->_state, node->_turn, node->_lastmove, node->_ko);
        set_dist(game, rollout);
    }

    Node* selection(Node* node){
        int tot_visits = 0;
        Node* best;
        float best_puct = -1;
        for(auto &n : node->children){
            tot_visits += n->_visited;
        }
        for(auto &n : node->children){
            std::string state = node->_state;
            if(_dist_map.count(state) == 0){
                set_dist(node);
            }
            float p_lm = _dist_map[node->_state][n->_lastmove];
            float avg_reward = n->_visited > 0.0 ? float(n->_reward / n->_visited) : 0;
            float puct = avg_reward + _alpha * p_lm * sqrt(tot_visits) / (1 + n->_visited);
            if (puct > best_puct){
                best_puct = puct;
                best = n;
            }
        }
        return best;
    }
    
    Node* choose(){
        if (_root->_turn > _max_turns){
            return nullptr;
        }
        if (_root->children.size() == 0){
            int mv = getNextMove(_root);
            return play(_root, mv);
        }
        int best_v = 0;
        Node* best;
        for(auto &n : _root->children){
            if (n->_visited != 0){
                if ( n->_visited > best_v){
                    best_v = n->_visited;
                    best = n;
                }
            }
        }
        this->_root = best;
        best->_parent = nullptr;
        return best;
            
    }

    std::vector<int> topk(std::vector<float> const &arr, int k){
        //return k indices with largest values
        std::vector<int> idx;
        std::priority_queue<std::pair<float, int>> q;
        int min = arr[0];
        for(int i = 0; i<arr.size(); ++i){
            if (i<k){
                q.push(std::pair<float, int>(arr[i], i));
                if (arr[i] < min){
                    min = arr[i];
                }
            }else if (arr[i] >= min){
                q.push(std::pair<float, int>(arr[i], i));
            }
        }
        for (int i = 0; i< k; ++i){
            int top = q.top().second;
            idx.push_back(top);
            q.pop();
        }
        return idx;
    }     

    void expansion(Node* node){
        std::string state = node->_state;
        if(_dist_map.count(state) == 0){
            set_dist(node);
        }
        std::vector<float> probs = _dist_map[state];
        std::vector<int> top = topk(probs, _branch_num);
        Board game = Board(9);
        for (auto itr = top.begin(); itr != top.end(); ++itr){
            game.loadState(node->_state, node->_turn, node->_lastmove, node->_ko);
            int i = *itr/9 + 1; 
            int j = (*itr % 9) + 1;
            if(game.isValidMove((node->_turn%2) +1, i, j)){
                game.play((node->_turn%2) + 1, i,j);
                Node* temp = new Node(game.stateToString());
                temp->_ko = game.getKo();
                temp->_lastmove = *itr;
                temp->_parent = node;
                temp->_turn = game.getTurn();
                node->children.push_back(temp);
                if (DEBUG){
                    std::cout << "EXPANDING CHILD: " << std::endl;
                    temp->print();
                }
            }
        }
    }

    Node* descend(Node* node){
        Node* n = node;
        while(n->children.size() != 0){
            n = selection(n);
        }
        if (n->_visited > _expand_thresh){
            expansion(n);
        }
        return n;
    }

    void rollout(Node* node, int n_rolls){
        for (int i = 0; i< n_rolls; i++){
            Node* leaf = this->descend(node);
            int reward = this->simulation(leaf);
            if (DEBUG){
                std::cout << "Simulating from node: " << std::endl;
                leaf->print();
                std::cout << "Visits: " << leaf->_visited << std::endl;
                std::cout << "Reward: " << reward << std::endl;
            }
            this->backpropagate(leaf, reward);
        }
    }

    int getNextMove(Node* node){
        Board game = Board(9);
        game.loadState(node->_state, node->_turn, node->_lastmove, node->_ko);
        return getNextMove(game);
    }

    int getNextMove(Board &refBoard, bool rollout = false){
        Board game = Board(9);
        game.loadState(refBoard.getBoardString(), refBoard.getTurn(), refBoard.getLastMove(), refBoard.getKo());
        std::string state = game.getBoardString();
        std::vector<float>* p;
        if (rollout){
            if (_dist_r_map.count(state) == 0){
                set_dist(game, true);
            }
            p = &_dist_r_map[state];
        }else{
            if (_dist_map.count(state) == 0){
                set_dist(game);
            }
            p = &_dist_map[state];
        }
        unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::discrete_distribution<float> distribution(p->begin(), p->end());
        return distribution(generator);
    }

    int simulation(Node* node){
        int invert = (node->_turn )%2;
        Board game = Board(9);
        game.loadState(node->_state, node->_turn, node->_lastmove, node->_ko);

        for(int i = node->_turn ; i<_max_turns; i++){
            int mv = getNextMove(game, true);
            game.play((i%2) + 1, mv);
        }
        //gnugo returns who wins 1 = B
        int reward = game.getScore();
        return int(invert^reward);
    }

    void backpropagate(Node* node, int result){
        while (node != nullptr){
            node->_visited++;
            node->_reward += result;
            result = int( not result); 
            node = node->_parent;
        }
    }

    void deleteTree(Node* node){
        if(node != nullptr){
            for(auto &n : node->children){
                deleteTree(n);
            }
            delete node;
        }
    }

    ~MCTS(){
        deleteTree(_root);
    }
};


#endif
