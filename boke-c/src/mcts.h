#include<string>
#include<vector>
#include"board.h"
#include"policy_net.h"
#include<torch/script.h>
#include<iostream>
#include<chrono>
#include<map>
#include<random>

#ifndef MCTS_H
#define MCTS_H

#define EMPTY_STATE "................................................................................."


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
    int _max_rollouts, _max_turns, _expand_thresh;
    float _alpha; //exploration weight
    Node* _root;
    torch::jit::script::Module _model;
    PolicyNet _network;
    std::map<std::string, std::vector<float>> _dist_map;

    MCTS(int rollouts, float alpha, torch::jit::script::Module &model, std::string state = EMPTY_STATE){
        _max_rollouts = rollouts;
        _max_turns  = 80;
        _expand_thresh = 10;
        _alpha = alpha;
        _root = new Node(state);
        _model = model;
        _network = PolicyNet();
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

    Node* selection(Node* node){
        int tot_visits = 0;
        Node* best;
        float best_puct = -1;
        for(auto &n : node->children){
            tot_visits += n->_visited;
        }
        for(auto &n : node->children){
            Board game = Board(9);
            game.loadState(node->_state, node->_turn, node->_lastmove, node->_ko);
            std::string state = node->_state;
            if(_dist_map.count(state) == 0){
                torch::Tensor x = _network.features(game);
                x = torch::unsqueeze(x, 0);
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(x);

                at::Tensor output = _model.forward(inputs).toTensor();
                output = torch::softmax(output, 1);
                output = torch::squeeze(output, 0).contiguous();

                std::vector<float> probs(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
                _dist_map.insert(std::pair<std::string, std::vector<float>>(state, probs));
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
    
    void expansion(Node* node){
        Node* temp;
        for(int i = 1; i<=9; i++){
            for(int j=1; j<=9; j++){
                Board game = Board(9);
                game.loadState(node->_state, node->_turn, node->_ko);
                if(game.play(node->_turn %2 + 1, i,j)){
                    temp = new Node(game.stateToString());
                    temp->_ko = game.getKo();
                    temp->_lastmove = (i-1)*9 + (j-1);
                    temp->_parent = node;
                    temp->_turn = game.getTurn();
                    node->children.push_back(temp);
                }
            }
        }
    }

    Node* descend(Node* node){
        Node* n = node;
        while(n->children.size() != 0){
            if (n->_visited > _expand_thresh){
                expansion(n);
            }
            n = selection(n);
        }
        return n;
    }

    void rollout(Node* node, int n_rolls){
        for (int i = 0; i< n_rolls; i++){
            Node* leaf = this->descend(node);
            int reward = this->simulation(leaf);
            this->backpropagate(leaf, reward);
        }
    }

    int getNextMove(Node* node){
        Board game = Board(9);
        game.loadState(node->_state, node->_turn, node->_lastmove, node->_ko);
        return getNextMove(game);
    }

    int getNextMove(Board &refBoard){
        Board game = Board(9);
        game.loadState(refBoard.getBoardString(), refBoard.getTurn(), refBoard.getLastMove(), refBoard.getKo());
        std::string state = game.getBoardString();
        if(_dist_map.count(state) == 0){
            torch::Tensor x = _network.features(game);
            x = torch::unsqueeze(x, 0);
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(x);

            at::Tensor output = _model.forward(inputs).toTensor();
            output = torch::softmax(output, 1);
            output = torch::squeeze(output, 0).contiguous();

            std::vector<float> probs(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
            _dist_map.insert(std::pair<std::string, std::vector<float>>(state, probs));
        }

        std::vector<float>* p = &_dist_map[state];
        unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::discrete_distribution<float> distribution(p->begin(), p->end());

        return distribution(generator);
    }

    int simulation(Node* node){
        Node* current = node;
        int invert = (node->_turn)%2;
        Board game = Board(9);
        game.loadState(node->_state, node->_turn, node->_lastmove, node->_ko);

        for(int i = node->_turn ; i<_max_turns; i++){
            int mv = getNextMove(game);
            game.play((i%2) + 1, mv);
        }

        // game.disp();
        
        //gnugo returns who wins 1 = B
        int reward = game.getScore();
        return invert^reward;
    }

    void backpropagate(Node* node, int result){
        while (node != nullptr){
            node->_visited++;
            node->_reward += result;
            result = 1 - result;
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