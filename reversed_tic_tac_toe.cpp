#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cfloat>

static const int num_episode = 50000;
static const int b_size = 3;

// state, 3*3=9, aciton (x, y, 1) or (x, y, -1), toal reward sum and its count
static std::vector<std::vector<std::vector<float>>> q_value;
static const float player1 = 1;
static const float player2 = -1;
static const float p_reward = 1.0;
static const float n_reward = -1.0;

std::vector<float> gen_rand(){
    std::vector<float> r;
    r.push_back(rand() % b_size);
    r.push_back(rand() % b_size);
    return r;
}

bool has_conflict(const std::vector<float>& r, const std::vector<float>& board){
    if(board.at(r.at(0)*b_size + r.at(1)) != 0){
        return true;
    }
    return false;
}

// select next move randomly
std::vector<float> update_board(std::vector<float>& board, const float& player_id){
    std::vector<float> r;
    while(has_conflict(r = gen_rand(), board)){
        continue;
    }
    board.at(r.at(0) * b_size + r.at(1)) = player_id;
    r.push_back(player_id);
    return r;
}

// check if one episode ends given current board
bool episode_end(const std::vector<float>& board){
    // board is full
    auto it = find(board.begin(), board.end(), 0);
    if(it == board.end()){
        return true;
    }
    
    // row, column is finished
    for(int i=0;i<b_size;i++){
        if((board.at(i*b_size) == 1 && board.at(i*b_size+1) == 1 && board.at(i*b_size+2) == 1) ||
           (board.at(i*b_size) == -1 && board.at(i*b_size+1) == -1 && board.at(i*b_size+2) == -1)){
            return true;
        }
        
        if((board.at(i) == 1 && board.at(i+b_size) == 1 && board.at(i+2*b_size) == 1) ||
           (board.at(i) == -1 && board.at(i+b_size) == -1 && board.at(i+2*b_size) == -1)){
            return true;
        }
    }
        
    // diagonal is finished
    if((board.at(0) == 1 && board.at(4) == 1 && board.at(8) == 1) ||
       (board.at(0) == -1 && board.at(4) == -1 && board.at(8) == -1)){
        return true;       
    }
    return false;
}

std::vector<float> init_board(){
    std::vector<float> board(b_size * b_size, 0);
    
    // shuffle the indices
    std::vector<int> indices;
    for(int i=0;i<b_size*b_size;i++){
        indices.push_back(i);
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);  
    
    // player1 play 0~5 times, player2 play 0~4 times
    int times = rand()%4;
    for(int i=0;i<times;i++){
        board.at(indices.at(i)) = player2;
    }
    for(int i=0;i<times+1;i++){
        board.at(indices.at(i+times)) = player1;
    }
    return board;
}

void insert_q(const std::vector<float>& state, const std::vector<float>& action, const std::vector<float>& reward_count_pair){
    bool existing = false;
    for(auto it1=q_value.begin(); it1 != q_value.end(); it1++){
        auto it2 = it1->begin(); 
        std::vector<float>& s = *(it2);
        std::vector<float>& a = *(it2+1);
        std::vector<float>& r_c_pair = *(it2+2);
        if(state == s && a == action){
            existing = true;
            r_c_pair.at(0) += reward_count_pair.at(0);
            r_c_pair.at(1) += reward_count_pair.at(1);
            //std::cout<<"find the same record: "<<q_value.size()<<std::endl;
            break;
        }
        if(existing) break;
    }
    
    // insert a new record
    if(!existing){
        std::vector<std::vector<float>> tmp;
        tmp.push_back(state);
        tmp.push_back(action);
        tmp.push_back(reward_count_pair);
        q_value.push_back(tmp);
    }
}

int play(std::vector<float>& board){
    std::vector<float> s = board;
    std::vector<float> a = update_board(board, player2);
    std::vector<float> r_c_pair;
    int count = 1;

    if(episode_end(board)){
        r_c_pair.push_back(n_reward);
        r_c_pair.push_back(count);
        insert_q(s, a, r_c_pair);
        return 1; 
    }
    
    while(!episode_end(board)){
        // player1
        update_board(board, player1);
        if(episode_end(board)){
            r_c_pair.push_back(p_reward);
            r_c_pair.push_back(count);
            insert_q(s, a, r_c_pair);
            return -1;
        }
        
        // player2 
        update_board(board, player2);
        if(episode_end(board)){
            r_c_pair.push_back(n_reward);
            r_c_pair.push_back(count);
            insert_q(s, a, r_c_pair);
            return 1;
        }
    }
}

void print_board(const std::vector<float>& board){
    for(int i=0;i<b_size;i++){
        for(int j=0;j<b_size;j++){
            std::cout<<board.at(i*b_size + j)<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"--------------"<<std::endl;
}

int main()
{
    // select a starting state randomly
    std::cout<<"------start training "<<num_episode<<" episodes-------"<<std::endl;
    for(int i=0; i<num_episode; i++){
        std::vector<float> starting_state;
    
        while(episode_end(starting_state = init_board())) continue;
        
        //print_board(starting_state);
        //std::cout<<"episode: "<<i<<", "<<"q_value size: "<<q_value.size()<<std::endl;
        
        //for each episode, play untl to the end
        play(starting_state);
    }
    
    std::cout<<"------training ends-------"<<std::endl;
    std::cout<<"------play 100 rounds randomly-------"<<std::endl;
    // iterate q_value, find the best policy, play 100 rounds
    int win_times = 0;
    int lose_times = 0;
    int rounds = 100;
    for(int i=0; i<rounds; i++){
        std::vector<float> board(9, 0);
        // player1 plays first randomly
        std::vector<float> first_step = gen_rand();
        board.at(first_step.at(0) * b_size + first_step.at(1)) = 1;
        while(!episode_end(board)){
            float largest_rwd = -FLT_MAX;
            std::vector<float> best_action;
            for(auto it1=q_value.begin(); it1 != q_value.end(); it1++){
                auto it2 = it1->begin();
                std::vector<float>& s = *(it2);
                std::vector<float>& a = *(it2+1);
                std::vector<float>& r_c_pair = *(it2+2);
                if(s == board && r_c_pair.at(0) / r_c_pair.at(1) > largest_rwd){
                    best_action = a;
                    largest_rwd = r_c_pair.at(0) / r_c_pair.at(1);
                    //std::cout<<"average rewards: "<<r_c_pair.at(0) / r_c_pair.at(1)<<std::endl;
                }
            }
            
            //player2 update board according to the best_action
            //cannot find the optimal action, just chose one randomly
            if(best_action.empty()){
                update_board(board, player2);
            }else{
                board.at(best_action.at(0) * b_size + best_action.at(1)) = best_action.at(2);
            }
            if(episode_end(board)){
                lose_times++;
                break;
            }
            
            // player1 update board randomly
            update_board(board, player1);
            if(episode_end(board)){
                win_times++;
                break;
            }
        }
    }
    std::cout<<"win percentage: "<<float(win_times)/rounds*100<<"%"<<std::endl;
    std::cout<<"lose percentage: "<<float(lose_times)/rounds*100<<"%"<<std::endl;
    return 0;
}

