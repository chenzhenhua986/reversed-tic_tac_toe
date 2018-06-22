# Reversed tic-tac-toe
Solving reversed tic-tac-toe by using Reinforcement Learning. For the second player, the winning rate is above 95% by training 20000 iterations. The winning rate is almost 100% by training 50000 iterations. The current version only works for 3 by 3, but can be easily extended to larger dimensions. 

The RL version used here is Monte Carlo, namely, we give rewards only after each episode. We also adopt diversified initialization to make sure the player can make the optimal choice no matter what current state it is in.
