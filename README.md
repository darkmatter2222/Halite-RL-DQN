# Halite-RL-DQN :robot:
Exploring Deep Rnforcement Larning to play Halite, specifically the version of Halite on:  
https://www.kaggle.com/c/halite  
https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/halite  

# Baby Steps First :baby:
This was my first attempt at reinforcement learning, before chomping off a huge bite, lets train a simple bot. Yes, 
I studied Q Learning and Q tables, and jumped right into  Deep Reinforcement Learning.:laughing:  
https://github.com/darkmatter2222/Halite-RL-DQN/blob/master/susman_rl/dqn_bots/find_the_dot_v0/dqn.py  
This is a POC of sample code from the tf_agents documentation and my idea for a good challenge similar to Halite::thinking:  
https://www.tensorflow.org/agents/overview  
  
![Baby Steps]( https://imgur.com/7TEi2NT.gif) (**Speed reduced for GIF representation**)  
https://www.youtube.com/watch?v=-33ehDNT3kY  
  
**Goal:** The white dot to find the green dot  
**Avoid:** Falling off the map or taking too many steps  

* On a 5x5 Grid and a typical 2018-2020 CPU/GPU, training time can take ~4k steps and 10-15 minutes. w/ >95% Win rate.    
* On a 6x6 Grid and a typical 2018-2020 CPU/GPU, training time can take ~12k steps and 40-60 minutes w/ >95% Win rate.  
* ...  
* On a 15x15 Grid and a typical 2018-2020 CPU/GPU, training time can take ~10M-20M steps and 1-2 days w/ >95% Win rate.:woozy_face:  
  * Can anyone help improve this?  

# Try 'Baby Steps' yourself? :desktop_computer:
1. Config directories here:  
   1. https://github.com/darkmatter2222/Halite-RL-DQN/blob/master/susman_rl/dqn_bots/find_the_dot_v0/config.json  
2. Set board size here:  
   1. https://github.com/darkmatter2222/Halite-RL-DQN/blob/master/susman_rl/environments/find_the_dot_v0/env.py  
3. Train bot here:  
   1. https://github.com/darkmatter2222/Halite-RL-DQN/blob/master/susman_rl/dqn_bots/find_the_dot_v0/train_model.py  
4. Evaluate bot here:  
   1. https://github.com/darkmatter2222/Halite-RL-DQN/blob/master/susman_rl/dqn_bots/find_the_dot_v0/execute_trained_model.py








