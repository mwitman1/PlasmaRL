# PlasmaRL
Actor-critic controller of thermal effects of an atomospheric pressure plasma jet

Supporting Code for:
Witman, M; Gidon, D.; Graves, D.B.; Smit, B.; Mesbah, A. "Sim-to-real transfer reinforcement learning for control of thermal effects of an atomospheric pressure plasma jet"

1. Train the RLC:
python train_ac_f18_plasma.py PlasmaModel -ep 100 -epext 1 --discount 0.99 -n 3000 -e 1 -l 2 -s 64 -b 10000 -lr 0.005 -ntu 10 -ngsptu 10 --exp_name test
  
    PlasmaModel     : use the plasma model dynamics files  
    -ep 100         : number of episode/roll-outs per training epoch  
    -epext 1        : number of timesteps at the end which are NOT included   
    --discount 0.99 : the discount factor  
    -n 3000         : number of training epochs  
    -e 1            : number of agents to train starting from different random seeds  
    -l 2            : number of hidden layers in actor/critic networks  
    -s 64           : nodes per hidden layer  
    -b 10000        : number of timesteps to be included in the batch of training data in each epoch  
    -lr 0.005       : learning rate  
    -ntu 10         : number of target updates (ntu), num. times the critic networks targets are updated per actor update  
    -ngsptu 10      : number of gradient steps per target update (ngsptu)  
    --exp_name test : data is saved in "./data/ac_test_PlasmaModel_*timestamp*/  

2. Validate the RLC on the physics model over randomized dynamics swith invivo=True to use the RLC to send messages to a socket):
python train_ac_f18_plasma.py PlasmaModel --execute data/ac_test_PlasmaModel_*timestamp*/1/

3. If you want to validate the RLC on a live device, set invivo=True in train_ac_f18_plasma.py and modify the destination for your message

4. The ./data folder ships with the E-RLC in the above publication

The code is quite specific to the application of the paper above since not much focus
has yet been placed to generalize it for different physics based models of plasma dynamics. 
Feel free to contact authors for advice/assistance using the code with different models.
