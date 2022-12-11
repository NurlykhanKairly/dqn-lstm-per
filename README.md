# Team 19: DQN implementation with LSTM and Prioritized Experience Replay

## Dependencies 

| Dependencies  | Versions       |
| ------------- | ------------- |
| Gym Environment  | 0.24.0  |
| Tensorflow  | 2.11.0  |
| Python      | 3.8.10   |
| Wandb       | 0.13.5   | 

## Run 

To run the code: 

`python3 main.py`. 

The `memory.py` contains the Prioritized Experience Replay implementation. 

## Notes 

Wandb was used to graph the rewards while running the code. During the runs, you can skip the option of using Wandb. However, by doing so, there will be no graphs. To see the graphs, follow the Wandb installation guide. 

In the code, we mention two references to **DQN** and **DRQN (DQN + LSTM)** implementations. Please follow the guides on the respective repositories to run the codes and compare the graphs: 
1) 


## Conclusions 

The improvements on DQN have been tested on the **CartPole-v1** environment provided by OpenAI Gym API. 
