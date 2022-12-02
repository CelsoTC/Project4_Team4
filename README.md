# Project 4 Team 4
Project 4 for reinforcement learning

## Environment information

The package `gym-2048` does not work with newer versions of Python. For now I have installed `python==3.7` in a new environment (I know version 3.10 does not work since it requires numpy 1.14)

## Files

- `main.py` has an example from gym-2048 on how to run the environment
- `dqn_model.py` has a simple neural network to obtain the action. Since the state is a 4x4 matrix there is a flatten in the beginning and then the network will find the best action
- `replay_memory.py` has a replay memory buffer implementation with functions to add experience to the buffer and sample experiences with some batch size from the buffer
- `agent_dqn.py` has the training loop for the neural network (it is still in progress)
- `new_action.py` has the function to take an action from the neural network based on the observation (it is used in the main.py file)
- `plots.ipynb` Jupyter notebook for plotting the reward
- `test.py` - used just to add some quick funtion for tests
- `reward.npy` - File with the rewards saved during training
- `play_game.ipynb` - File with user input episode and runs the episode similar to main.py but from jupyter notebook
- `game.npy` - One user played game state transition saved


## Important links

https://pypi.org/project/gym-2048/

https://github.com/voice32/2048_RL

https://github.com/SergioIommi/DQN-2048

https://github.com/navjindervirdee/2048-deep-reinforcement-learning

https://tjwei.github.io/2048-NN/ - This link also has a repository, however, it's a complex model

https://github.com/FelipeMarcelino/2048-Gym - Rendering the episodes
