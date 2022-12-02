import gym_2048
import gym
import torch
from dqn_model import DQN
from new_action import make_action

if __name__ == '__main__':
  env = gym.make('2048-v0')
  env.seed()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  online_net = DQN(size_state=4, num_actions=4)
  online_net.load_state_dict(torch.load('./trained_model.pth'))
  online_net.eval()
  obs = env.reset()
  env.render()

  done = False
  moves = 0
  count_action = 0
  old_action = 0
  while not done:
    action = make_action(online_net, obs, device)
    if old_action == action:
      count_action +=1
    
    if count_action > 10:
      action = env.np_random.choice(range(4), 1).item()
      count_action = 0

    next_state, reward, done, info = env.step(action)
    moves += 1
    obs = next_state
    old_action = action

    print('Next Action: "{}"\n\nReward: {}'.format(
      gym_2048.Base2048Env.ACTION_STRING[action], reward))
    env.render()

  print('\nTotal Moves: {}'.format(moves))