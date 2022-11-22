import gym_2048
import gym
from action import epsilon_greedy
from collections import defaultdict
import numpy as np


if __name__ == '__main__':
  env = gym.make('2048-v0')
  env.seed(42)

  observation = env.reset()
  env.render()
  returns_sum = defaultdict(float)
  returns_count = defaultdict(float)
  epsilon = 1
  done = False
  moves = 0
  gamma = 1
  Q = defaultdict(lambda: np.zeros(env.action_space.n))
  nA = env.action_space.n
  while not done:
    states, actions, rewards = [], [], []
    returns = 0
    epsilon = epsilon-(0.1/100)
    while True:
      states.append(observation)
      action = epsilon_greedy(Q, observation, nA, epsilon)
      actions.append(action)
      observation, reward, done, _  = env.step(action)
      rewards.append(reward)

      if done:
        break

    for j in range(len(states) - 1, -1, -1):
      S = states[j]
      R = rewards[j]
      A = actions[j]
      state_action = (S,A)
      returns = gamma*returns + R

      if state_action not in [(states[k],actions[k]) for k in range(j)]:
        returns_sum[S,A] += returns
        returns_count[S,A] += 1
        Q[S][A] = (returns_sum[S,A])/returns_count[S,A]


    next_state, reward, done, info = env.step(action)
    moves += 1

    print('Next Action: "{}"\n\nReward: {}'.format(
      gym_2048.Base2048Env.ACTION_STRING[action], reward))
    env.render()

  print('\nTotal Moves: {}'.format(moves))