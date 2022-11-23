import gym_2048
import gym
from agent_dqn import Agent_DQN

if __name__ == '__main__':
  env = gym.make('2048-v0')
  env.seed(42)

  env.reset()
  env.render()
  agent = Agent_DQN(env)
  agent.train()