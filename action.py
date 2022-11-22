import numpy as np
from collections import defaultdict

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    prob = np.random.random()
    if prob < epsilon:
        action = np.random.choice(nA)
        
    else:
        action = np.argmax(Q[state])
    
    return action

def mc_action(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    nA = env.action_space.n

    for i in range(n_episodes):
        states, actions, rewards = [], [], []
        observation = env.reset()
        returns = 0
        epsilon = epsilon-(0.1/n_episodes)
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

    return Q