'''
refer to inflearn.com/course/reinforcement-learning/lecture/5778
Q(state, action)

Q(s1, LEFT):0
Q(s1, RIGHT):0.5
Q(s1, UP):0
Q(s1, DOWN):0.3

Max Q = Q(s1, a) -> 0.5
π*(s) = argmaxQ(s1, a) -> RIGHT

Finding, Learning Q

Assume(believe)Q in s' exists!
    - I am in S
    - when I do action a, i'll go to s'
    - when I do action a, I'll get reward r
    - Q in s', Q(s', a')exist!
How can we express Q(s,a) using Q(s', a')?

Q(s,a) <- r + maxQ(s',a')

'''

import gym
from gym.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
import random as pr


def rargmax(vector):
    """
    https://gist.github.com/stober/1943451
    Argmax that chooses randomly among eligible maximum indices. because it may set 0 in all values,
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name':'4x4',
        'is_slippery':False
    }
)

env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n]) #(16,4)
# Set learning parameters
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        action = rargmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()