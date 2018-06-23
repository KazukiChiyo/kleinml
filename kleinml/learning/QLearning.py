'''
Q-learning on a given reward matrix.
Author: Kexuan Zou
Date: Apr 7, 2018
Q matrix:
[[  0.         0.         0.         0.        81.0101     0.      ]
 [  0.         0.         0.        64.810104   0.       101.0101  ]
 [  0.         0.         0.        64.810104   0.         0.      ]
 [  0.        81.0101    51.6481     0.        81.0101     0.      ]
 [ 64.810104   0.         0.        64.810104   0.       101.0101  ]
 [  0.        81.0101     0.         0.        81.0101   101.0101  ]]
Optimal routes for all starting states:
State 0: 0 -> 4 -> 5
State 1: 1 -> 5
State 2: 2 -> 3 -> 1 -> 5
State 3: 3 -> 1 -> 5
State 4: 4 -> 5
State 5: 5
external source: http://people.revoledu.com/kardi/tutorial/ReinforcementLearning/Q-Learning-Example.htm
'''

import numpy as np

class QLearning(object):
    def __init__(self, alpha=1.0, gamma=0.01, epsilon=0.05, iter=1000, min_goal=1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.iter = iter
        self.min_goal = min_goal

    # evaluate q learning model based on the reward matrix
    def evaluate(self, reward):
        if type(reward) is not np.ndarray:
            reward = np.array(reward)
        self.r = reward.astype("float32")
        self.n_states, self.n_actions = np.shape(reward)
        self.q = np.zeros_like(self.r)
        self.train_q()
        return self.q

    # update q given transition function Q: state X action -> next_state
    def update_q(self, state, next_state, action):
        r_cur, q_cur = self.r[state, action], self.q[state, action]
        self.q[state, action] = (1-self.alpha)*q_cur + self.alpha*(r_cur + self.gamma*max(self.q[next_state]))

    # epsilon-greedy method to train for q
    def train_q(self):
        rseed = np.random.RandomState()
        for i in range(self.iter):
            goal = False
            states = list(range(self.n_states))
            rseed.shuffle(states)
            curr = states[0] # random starting state
            while not goal:
                actions = np.array(list(range(self.n_actions)))[self.r[curr] >= 0] # select all reachable next states
                if type(actions) is int: # if only one state left, make it an array
                    actions = [actions]
                if rseed.rand() < self.epsilon: # explore into new moves
                    rseed.shuffle(actions)
                    next = actions[0] # select random move as next step
                else: # exploit existing moves
                    if np.sum(self.q[curr]) > 0: # greedy traversal to choose the most rewarding step
                        next = np.argmax(self.q[curr])
                    else: # q is 0, initial state, make a random move
                        rseed.shuffle(actions)
                        next = actions[0]
                self.update_q(curr, next, next)
                if self.r[curr, next] > self.min_goal: # if current state is identified as a goal state, we are done
                    goal = True
                curr = next

    # given start state and goal state, find optomal route
    def get_route(self, start_state, goal_state, max_step=50):
        path = [start_state]
        step = 0
        while start_state != goal_state and step <= max_step:
            next_state = np.argmax(self.q[start_state])
            path.append(next_state)
            start_state = next_state
            step += 1
        return path, step

    # summarize the learning model
    def summary(self, goal_state, max_step=50):
        print("Q matrix:")
        print(model.q)
        print("")
        print("Optimal routes for all starting states:")
        for i in range(len(self.q)):
            curr, step = i, 0
            path = "%i -> " % curr
            while curr != goal_state and step <= max_step:
                next_state = np.argmax(self.q[curr])
                curr = next_state
                path += "%i -> " % curr
                step += 1
            path = path[:-4]
            print("State %i:" % i)
            print(path)

if __name__ == '__main__':
    reward = np.array([[-1, -1, -1, -1,  80,  -1],
        [-1, -1, -1,  64, -1, 100],
        [-1, -1, -1,  64, -1,  -1],
        [-1,  80,  51, -1,  80,  -1],
        [ 64, -1, -1,  64, -1, 100],
        [-1,  80, -1, -1,  80, 100]])
    model = QLearning()
    model.evaluate(reward)
    model.summary(goal_state=5)
