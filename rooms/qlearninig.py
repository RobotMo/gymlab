#! ../../gym_env/python

import Agent
import Room
from random import random, randint
from matplotlib import pyplot as plt


if __name__ == "__main__":
    config = [
        [-1, -1, -1, -1, 0, -1],
        [-1, -1, -1, 0, -1, 100],
        [-1, -1, -1, 0, -1, -1],
        [-1, 0, 0, -1, 0, -1],
        [0, -1, -1, 0, -1, 100],
        [-1, 0, -1, -1, 0, 100]
    ]
    env = Room.Room(2, config)
    agent = Agent.QLAgent(6, 6, 0.8, 0.7)
    epsilon = 0.7
    epoch = 100
    won = []
    for i in range(epoch):
        state = env.reset()
        action = agent.get_action(state)
        if random() < 1-epsilon:
            action = randint(0, 5)
        new_state, reward, done, _ = env.step(action)
        agent.update(state, action, new_state, reward)
        while not done:
            state = new_state
            action = agent.get_action(state)
            if random() < 1-epsilon:
                action = randint(0, 5)
            new_state, reward, done, _ = env.step(action)
            agent.update(state, action, new_state, reward)
        if reward == 100:
            won.append(1)
        else:
            won.append(0)

        
    agent.save('./model.csv')
            
    # testing
    win = 0.
    for i in range(1000):
        state = env.reset()
        while True:
            action = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            if not done:
                state = new_state
            else:
                if reward == 100:
                    win += 1.
                break
    print(win / 1000.)