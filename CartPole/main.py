#! ../../gym_env/python
import numpy as np
from agent import Agent
from matplotlib import pyplot as plt
import gym

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    env = env.unwrapped
    state_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    memory_size = 500
    agent = Agent(state_dim, action_num,epsilon=0.6, memory_size=500, alpha=0.92, gamma=0.8, lr=0.002, batch_size=16)
    agent.load("script/CartPole/dqn_model.pth")
    reward_list_1 = []
    reward_list_2 = []
    
    for epoch in range(100):
        state = env.reset()
        total_reward = 0.
        tr = 0
        while True:
            env.render()

            action = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            pos = new_state[0]
            ang = new_state[2]

            r1 = (env.x_threshold - abs(pos)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(ang)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            agent.memorize(state, action, r, new_state)
            total_reward += r
            tr += reward

            if agent.memory_counter > agent.memory_size:
                agent.learn()
            if done:
                reward_list_1.append(total_reward)
                reward_list_2.append(tr)
                print(f"Epoch {epoch}, total reward {round(total_reward, 2)} --- {tr}")
                break
            state = new_state

    plt.subplot(1,2,1)
    plt.plot(reward_list_1)
    plt.title("reward self-designed")
    plt.ylabel("reward")
    plt.xlabel("epoch")

    plt.subplot(1,2,2)
    plt.plot(reward_list_2)
    plt.title("real reward")
    plt.ylabel("reward")
    plt.xlabel("epoch")

    plt.show()
    plt.savefig("script/CartPole/result.png")
    agent.save_model("script/CartPole/")
    env.close()
