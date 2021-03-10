import gym
from agent import Agent
from matplotlib import pyplot as plt

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    env = env.unwrapped        # Remove the limit of the upper score
    state_dim = env.observation_space.shape[0]
    action_num = env.action_space.n

    # If you have trained a good enough model and used th parameter 
    # 'train=False' here, then when you test it, you will find that 
    # it can take a very long time (sometimes never to end) the game.
    agent = Agent(state_dim, action_num, train=False)
    agent.load("script/CartPole/dqn_model.pth")
    reward_list = []

    for epoch in range(100):
        state = env.reset()
        total_reward = 0
        while True:
            env.render()
            action = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = new_state
            if done:
                print(f"Epoch {epoch} Reward {total_reward}")
                reward_list.append(total_reward)
                break
    
    plt.figure()
    plt.plot(reward_list)
    plt.savefig("script/CartPole/test_result.png")
    plt.show()

