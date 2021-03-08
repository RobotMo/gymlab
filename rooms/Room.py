#! ../../gym_env/python

class Room(object):
    def __init__(self, init_state:int, env:list):
        self.state = init_state
        self.init_state = init_state
        self.env = env
        self.action_space = list(range(len(env)))
        self.state_space = list(range(len(env)))
    
    def step(self, action:int):
        reward = self.env[self.state][action]
        done = reward < 0 or reward==100
        state = action if reward >=0 else self.state
        self.state = state
        info = ""
        return state, reward, done, info

    def reset(self):
        self.state = self.init_state
        return self.state

    
if __name__ == "__main__":
    env = [
        [-1, -1, -1, -1, 0, -1],
        [-1, -1, -1, 0, -1, 100],
        [-1, -1, -1, 0, -1, -1],
        [-1, 0, 0, -1, 0, -1],
        [0, -1, -1, 0, -1, 100],
        [-1, 0, -1, -1, 0, 100]
    ]
    r = Room(2, env)
    print(r.action_space)