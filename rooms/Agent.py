#! ../../gym_env/python
import pandas as pd

class QLAgent(object):
    '''Q-Learning Agent
    :states number of states
    :actions number of actions
    :alpha learning rate
    :gamma just gamma
    '''
    def __init__(self, states:int, actions:int, alpha:float, gamma:float):
        self.table = pd.DataFrame(columns=list(range(actions)), index=list(range(states)), dtype=float)
        self.table.fillna(value=0, inplace=True)
        self.action_num = actions
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state:int)->int:
        action = self.table.iloc[state].idxmax(1)
        return action
    
    def update(self, state, action, new_state, reward):
        remember = self.table.iloc[state, action] * (1 - self.alpha)
        learning = self.alpha * (reward + self.gamma * self.get_qvalue(new_state))
        q_value = remember + learning
        self.set_value(state, action, q_value)

    def get_qvalue(self, new_state:int)->float:
        value = self.table[new_state].max()
        return value
    
    def set_value(self, state, action, q):
        self.table.iloc[state, action] = q
        
    def save(self, path):
        self.table.to_csv(path)
    
    def load(self, path):
        self.table = pd.read_csv(path)

if __name__ == "__main__":
    qt = QLAgent(6, 6, 0.8, 0.8)
    qt.set_value(2, 3, 1.9)
    qt.update(3, 4, 2, 2.7)
    print(qt.table)
    print(qt.get_action(2))