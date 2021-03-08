from models import QNet
import numpy as np
from os.path import join
import torch


class Agent(object):
    def __init__(self, state_dim, action_num, train=True, alpha=0.9, gamma=0.3, memory_size=100, batch_size=8, lr=0.001, epsilon=0.7):
        self.state_dim = state_dim
        self.action_num = action_num
        self.alpha = alpha
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.lr = lr
        self.eps = epsilon
        self.train = train

        self.qnet = QNet(state_dim, action_num)
        self.target_net = QNet(state_dim, action_num)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.lossfunc = torch.nn.MSELoss()

        self.memory = np.zeros((self.memory_size, self.state_dim*2+2))
        self.memory_counter = 0
        self.update_rate = 20
        self.update_step = 0

    def get_action(self, state):
        self.qnet.eval()
        x = torch.Tensor(state).unsqueeze(0)

        if  np.random.uniform() < self.eps or not self.train:
            act_values = self.qnet(x)
            actions = act_values.max(1)
            action = actions[1].item()
        else:
            action = np.random.randint(0,self.action_num)
        return action
    
    def memorize(self, state, action, reward, new_state):
        transition = np.hstack((state, [action, reward], new_state))

        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn(self):
        # if self.update_step <= self.memory_size:
        #     return

        self.qnet.train()
        self.update_step += 1
        if 0 == self.update_step % self.update_rate:
            self.target_net.load_state_dict(self.qnet.state_dict())
        
        sample_index = np.random.choice(self.memory_size, self.batch_size, replace=False)

        batch_memory = self.memory[sample_index, :]

        batch_state = torch.Tensor(batch_memory[:, :self.state_dim])
        batch_action = torch.Tensor(batch_memory[:, self.state_dim:self.state_dim+1]).type(torch.int64)
        batch_reward = torch.Tensor(batch_memory[:, self.state_dim+1:self.state_dim+2])
        b_new_state = torch.Tensor(batch_memory[:, -self.state_dim:])
        
        q = self.qnet(batch_state)
        # print(q.shape)
        # print(batch_action.shape)
        q = torch.gather(q, 1, batch_action)
        q_next = self.target_net(b_new_state)
        q_next = q_next.max(1)[0].view(self.batch_size, 1)
        q_target = (1. - self.alpha) * q + self.alpha * (batch_reward + self.gamma * q_next)
        loss = self.lossfunc(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()     

    def save_model(self, path="./"):
        path = join(path, "dqn_model.pth")
        torch.save(self.qnet, path)
    
    def load(self, path):
        self.qnet = torch.load(path)
        self.target_net = torch.load(path)