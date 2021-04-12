import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import constants as C

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(CriticNetwork, self).__init__()

        self.lr = lr
        self.input_shape = input_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*input_dims,40)
        self.batch1 = nn.LayerNorm(40)
        self.fc2 = nn.Linear(40,30)
        self.batch2 = nn.LayerNorm(30)
        self.fc3 = nn.Linear(30, 1)
    
        self.action_value = nn.Linear(n_actions, 30)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay=0.01)

        #self.initialize_weights_bias()
    def initialize_weights_bias(self):
    
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        self.fc3.weight.data.uniform_(-0.003, 0.003)
        self.fc3.bias.data.uniform_(-0.003, 0.003)

        f4 = 1/np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.batch2(x)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(x,action_value))
        state_action_value = self.fc3(state_action_value)

        return state_action_value
    

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, hd1_dims, hd2_dims,action_dim, agent_no):
        super(ActorNetwork, self).__init__()

        self.lr = lr
        self.input_dims = input_dims
        self.hd1_dims = hd1_dims
        self.hd2_dims = hd2_dims
        self.action_dim = action_dim
        self.agent_no = agent_no
        self.fc1 = nn.Linear(*self.input_dims, self.hd1_dims)
        self.fc2 = nn.Linear(self.hd1_dims, self.hd2_dims)
        self.fc3 = nn.Linear(self.hd2_dims, self.action_dim)

        self.nb1 = nn.LayerNorm(self.hd1_dims)
        self.nb2 = nn.LayerNorm(self.hd2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

        #self.initialize_weights_bias()
    def initialize_weights_bias(self):

        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)


        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.fc3.weight.data.uniform_(-f3,f3)
        self.fc3.bias.data.uniform_(-f3,f3)
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.nb1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.nb2(x)
        x = F.relu(x)
        #x = self.fc3(x)
        x = T.tanh(self.fc3(x))
        
        return C.MAX_ACTION[self.agent_no]*x 
  