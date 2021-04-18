import constants as C
import torch
import numpy as np
from math import sqrt

params = {'batch_size': C.TRAIN_BATCH_SIZE,
          'shuffle': True}

def load(X_train, X_test, y_train, y_test):

    # Numpy to Tensor Conversion (Train Set)
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).view(-1, 1)
    
    # Numpy to Tensor Conversion (Test Set)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test).view(-1, 1)

    # Make torch datasets from train and test sets
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    
    # Create train and test data loaders
    train_loader = torch.utils.data.DataLoader(train, **params)
    test_loader = torch.utils.data.DataLoader(test, **params)
    return train_loader, test_loader

#0 is reward for layers
#1 is reward for nodes
def reward(train_acc, train_loss, test_acc, test_loss,next_state, agent_no, in_dims, out_dims, action,current_neurons):
  
  	value = 0
  	value += (test_acc/100)*10
  	value -= train_loss*7
  	value -=1.5**(agent_no+1)*next_state[agent_no]
  	return value
    #return np.random.randint(-10, 10)
    #trainable = (in_dims + out_dims)*x[1] + x[1]*x[1]*x[0] + out_dims + x[1]*x[0]
    #value = 0
    #if which == 0:
    #    value -= (x[0]/10)*2.5
    #    if action <=0 and main_param == 1:
    #        value -= 500
    #else:
    #    value -= ((x[1]**2)/100)*6.25
    #    if action <=0 and main_param == 1:
    #        value -= 500
            
    #value -= train_loss*7
    #value += (test_acc/100)*10

def sample(data, limit=C.SAMPLE_SIZE):
    sample = data.sample(n=limit)
    return sample

def shuffle(data):
    return data.sample(frac=1)

def get_dimensions(data):
    return data.shape
    
def print_debug(steps, agent_no, action, next_state, reward, returns):
    print("Step: ", steps)
    print("Agent : ", agent_no)
    print("Action: ", action)
    print("Next state : ", next_state)
    print("Agent ", agent_no, " -> Reward ", reward)
    print("Agent ", agent_no, " -> Returns ",returns)
    print("\n-----------------------------------------------------------------\n")

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma = 0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)