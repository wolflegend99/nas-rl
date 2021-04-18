import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from dataset import Dataset
import constants as C
import helper as H
from variableModel import Model

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class Environment():
  def __init__(self, num_agents, path='churn_modelling.csv'):
    self.path = path
    self.dataset = Dataset(path = self.path)
    self.X, self.y = self.dataset.preprocess()
    self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.split(self.X, self.y, fraction=0.2)
    self.X_train, self.X_test = self.dataset.scale(self.X_train, self.X_test)
    self.train_loader, self.test_loader = H.load(self.X_train, self.X_test, self.y_train, self.y_test)
    
    self.input_dims = [self.X.shape[1]]
    self.output_dims = len(np.unique(self.y))
    print("Dims of X_train is {}".format(H.get_dimensions(data = self.X_train)))
    print("Dims of y_train is {}".format(H.get_dimensions(data = self.y_train)))
    print("Input dims is {}, output dims is {}".format(self.input_dims, self.output_dims))
    self.num_agents = num_agents
    self.func = 'relu'
    self.agent_states = [[] for i in range(self.num_agents)]
    self.activations = [self.func]*self.num_agents
    #self.agent_states = nodes
    print("env initialised here ...")
    self.model = [Model(0.01, self.input_dims, self.output_dims, self.agent_states[i], self.activations, self.train_loader, self.test_loader) for i in range(self.num_agents)]
    for i in range(self.num_agents):
      self.model[i] = self.model[i].to(device)

  def reset(self):

    self.agent_states = [np.random.randint(C.MIN_NODES, C.MAX_NODES)]*self.num_agents

    for i in range(self.num_agents):
      self.model[i].initialise(self.agent_states)
      self.model[i] = self.model[i].to(device)

    return self.agent_states

  def reset1(self, agent_no):

    neurons = np.random.randint(C.MIN_NODES, C.MAX_NODES)

    self.model[agent_no].initialise(self.agent_states)
    self.model[agent_no] = self.model[agent_no].to(device)
    

  def sample_reset(self, agent_no, state_passed):

    self.agent_states[agent_no] = state_passed

    self.model[agent_no].initialise(self.agent_states[agent_no])
    self.model[agent_no] = self.model[agent_no].to(device)
    
    return state_passed

  def step(self, action, agent_no):
    state_, reward = self.change_neurons(action, agent_no)
    return (state_, reward)
  
  def synch(self):
    neurons = []
    for i in range(self.num_agents):
      neurons.append(self.model[i].hidden_layer_array[i])
    
    for i in range(self.num_agents):
      self.model[i].initialise(neurons)
      self.model[i] = self.model[i].to(device)
    
    return neurons
  
  def synch1(self, synched_state, agent_no):

    self.model[agent_no].initialise(synched_state)
    self.model[agent_no] = self.model[agent_no].to(device)
    self.agent_states[agent_no] = synched_state

    return synched_state

  def change_neurons(self, action, agent_no):
    
    current_nodes = self.model[agent_no].hidden_layer_array[agent_no]
    if action > 0:
        next_state = self.model[agent_no].add_neurons(int(action),agent_no)
    else:
        next_state = self.model[agent_no].remove_neurons(-int(action),agent_no)
    self.model[agent_no] = self.model[agent_no].to(device)
    train_acc, train_loss = self.model[agent_no].train()
    test_acc, test_loss = self.model[agent_no].test()
    reward = H.reward(train_acc, train_loss,
                      test_acc, test_loss,
                      next_state, agent_no,
                      self.X.shape[1], 
                      self.output_dims, 
                      action, current_nodes)
    print("Train_acc : ", train_acc)
    print("Test_acc : ", test_acc)
    print("Train_loss : ", train_loss)
    print("Test_loss : ", test_loss)
    return (next_state, reward)
  
  def seed(self):
    pass 