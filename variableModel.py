import torch as T
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
  def __init__(self, input_dims, output_dims, hidden_dim_array = [], non_linear_func_array = []):
    super(Model, self).__init__()
    self.input_dims = input_dims
    self.non_linear_func_array = non_linear_func_array
    self.hidden_dim_array = hidden_dim_array
    self.output_dims = output_dims
    self.linear_functions = []
    self.non_linear_functions = [i() for i in self.non_linear_func_array]
    self.hidden_layers = len(self.hidden_dim_array)
    temp=self.input_dims
    for i in range(self.hidden_layers):
      self.linear_functions.append(nn.Linear(temp, self.hidden_dim_array[i]))
      temp = self.hidden_dim_array[i]
    self.linear_functions=nn.ModuleList(self.linear_functions)
    self.output_layer = nn.Linear(temp, self.output_dims)

  def forward(self, x):
    out = x
    for i in range(self.hidden_layers):
      out = self.linear_functions[i](out)
      out = self.non_linear_functions[i](out)
    out = self.output_layer(out)
    return out

  def train(self, X_train, X_test):
    pass
  
  def test(self):
    pass