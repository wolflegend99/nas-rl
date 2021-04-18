import numpy as np

class ReplayBuffer():
  def __init__(self, max_size, state_shape, action_shape):
    self.mem_size = max_size
    self.mem_cntr = 0
    self.state_mem = np.zeros((self.mem_size,*state_shape))
    self.next_state_mem = np.zeros((self.mem_size,*state_shape))
    self.action_mem = np.zeros((self.mem_size,action_shape))
    self.reward_mem = np.zeros((self.mem_size))
    #self.terminal_mem = np.zeros((self.mem_size), dtype= np.bool)
  
  def store_transition(self, state, action, reward, next_state):
    index = self.mem_cntr % self.mem_size
    self.state_mem[index] = state
    self.next_state_mem[index] = next_state
    self.action_mem[index] = action
    self.reward_mem[index] = reward
    #self.terminal_mem[index] = done

    self.mem_cntr += 1
  
  def sample_buffer(self, batch_size):
    sampling_size = min(self.mem_cntr, self.mem_size)
    sample_index = np.random.choice(sampling_size, batch_size)
     
    states = self.state_mem[sample_index]
    next_states = self.next_state_mem[sample_index]
    actions = self.action_mem[sample_index]
    rewards = self.reward_mem[sample_index]
    #terminals = self.terminal_mem[sample_index]

    return states, actions, rewards, next_states
    #, terminals