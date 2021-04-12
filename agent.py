from Network_model import CriticNetwork, ActorNetwork
from ReplayBuffer import ReplayBuffer
import torch
import numpy as np
import torch.nn.functional as F
from helper import OrnsteinUhlenbeckActionNoise
import constants as C

class DDPGAgent():
    def __init__(self, alpha, beta, tau, input_dims, n_actions, hd1_dims = 400, hd2_dims = 300, mem_size = 1000000, gamma = 0.99, batch_size = 64, agent_no=1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.localActor = ActorNetwork(self.alpha, input_dims, hd1_dims, hd2_dims, n_actions, agent_no)
        self.localCritic = CriticNetwork(self.beta, input_dims, n_actions)
        self.targetActor = ActorNetwork(self.alpha, input_dims, hd1_dims, hd2_dims, n_actions,agent_no)
        self.targetCritic = CriticNetwork(self.beta, input_dims, n_actions)

        self.replayBuffer = ReplayBuffer(mem_size, input_dims, n_actions)

        self.actionNoise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(n_actions))
        
        self.update_parameter_weights(tau = 0.99)
        
    def choose_action(self, observation, agent_no):
        self.localActor.eval()
        state = torch.tensor([observation], dtype = torch.float32)
        action = self.localActor.forward(state)
        noisy_action = action + torch.tensor(self.actionNoise(), dtype = torch.float32)

        self.localActor.train()
        #final_action = C.MAX_ACTION[agent_no]*noisy_action.detach().numpy()[0]
        final_action = noisy_action.detach().numpy()[0]

        return (final_action[0], np.round(final_action)[0])
    
    def store_transition(self, state, action, reward, next_state):
        self.replayBuffer.store_transition(state, action, reward, next_state)
    
    def learn(self, states, actions, rewards, next_states):

        #if self.replayBuffer.mem_cntr < self.batch_size:
        #    return

        #states, actions, rewards, next_states = self.replayBuffer.sample_buffer(self.batch_size)

        states = torch.tensor(states, dtype = torch.float)
        actions = torch.tensor(actions, dtype = torch.float)
        rewards = torch.tensor(rewards, dtype = torch.float)
        next_states = torch.tensor(next_states, dtype = torch.float)
        #done = torch.tensor(done)

        Q = self.localCritic.forward(states, actions)
        target_actions = self.targetActor.forward(next_states)
        Q_prime = self.targetCritic.forward(next_states, target_actions)

        #Q_prime[done] = 0.0
        #Q_prime = Q_prime.view(-1)
        y_prime = rewards + self.gamma*Q_prime
        #y_prime = y_prime.view(self.batch_size, 1)


        self.localCritic.optimizer.zero_grad()
        criticLoss = F.mse_loss(y_prime, Q)
        criticLoss.backward()
        self.localCritic.optimizer.step()

        self.localActor.optimizer.zero_grad()
        actorLoss = -self.localCritic.forward(states, self.localActor.forward(states))
        actorLoss = torch.mean(actorLoss)
        actorLoss.backward()
        self.localActor.optimizer.step()
    
        self.update_parameter_weights()
        
    def update_parameter_weights(self, tau = None):
        if tau is None:
            tau = self.tau
        actor_dict = self.localActor.state_dict()
        target_actor_dict = self.targetActor.state_dict()

        critic_dict = self.localCritic.state_dict()
        target_critic_dict = self.targetCritic.state_dict()

        for key in target_actor_dict:
            target_actor_dict[key] = tau*target_actor_dict[key] + (1-tau)*actor_dict[key]

        self.targetActor.load_state_dict(target_actor_dict)

        for key in target_critic_dict:
            target_critic_dict[key] = tau*target_critic_dict[key] + (1-tau)*critic_dict[key]
    
        self.targetCritic.load_state_dict(target_critic_dict)