import torch as T
import numpy as np
from agent import DDPGAgent
import constants as C
class MADDPG:

    def __init__(self, env, num_agents, alpha, beta, tau, input_dims, n_actions,
                 hd1_dims = 400, hd2_dims = 300, mem_size = 1000000,
                 gamma = 0.99, batch_size = 64):
        self.env = env
        self.num_agents = int(num_agents)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.agents = [DDPGAgent(alpha = self.alpha, beta = self.beta, tau = self.tau, 
                                 input_dims = input_dims, n_actions = n_actions, hd1_dims = hd1_dims,
                                 hd2_dims = hd2_dims, mem_size = mem_size, gamma = self.gamma,
                                 batch_size = self.batch_size, agent_no = i) for i in range(self.num_agents)]
        self.agents_states = []
        self.local_agent_states = [[] for i in range(self.num_agents)]
    
    def run(self, max_episode, max_steps):
        return_list = [[] for i in range(self.num_agents)]
        means = [0 for i in range(self.num_agents)]
        
        for i in range(max_episode):
            print("Episode : {}".format(i))
            returns = [0 for i in range(self.num_agents)]
            self.agent_states = self.env.reset() 
            self.local_agent_states = [self.agent_states for i in range(self.num_agents)]
            steps = 0
            agent_no = 0
            for i in range(self.num_agents):
                self.agents[i].actionNoise.reset()
            while steps != self.num_agents * max_steps:
                steps += 1
                action, rounded_action = self.agents[agent_no].choose_action(self.local_agent_states[agent_no], agent_no)
                print("Step: ", steps)
                print("Agent : ", agent_no)
                print("Action: {}, Rounded Action : {}".format(action, rounded_action))
                next_state, reward = self.env.step(rounded_action, agent_no)
                done = False
                if reward == 0:
                    done = True
                self.agents[agent_no].store_transition(self.local_agent_states[agent_no], action, reward, next_state)
                print("Next state : ", next_state)
                self.agents[agent_no].learn([self.local_agent_states[agent_no]], [action], [reward], [next_state])
                returns[agent_no] += reward
                self.local_agent_states[agent_no] = next_state
                
                #debug info goes here...
                print("Agent ", agent_no, " -> Reward", reward)
                print("Agent ", agent_no, " -> Returns ",returns[agent_no])
                if steps % C.SYNCH_STEPS == 0:
                    print("Syncing at step ", steps, "...")
                    self.agents_states = self.env.synch()
                    self.local_agent_states = [self.agents_states for i in range(self.num_agents)]
                agent_no = (agent_no + 1) % self.num_agents
                #steps+=1
                print("\n-----------------------------------------------------------------\n")
                
            for i in range(self.num_agents):
                return_list[i].append(returns[i])
                means[i] = np.mean(return_list[i][-20:])
            for i in range(self.num_agents):
                print("Score model {} : {}".format(i, means[i]))

    def run_parallel(self, max_episode, max_steps):
        
        arr = Array('i', [3,3])
        m = Manager()
        printlock = m.Lock()
        synchlock = m.Lock()
        for i in range(self.num_agents):
            p = Process(target = self.run, args = (max_episode, max_steps, i,
                                                printlock, synchlock, arr, self.env,
                                                self.agents))
            p.start()
            p.join()