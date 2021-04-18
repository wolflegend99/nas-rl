import torch as T
import numpy as np
from agent import DDPGAgent
import constants as C
import helper as H
from torch.multiprocessing import Process, Lock, Value, Array, Manager, Pool, set_start_method
#try:
#    set_start_method('spawn')
#except RuntimeError:
#    pass

class MADDPG:
    def __init__(self, env, num_agents, alpha, beta, tau, input_dims, n_actions,
                 hd1_dims = 400, hd2_dims = 300, mem_size = 1000000,
                 gamma = 0.99, batch_size = 64):
        self.env = env
        self.num_agents = num_agents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.agents = [DDPGAgent(alpha = self.alpha, beta = self.beta, tau = self.tau, 
                                 input_dims = input_dims, n_actions = n_actions, hd1_dims = hd1_dims,
                                 hd2_dims = hd2_dims, mem_size = mem_size, gamma = self.gamma,
                                 batch_size = self.batch_size,agent_no = i) for i in range(self.num_agents)]
        self.agents_states = None
        self.x = 0
        
    def run(self, min_steps, max_steps, agent_no, l, l1, global_state):
        return_list = []
        means = 0
        env = self.env
        agent = self.agents[agent_no]
        for i in range(max_episode):
            print("Episode : {}".format(i+1))
            returns = 0
            state = env.reset1(agent_no)
            agent_states = state
            steps = 0
            self.agents[agent_no].actionNoise.reset()
            while steps != max_steps:
                steps += 1
                action, rounded_action = agents[agent_no].choose_action(agent_states, agent_no)
                next_state, reward = env.step(rounded_action, agent_no)
                done = False
                if reward >= 10:
                    done = True
                agents[agent_no].store_transition(agent_states, action, reward, next_state)
                agents[agent_no].learn([agent_states], [action], [reward], [next_state])
                returns += reward
                agent_states = next_state
                
                #debug info goes here...
                if steps % C.SYNCH_STEPS == 0:
                    l1.acquire()
                    try:
                      print("Syncing at step ", steps, "...")
                      common_state[agent_no] = agent_states[agent_no]
                      synched_state = [common_state[0], common_state[1]]
                      agent_states = env.synch1(synched_state, agent_no)
                    finally:
                      l1.release()
                l.acquire()
                try:
                  print("Episode: ", i+1)
                  print("Step: ", steps)
                  print("Agent : ", agent_no)
                  print("Action: ", action)
                  print("Next state : ", next_state)
                  print("Agent ", agent_no, " -> Reward ", reward)
                  print("Agent ", agent_no, " -> Returns ",returns)
                  print("\n-----------------------------------------------------------------\n")
                finally:
                  l.release()

            return_list.append(returns)
            means = np.mean(return_list[-20:])
            print("Score Model for {} : {}".format(agent_no, means))
 
    def run_parallel_episodes(self, max_episodes, max_steps):
        for episode in range(max_episodes):
            print("Episode : {}".format(episode+1))   
            agent_list = []
            for i in range(self.num_agents):
                agent_list.append(np.random.randint(C.MIN_NODES, C.MAX_NODES))

            arr = Array('i', agent_list)
            m = Manager()
            printlock = m.Lock()
            synchlock = m.Lock()
            all_processes = [Process(target = self.sample_run, args = (30, max_steps, printlock, synchlock, arr, episode, j)) for j in range(self.num_agents)]
            for p in all_processes:
              p.start()
              
            for p in all_processes:
              p.join()
            
            for p in all_processes:
              p.terminate()

    def sample_run(self, min_steps, max_steps, l, l1, global_state, episode, agent_no=0):
        return_list = []
        returns = 0
        means = 0
        steps = 0
        state_passed = global_state
        agent_states = self.env.sample_reset(agent_no, state_passed)
        self.agents[agent_no].actionNoise.reset()
        while steps != max_steps:
            steps += 1
            action, rounded_action = self.agents[agent_no].choose_action(agent_states, agent_no)
            next_state, reward = self.env.step(rounded_action, agent_no)
            self.agents[agent_no].store_transition(agent_states, action, reward, next_state)
            self.agents[agent_no].learn([agent_states], [action], [reward], [next_state])
            returns += reward
            agent_states = next_state
            
            #debug info goes here...
            if steps % C.SYNCH_STEPS == 0:
                l1.acquire()
                try:
                    print("Syncing at step ", steps, "...")
                    global_state[agent_no] = agent_states[agent_no]
                    synched_state = global_state
                    agent_states = self.env.synch1(synched_state, agent_no)
                finally:
                    l1.release()
            print_state = list(next_state)        
            l.acquire()
            try:
                H.print_debug(steps, agent_no, action, print_state, reward, returns)
            finally:
                l.release()

            return_list.append(returns)
        means = np.mean(return_list[-20:])
        print("Score Model(n=20) for agent {} : {}".format(agent_no, means))