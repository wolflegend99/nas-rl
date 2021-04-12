from multiAgent import MADDPG
from environment import Environment
import constants as C

layers = input("Enter hidden layers: ")
layers = int(layers)
env = Environment(layers)

C.NUM_AGENTS = layers
C.MAX_ACTION = C.ACTION_SPACE*C.NUM_AGENTS

#call multiAgent here 
controller = MADDPG(env, num_agents=layers, alpha=C.ALPHA, beta=C.BETA, tau=C.TAU, input_dims=[C.NUM_AGENTS] ,n_actions=C.N_ACTIONS, hd1_dims = C.H1_DIMS, hd2_dims = C.H2_DIMS, mem_size = C.BUF_LEN,gamma = C.GAMMA, batch_size = C.BATCH_SIZE)

controller.run(max_episode=C.MAX_EPISODES, max_steps=C.MAX_STEPS)