# nas-rl

### Despcription
Neural Architecture Search poses a problem for Deep Learning. We use a Reinforcement Learning method to solve this issue, with multiple worker agents. The approach uses multiple DDPG agents, which are run by a controller and treats the environment of the dataset as a input to work on. 

- Model 1: variable number of nodes, fixed number of hidden layers
- Model 2: variable layers, variable (same number of nodes in each hidden layer)

```
Framework: PyTorch
Mode of Update: Asynchronous
Algorithm: Multi-Agent DDPG
Environment: Churn Modelling Dataset
Neural Network: Feed Forward
Sampling: Replay Buffer
Type of Learning: Actor-Critic Method

The controller can be trained on GPUs as well.
```
### Installation
```
git clone https://github.com/wolflegend99/nas-rl.git

```
