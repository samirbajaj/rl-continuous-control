
# Project Report

### Introduction

This project implements DDPG to solve the Reacher Unity environment. The implementation borrows code from examples provided in the exercises.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Training the Agent

The agent was trained using Actor-Critic networks with Experience Replay and Fixed Q-Targets. It utilized a neural network with three fully-connected (dense) layers interspersed with ReLU activation. Each dense layer had output size 64, except the final one, which had one node for each action.

The network was tuned by reducing MSE loss using the Adam optimizer with a learning rate of 1e-3. Training was carried out in minibatches of size 256.

Here is the complete list of hyperparameters:
```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

UPDATE_EVERY = 20       # how often to update the network
UPDATE_TIMES = 10       # how many times to update the network each time

EPSILON = 1.0           # epsilon for the noise process added to the actions
EPSILON_DECAY = 1e-6    # decay for epsilon above
```
### Rewards

The image below plots the rewards per episode obtained by the agent during training. Console logs show that the agent is able to receive an average reward (over 100 episodes) of +30.06 in 239 episodes.

![Rewards Plot](rewards.png)

### Improving the Agent

To improve the agent, we can use algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  
