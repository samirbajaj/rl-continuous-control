[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Project 2: Continuous Control

### Introduction

For this project, I worked with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) Unity environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

This project uses the version containing 20 identical agents, each with its own copy of the environment.  

### Solving the Environment

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

The solution was developed in Python 3.6.9 on a Linux machine running Ubuntu 18.04. Instructions for setting up the required Python modules are available [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). In addition, `PyQt5` needs to be installed for use by `matplotlib`. 

Finally, [the 20-agent Unity environment for Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) should be downloaded and decompressed in the working directory. 

### Training the Agent

The `train.py` script offers a help message:
```commandline
$ python train.py -h
usage: train.py [-h] [--train]

DDPG - Udacity Reinforcement Learning Nanodegree.

optional arguments:
  -h, --help  show this help message and exit
  --train     train the Actor Critic Model
``` 
The script can be run in two modes, depending on whether the `--train` argument has been specified.

To train the agent, simply invoke the script with the `--train` argument. Doing so will start the training loop that runs until the desired minimum average score is recorded, or 5,000 training episodes have been completed. If an average score of +30 over 100 consecutive episodes is obtained, training stops and model parameters are written to disk. The parameters for the Actor network are written to `checkpoint_actor.pth`, and those for the Critic network are written to `checkpoint_critic.pth`.

These saved parameters are available in the repository.
