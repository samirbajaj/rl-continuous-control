import argparse
from collections import deque

import matplotlib

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from ddpg_agent import Agent


def create_env(file_name="Reacher_Linux/Reacher.x86_64"):
    env = UnityEnvironment(file_name=file_name)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    return env, brain_name, num_agents, state_size, action_size


def train_ddpg(agent, env, num_agents, n_episodes=5000, max_t=1000, print_every=100):

    episode_scores = []                                    # list containing scores from each episode
    scores_window = deque(maxlen=print_every)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations              # get the current state (for each agent)
        agent.reset()
        score = np.zeros(num_agents)                       # initialize the score (for each agent)
        for t in range(max_t):
            actions = agent.act(states)                    # select an action (for each agent):
            # actions = np.clip(actions, -1, 1)            # all actions (already being clipped) between -1 and 1

            env_info = env.step(actions)[brain_name]       # send all actions to tne environment
            next_states = env_info.vector_observations     # get next state (for each agent)
            rewards = env_info.rewards                     # get reward (for each agent)
            dones = env_info.local_done                    # see if episode finished

            for i_agent in range(num_agents):
                agent.step(states[i_agent],
                           actions[i_agent],
                           rewards[i_agent],
                           next_states[i_agent],
                           dones[i_agent], t)              # update the system

            # CONSIDER: vectorize agent.step() so that we can replace the above for-loop with this single line:
            # agent.step(states, actions, rewards, next_states, dones, t)

            score += rewards                               # update the score (for each agent)
            states = next_states                           # roll over states to next time step
            if np.any(dones):                              # exit loop if episode finished
                break
        scores_window.append(score)
        episode_scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) > 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return episode_scores


def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDPG - Udacity Reinforcement Learning Nanodegree.')
    parser.add_argument("--train", help="train the Actor Critic Model", action="store_true")
    args = parser.parse_args()

    env, brain_name, num_agents, state_size, action_size = create_env()
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)
    if args.train:
        print(f"Commencing training loop...")
        scores = train_ddpg(agent, env, num_agents)
        plot_scores(scores)
    else:
        # load parameters
        agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

        # test drive for one episode
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        while True:
            actions = agent.act(states)  # select an action (for each agent)
            env_info = env.step(actions)[brain_name]  # send all actions to the environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    env.close()
