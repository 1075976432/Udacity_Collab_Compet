from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from ddpg_agent import Agent
from ddpg_agent import ReplayBuffer
import torch
import matplotlib.pyplot as plt


env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", no_graphics=True)
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
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128
memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, 2)
agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=2, memory=memory)
agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=2, memory=memory)


def maddpg(n_episodes=50, print_every=100, score_threshold=0.5):
    scores_deque = deque(maxlen=print_every)
    avg_score = []
    score = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent1.reset()
        agent2.reset()
        episode_score = np.zeros(num_agents)
        while True:
            action1 = agent1.act(state[0])[0]
            action2 = agent2.act(state[1])[0]
            env_info = env.step([action1, action2])[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent1.step(state[0], action1, reward[0], next_state[0], done[0])
            agent2.step(state[1], action2, reward[1], next_state[1], done[1])
            state = next_state
            episode_score += reward
            if np.any(done):
                break
        episode_score = np.max(episode_score)
        scores_deque.append(episode_score)
        avg_score.append(np.mean(scores_deque))
        score.append(episode_score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if np.mean(scores_deque) >= score_threshold:
            print('\nEnvironment solved in {} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent1.actor_local.state_dict(), 'models/checkpoint_actor0.pth')
            torch.save(agent1.critic_local.state_dict(), 'models/checkpoint_critic0.pth')
            torch.save(agent2.actor_local.state_dict(), 'models/checkpoint_actor1.pth')
            torch.save(agent2.critic_local.state_dict(), 'models/checkpoint_critic1.pth')
            break
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    return score, avg_score


scores, avg_score = maddpg()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, label='episode_score')
plt.plot(np.arange(1, len(avg_score)+1), avg_score, c='r', label='avg_score')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.savefig('scores_plot.png')
env.close()