import numpy as np
import gym_super_mario_bros
import gym
import torch

enviroment = gym_super_mario_bros.make("SuperMarioBros-v0")
# enviroment = gym.make("CartPole-v0")

# Globals
ALPHA = 0.1
GAMMA = 0.6
EPSILON = 0.05

def q_learning(enviroment, num_states, num_actions, num_of_episodes=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_table = torch.zeros((num_states, num_actions), device=device)
    rewards = np.zeros(num_of_episodes)

    for episode in range(num_of_episodes):
        state = enviroment.reset()
        print(str(state))
        terminated = False
        count = 0

        while not terminated:
            print(count)

            enviroment.render()

            # Pick action a...
            if np.random.rand() < EPSILON:
                action = enviroment.action_space.sample()
            else:
                action = torch.argmax(q_table[state]).item()

            # ...and get r and s'
            next_state, reward, terminated, _ = enviroment.step(action)

            # Update Q-Table
            q_table[state, action] += ALPHA * (reward + GAMMA * torch.max(q_table[next_state]) - q_table[state, action])

            state = next_state
            rewards[episode] += reward
            count += 1

    return rewards, q_table

observation_space = enviroment.observation_space.shape[0]
action_space = enviroment.action_space.n

q_reward, q_table = q_learning(enviroment, 500, action_space)