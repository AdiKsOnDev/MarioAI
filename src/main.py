import numpy as np
import gym_super_mario_bros

enviroment = gym_super_mario_bros.make("SuperMarioBros-v0")

# Globals
ALPHA = 0.1
GAMMA = 0.6
EPSILON = 0.05

def q_learning(enviroment, num_states, num_actions, num_of_episodes=1000):
    # Initialize Q-table
    q_table = np.zeros((num_states, num_actions))

    for episode in range(0, num_of_episodes):
        # Reset the enviroment
        state = enviroment.reset()
        enviroment.render()
        
        # Initialize variables
        terminated = False

        while not terminated:
            # Pick action a....
            if np.random.rand() < EPSILON:
                action = enviroment.action_space.sample()
            else:
                max_q = np.where(np.max(q_table[state]) == q_table[state])[0]
                action = np.random.choice(max_q)

            # ...and get r and s'    
            next_state, reward, terminated, _ = enviroment.step(action)
            
            # Update Q-Table
            q_table[state, action] += ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state
            rewards[episode] += reward
            
    return rewards, q_table

observation_space = enviroment.observation_space.shape[0]
action_space = enviroment.action_space.n

q_reward, q_table = q_learning(enviroment, observation_space, action_space)