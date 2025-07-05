"""
Q-Learning Agent for FrozenLake-v1 (8x8, deterministic)

This script implements a tabular Q-learning algorithm using an epsilon-greedy policy
on the FrozenLake-v1 environment from OpenAI Gym. The agent learns to reach the goal
while avoiding holes on a frozen lake. The results are plotted as a learning curve.

Author: Finn Stäcker
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_q_learning(episodes: int = 5000, map_name: str = "8x8"):
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False, render_mode=None)
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    alpha = 0.9   # learning rate
    gamma = 0.9   # discount factor
    epsilon = 1.0 # exploration rate
    epsilon_decay = 0.00001

    rewards = np.zeros(episodes)
    rng = np.random.default_rng()

    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            best_next = np.max(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * best_next - q_table[state, action])

            state = next_state

        epsilon = max(epsilon - epsilon_decay, 0)
        if epsilon == 0:
            alpha = 0.0001  # stabilize learning

        rewards[episode] = reward

    env.close()

    # Plot rolling average of rewards (success rate)
    rolling_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    plt.plot(rolling_avg)
    plt.title("Q-Learning Success Rate on FrozenLake")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (100 ep avg)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("q_learning_frozenlake.png")
    plt.show()

    # Demonstration
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False, render_mode="human")
    state = env.reset()[0]
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)

    env.close()

if __name__ == "__main__":
    run_q_learning(episodes=5000)