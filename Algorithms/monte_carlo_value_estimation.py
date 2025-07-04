"""
Monte Carlo Value Estimation with Epsilon-Greedy Policy Improvement

This script simulates a learning agent navigating a custom FrozenLake environment.
The agent learns a value function by averaging episode outcomes (Monte Carlo style),
and follows an epsilon-greedy strategy to improve its behavior over time.

Note: This is not classical Value Iteration using Bellman updates,
but a form of policy evaluation via sampling and incremental reward shaping.

Author: Finn Stäcker
"""

import gym
import numpy as np
import random

# Parameters
num_episodes = 15000
exploration_rate = 1.0
exploration_decay_rate = 0.000099
columns, rows = 8, 8

goalcounter = 0

# Define custom FrozenLake map
custom_map8 = [
    "SFFFFFFF",
    "FHFHHHFH",
    "FHFFFHFF",
    "FHHHFFFH",
    "FFFFFHFF",
    "HFHFHHHF",
    "FFFFFHFF",
    "FHFHFFFG",
]

# Create FrozenLake environment
env = gym.make('FrozenLake-v1', desc=custom_map8, is_slippery=False)

# Select best action based on current value estimates
def best_move(state, value_maze):
    best_value = float('-inf')
    best_action = 0
    for move in range(4):
        predict_state = env.P[state][move][0][1]
        if predict_state != state and value_maze[predict_state] > best_value:
            best_action = move
            best_value = value_maze[predict_state]
    return best_action

# Initialize value table and visitation counters
value_maze = np.zeros(columns * rows)
count_visits = np.zeros(columns * rows)

# Training loop
for episode in range(num_episodes + 1):
    next_state = env.reset()[0]
    terminated = False
    agent_path = [next_state]
    value_maze_average = np.zeros(columns * rows)
    explore_counter = 0
    exploit_counter = 0

    while not terminated:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) > exploration_rate:
            exploit_counter += 1
            action = best_move(next_state, value_maze_average)
        else:
            explore_counter += 1
            action = env.action_space.sample()

        count_visits[next_state] += 1
        next_state, reward, terminated, _, _ = env.step(action)
        agent_path.append(next_state)

        if reward == 0:
            for state in agent_path:
                value_maze[state] -= 1
        if terminated:
            count_visits[next_state] += 1
            if reward > 0:
                for state in agent_path:
                    value_maze[state] += 100
                goalcounter += 1
            else:
                for state in agent_path:
                    value_maze[state] -= 100
                value_maze[next_state] -= 100

        # Update average state values
        for i in range(len(value_maze)):
            if count_visits[i] != 0:
                value_maze_average[i] = value_maze[i] / count_visits[i]

    print(f"Episode: {episode}, Explore: {explore_counter}, Exploit: {exploit_counter}, Exploration Rate: {exploration_rate:.4f}")
    print("Agent Path:", agent_path)
    print("Estimated Value Maze:\n", value_maze_average.reshape((columns, rows)))

    exploration_rate = max(0, exploration_rate - exploration_decay_rate)

print("\nTotal State Visit Counts:\n", count_visits.reshape(rows, columns))
print("\nTotal Goals Reached:", goalcounter)

# Test learned policy visually
env = gym.make('FrozenLake-v1', desc=custom_map8, is_slippery=False, render_mode="human")

next_state = env.reset()[0]
done = False
score = 0
path = [next_state]

while not done:
    env.render()
    action = best_move(next_state, value_maze_average)
    next_state, reward, done, _, _ = env.step(action)
    path.append(next_state)
    score += reward

print(f"\nFinal Episode Score: {score}, Path: {path}")
env.close()