import gym
import numpy as np
import random

num_episodes = 10000
exploration_rate = 1
exploration_decay_rate = 0.000099

goalcounter = 0

columns = 8
rows = 8
custom_map8 = ["SFFFFFFF",
              "FHFHHHFH",
              "FHFFFHFF",
              "FHHHFFFH",
              "FFFFFHFF",
              "HFHFHHHF",
              "FFFFFHFF",
              "FHFHFFFG",
              ]

env = gym.make('FrozenLake-v1', desc=custom_map8, is_slippery=False)
def best_move(s, value_maze):
    best_value = float('-inf')
    for move in range(4):
        predict_state = env.P[s][move]
        if predict_state[0][1] != s:
            for state in range(len(value_maze)):
                if state == predict_state[0][1]:
                    new_position_value = value_maze[state]
                    if new_position_value > best_value:
                        best_move = move
                        best_value = new_position_value
    return best_move

#-----------Training-----------------
value_maze = np.zeros(columns*rows)
count_visits =np.zeros(columns*rows)

for episode in range(num_episodes+1):
    next_state = env.reset()[0]
    terminated = False
    agent_path = [next_state]
    score = 0
    value_maze_average = np.zeros(columns*rows)

    explore_counter = 0
    exploit_counter = 0

    while not terminated:
        #Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            exploit_counter += 1
            action = best_move(next_state, value_maze_average)
        else:
            explore_counter += 1
            action = env.action_space.sample()  # exploration

        count_visits[next_state] += 1
        next_state, reward, terminated, _, _ = env.step(action)
        agent_path.append(next_state)

        score = score + reward

        if reward == 0: #step hat das ziel nicht erreicht
            for statea in agent_path:
                value_maze[statea] -= 1
        if terminated:
            count_visits[next_state] += 1
            if reward > 0:#ziel erreicht
                for statev in agent_path:
                    value_maze[statev] += 100
                    goalcounter +=1
            else:#ins loch gefallen
                for statea in agent_path:
                    value_maze[statea] -= 100
                value_maze[next_state] -= 100

        for i in range(len(value_maze)):
            value_maze_average[i]= np.divide(value_maze[i],count_visits[i])

    #print("epsisode: {}, explore_counter: {}, exploit_counter: {}, exploration_rate: {}".format(episode,explore_counter, exploit_counter, exploration_rate))

    exploration_rate = exploration_rate - exploration_decay_rate
    #print("Agentpath: ")
    #print(agent_path)
    #print()
    #print("aktuelle Value-maze:")
    #print(value_maze_average.reshape((columns, rows)))

print()
print("Anzahl Visits: ")
print(count_visits.reshape(rows,columns))

print()
print("goalcounter:")
print(goalcounter)

#-------Anwendung-----------------------------------------
env = gym.make('FrozenLake-v1', desc=custom_map8, is_slippery=False, render_mode = "human")

for episode in range(1):
    next_state = env.reset()[0]
    done = False
    score = 0
    path = [next_state]

    while not done:
        env.render()
        action = best_move(next_state, value_maze_average)
        next_state, reward, done, truncated, info = env.step(action)

        score += reward
        path.append(next_state)

    print('Episode: {} Score: {} Path: {}'.format(episode, score, path))

env.close()