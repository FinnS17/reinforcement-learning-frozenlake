import matplotlib.pyplot as plt
import numpy as np
import random

anzahl_cols = 8
anzahl_rows = 8
obstacles_list = []
count_visits = []

def generate_maze(rows, cols):
    anzahl_cols = cols
    anzahl_rows = rows

    #count_visits befüllen mit Zuständen/Positionen je nach Größe der Maze
    for i in range(rows):
      for j in range(cols):
        coordinates = (i, j)
        count_visits.append([coordinates, 0])

    maze = np.zeros((rows, cols))
    maze[0, 0] = 1  # Start
    maze[rows-1, cols-1] = 2  # End

    # Set fixed obstacles
    obstacles = [(1, 2),(1,1), (2, 2), (3,6), (4, 2),  (5, 4), (3, 4),(1,5), (7,1),(7,5), (6,3),(3,0),(5,1),(5,7) ]#
      # Define obstacle positions
    obstacles_list = obstacles


    for obstacle in obstacles:
        maze[obstacle] = -1  # Obstacle

    return maze

def plot_maze(maze, agent_path):
    cmap = plt.cm.binary
    cmap.set_bad(color='white')  # Set color for obstacles

    plt.imshow(maze, cmap=cmap, interpolation='nearest')

    rows, cols = maze.shape

    # Draw gridlines between cells
    for i in range(rows - 1):
        plt.axhline(i + 0.5, color='white', linewidth=1)

    for j in range(cols - 1):
        plt.axvline(j + 0.5, color='white', linewidth=1)

    # Find and plot the start position in green
    start_position = np.argwhere(maze == 1)
    plt.scatter(start_position[0, 1], start_position[0, 0], color='green', marker='o', s=100)

    # Find and plot the end position in red
    end_position = np.argwhere(maze == 2)
    plt.scatter(end_position[0, 1], end_position[0, 0], color='red', marker='o', s=100)

    # Plot the agent's path
    if agent_path is not None:
        agent_path = np.array(agent_path)

        # Plot arrows for the direction of the agent's path
        for i in range(1, len(agent_path)):
            dx = agent_path[i, 1] - agent_path[i - 1, 1]
            dy = agent_path[i, 0] - agent_path[i - 1, 0]
            plt.quiver(agent_path[i - 1, 1], agent_path[i - 1, 0], dx, dy, color='blue', angles='xy', scale_units='xy',
                       scale=1)

    plt.grid(False)  # Turn off default grid
    plt.xticks([]), plt.yticks([])
    plt.gcf().set_facecolor('white')
    plt.show()


def random_agent(maze):
    current_position = (0, 0)
    agent_path = [current_position]  # Use a list with tuple elements

    while maze[current_position] != 2:  # Continue until reaching the goal

        # Generate a random move and check what direction is possible
        possible_moves = []
        if current_position[0] + 1 < maze.shape[0] and maze[current_position[0] + 1, current_position[1]] != -1:
            possible_moves.append((1, 0))  # Move down
        if current_position[1] + 1 < maze.shape[1] and maze[current_position[0], current_position[1] + 1] != -1:
            possible_moves.append((0, 1))  # Move right
        if current_position[0] - 1 >= 0 and maze[current_position[0] - 1, current_position[1]] != -1:
            possible_moves.append((-1, 0))  # Move up
        if current_position[1] - 1 >= 0 and maze[current_position[0], current_position[1] - 1] != -1:
            possible_moves.append((0, -1))  # Move left

        move = possible_moves[np.random.choice(len(possible_moves))]
        current_position = (current_position[0] + move[0], current_position[1] + move[1])
        agent_path.append(current_position)  # Append the tuple to the list
    return agent_path

def calculate_reward(agent_path, coordinate):#reward für eine koordinate returnen

    indices = indices_koordinate(coordinate, agent_path)
    reward = 0
    for i in range(len(indices)):
        reward = reward + (indices[i]-len(agent_path) + 100)
        for y in range(len(count_visits)):
            if count_visits[y][0] == coordinate:
                count_visits[y][1] = count_visits[y][1] + 1
    return reward

def indices_koordinate(koordinate, agent_path):
    indices = []
    for index in range(len(agent_path)):
        if agent_path[index] == koordinate:
            indices.append(index)
    return indices

def generate_value_maze(agent_path):
    value_maze = []

    for i in range(anzahl_rows):
        for j in range(anzahl_cols):
            coordinates = (i, j)
            value = calculate_reward(agent_path, coordinates)
            value_maze.append([coordinates, value])
    coordinates_to_remove = obstacles_list

    filtered_value_maze = [] #obstacles entfernen
    for entry in value_maze:
        if entry[0] not in coordinates_to_remove:
            filtered_value_maze.append(entry)

    return filtered_value_maze

def best_move(possible_moves, current_position, total_value_maze):
    best_value = float('-inf')
    best_position = None
    for move in possible_moves:
        new_position = (current_position[0] + move[0], current_position[1] + move[1])
        for koordinate, value in total_value_maze:
            if koordinate == new_position:
                new_position_value = value
                if new_position_value > best_value:
                    best_position = new_position
                    best_value = new_position_value
                    break
                else:
                    break
    return best_position

def best_path(total_value_maze, maze):
  current_position = (0, 0)
  agent_path = [current_position]

  while maze[current_position] != 2:  # Continue until reaching the goal
    possible_moves = []

    if current_position[0] + 1 < maze.shape[0] and maze[current_position[0] + 1, current_position[1]] != -1:
      possible_moves.append((1, 0))  # Move down
    if current_position[1] + 1 < maze.shape[1] and maze[current_position[0], current_position[1] + 1] != -1:
      possible_moves.append((0, 1))  # Move right
    if current_position[0] - 1 >= 0 and maze[current_position[0] - 1, current_position[1]] != -1:
      possible_moves.append((-1, 0))  # Move up
    if current_position[1] - 1 >= 0 and maze[current_position[0], current_position[1] - 1] != -1:
      possible_moves.append((0, -1))  # Move left
    best_position = best_move(possible_moves, current_position, total_value_maze)
    agent_path.append(best_position)
    current_position = best_position
  return agent_path

def run_random(rows, cols, episodes):
    maze = generate_maze(rows, cols)

    #erster Durchgang
    agent_path = random_agent(maze)
    total_value_maze = generate_value_maze(agent_path)

    for _ in range(episodes):
        agent_path = random_agent(maze)
        value_maze = generate_value_maze(agent_path)
        for i in range(len(value_maze)):
            total_value_maze[i][1] = total_value_maze[i][1] + value_maze[i][1]
    for y in range(len(total_value_maze)):
        if count_visits[y][1] > 0:
            total_value_maze[y][1] = total_value_maze[y][1]/(count_visits[y][1])

    best_agent_path = best_path(total_value_maze, maze)
    plot_maze(maze, best_agent_path)
    print("")
    print(count_visits)
    print("")
    print("best_agent_path:")
    print(best_agent_path)
    print("")
    print("Anzahl_Schritte: ")
    print(len(best_agent_path))
    print("")
    print("totalvaluemaze:")
    print(total_value_maze)

#------------main random------------------------------------------

run_random(8, 8, 10000)