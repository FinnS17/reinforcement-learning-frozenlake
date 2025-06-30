"""
Random Maze Agent Simulator

This script simulates a random agent navigating through a maze from start to goal.
Over multiple episodes, it collects visit statistics and estimates a value for each cell
based on frequency and relative position, ultimately determining a high-value path.

Author: Finn Stäcker
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List, Tuple

Coordinate = Tuple[int, int]


def generate_maze(rows: int, cols: int) -> tuple[np.ndarray, List[Coordinate], dict]:
    """
    Generate a maze with fixed obstacles, start (1), and goal (2).
    
    Args:
        rows (int): Number of maze rows.
        cols (int): Number of maze columns.

    Returns:
        maze (np.ndarray): Maze grid.
        obstacles (List[Tuple[int, int]]): List of obstacle coordinates.
        visit_counter (dict): Dictionary to count visits per cell.
    """
    maze = np.zeros((rows, cols))
    maze[0, 0] = 1  # Start
    maze[rows - 1, cols - 1] = 2  # Goal

    obstacles = [(1, 2), (1, 1), (2, 2), (3, 6), (4, 2), (5, 4),
                 (3, 4), (1, 5), (7, 1), (7, 5), (6, 3), (3, 0),
                 (5, 1), (5, 7)]

    for obs in obstacles:
        maze[obs] = -1

    visit_counter = {(i, j): 0 for i in range(rows) for j in range(cols)}
    return maze, obstacles, visit_counter


def plot_maze(maze: np.ndarray, agent_path: List[Coordinate]) -> None:
    """
    Visualizes the maze with start, goal, obstacles, and the agent's path.
    
    Args:
        maze (np.ndarray): The maze structure.
        agent_path (List[Coordinate]): List of coordinates representing the path.
    """
    cmap = plt.cm.binary
    cmap.set_bad(color='white')
    plt.imshow(maze, cmap=cmap, interpolation='nearest')
    rows, cols = maze.shape

    for i in range(rows - 1):
        plt.axhline(i + 0.5, color='white', linewidth=1)
    for j in range(cols - 1):
        plt.axvline(j + 0.5, color='white', linewidth=1)

    start = np.argwhere(maze == 1)
    end = np.argwhere(maze == 2)
    plt.scatter(start[0, 1], start[0, 0], color='green', s=100, label='Start')
    plt.scatter(end[0, 1], end[0, 0], color='red', s=100, label='Goal')

    if agent_path:
        agent_path = np.array(agent_path)
        for i in range(1, len(agent_path)):
            dx = agent_path[i, 1] - agent_path[i - 1, 1]
            dy = agent_path[i, 0] - agent_path[i - 1, 0]
            plt.quiver(agent_path[i - 1, 1], agent_path[i - 1, 0],
                       dx, dy, color='blue', angles='xy',
                       scale_units='xy', scale=1)

    plt.xticks([]), plt.yticks([])
    plt.gcf().set_facecolor('white')
    plt.legend()
    plt.grid(False)
    plt.title("Random Agent: Estimated Best Path")
    plt.show()


def get_valid_moves(current: Coordinate, maze: np.ndarray) -> List[Coordinate]:
    """Returns a list of valid adjacent cells (not obstacles or out-of-bounds)."""
    r, c = current
    candidates = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    valid = []
    for dr, dc in candidates:
        nr, nc = r + dr, c + dc
        if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]:
            if maze[nr, nc] != -1:
                valid.append((dr, dc))
    return valid


def simulate_random_path(maze: np.ndarray) -> List[Coordinate]:
    """
    Simulates a random walk from start to goal.

    Args:
        maze (np.ndarray): The maze.

    Returns:
        List of coordinates visited by the agent.
    """
    current = (0, 0)
    path = [current]

    while maze[current] != 2:
        moves = get_valid_moves(current, maze)
        if not moves:
            break  # dead-end
        move = random.choice(moves)
        current = (current[0] + move[0], current[1] + move[1])
        path.append(current)

    return path


def estimate_state_values(path: List[Coordinate], visit_counter: dict, obstacles: List[Coordinate]) -> dict:
    """
    Estimate value of each cell based on visit position (later is better).

    Args:
        path (List[Coordinate]): Agent path for one episode.
        visit_counter (dict): Tracks visit counts.
        obstacles (List[Coordinate]): Blocked cells.

    Returns:
        dict: Estimated value per cell.
    """
    values = {}

    for idx, coord in enumerate(path):
        if coord in obstacles:
            continue
        visit_counter[coord] += 1
        step_reward = idx - len(path) + 100  # Later = higher reward
        values[coord] = values.get(coord, 0) + step_reward

    return values


def aggregate_value_tables(total: dict, episode: dict) -> None:
    """Accumulate episode values into total."""
    for key, val in episode.items():
        total[key] = total.get(key, 0) + val


def average_values(total_values: dict, visit_counter: dict) -> dict:
    """Calculate average value per cell (avoid division by zero)."""
    averaged = {}
    for coord, value in total_values.items():
        visits = visit_counter.get(coord, 0)
        if visits > 0:
            averaged[coord] = value / visits
    return averaged


def get_best_path(maze: np.ndarray, value_table: dict) -> List[Coordinate]:
    """
    Generate a path from start to goal by always picking the neighbor with highest value.

    Args:
        maze (np.ndarray): Maze grid.
        value_table (dict): Learned values per cell.

    Returns:
        List[Coordinate]: Deterministic path from start to goal.
    """
    current = (0, 0)
    path = [current]

    while maze[current] != 2:
        moves = get_valid_moves(current, maze)
        best_score = float('-inf')
        best_move = None

        for dr, dc in moves:
            neighbor = (current[0] + dr, current[1] + dc)
            score = value_table.get(neighbor, float('-inf'))
            if score > best_score:
                best_score = score
                best_move = neighbor

        if best_move is None:
            break
        current = best_move
        path.append(current)

    return path


def run_simulation(rows: int, cols: int, episodes: int = 10000) -> None:
    """
    Run the random agent simulation and visualize the learned best path.

    Args:
        rows (int): Maze rows.
        cols (int): Maze cols.
        episodes (int): Number of random runs.
    """
    maze, obstacles, visit_counter = generate_maze(rows, cols)
    total_values = {}

    for _ in range(episodes):
        path = simulate_random_path(maze)
        episode_values = estimate_state_values(path, visit_counter, obstacles)
        aggregate_value_tables(total_values, episode_values)

    averaged_values = average_values(total_values, visit_counter)
    best = get_best_path(maze, averaged_values)
    plot_maze(maze, best)

    print("\nBest path found:")
    print(best)
    print(f"\nSteps: {len(best)}")
    print("\nSample state values (top 10):")
    print(sorted(averaged_values.items(), key=lambda x: -x[1])[:10])


if __name__ == "__main__":
    run_simulation(rows=8, cols=8, episodes=10000)