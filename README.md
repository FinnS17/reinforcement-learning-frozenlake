# Reinforcement Learning: FrozenLake & Custom Maze

This project explores various Reinforcement Learning (RL) algorithms applied to two types of environments:

1. A **custom-built maze** environment (no external frameworks)  
2. The classic `FrozenLake-v1` environment from OpenAI Gym

---

## 📂 Project Structure

### 🔷 Custom Maze Environment

- **`random_agent.py`**  
  A random agent navigates an 8×8 maze with obstacles, start and goal states.  
  Collects visit statistics across episodes, estimates state values, and visualizes the highest-value path.

> This environment is implemented manually using NumPy, without Gym.

### 🧊 OpenAI Gym: FrozenLake-v1

- **`value_iteration.py`**  
  Solves the FrozenLake MDP using dynamic programming (Value Iteration).  
- **`q_learning.py`**  
  Tabular Q-learning with an ε-greedy policy on the deterministic 8×8 FrozenLake.  
- **`deep_q_learning.py`**  
  Deep Q-Network (DQN) implementation that uses a neural network to approximate Q-values.  
- **`monte_carlo.py`**  
  Monte Carlo value estimation with ε-greedy policy improvement on a custom FrozenLake map.

---

## 🎯 Objectives

- **Compare model-based** (Value Iteration) **vs. model-free** (Q-Learning, DQN, Monte Carlo) methods on FrozenLake-v1.  
- **Investigate and visualize** high-value paths in a custom maze environment using a random agent.

---

## 🛠️ Requirements

- Python ≥ 3.9  
- `numpy`  
- `matplotlib`  
- `gymnasium` (or `gym`)  
- `tqdm`  

Install dependencies:

```bash
pip install numpy matplotlib gymnasium tqdm
