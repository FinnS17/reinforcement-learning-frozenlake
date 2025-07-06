# Reinforcement Learning: FrozenLake & Custom Maze

This project explores different Reinforcement Learning (RL) algorithms applied to two types of environments:

1. A **custom-built maze** environment (no external libraries)
2. The classic `FrozenLake-v1` environment from OpenAI Gym

---

## 📂 Project Structure

### 🔷 Custom Maze Environment

- `random_agent.py`  
  A random agent that explores a self-defined 8x8 maze with obstacles, start and goal states.
  It collects visit statistics across episodes and estimates state values based on path frequency.

> This environment is implemented manually using NumPy and not based on Gym.

### 🧊 OpenAI Gym: FrozenLake-v1

Implemented using the official Gym environment:

- `value_iteration.py` – Solves the MDP with dynamic programming
- `q_learning.py` – Model-free Q-learning
- `deep_q_learning.py` – Uses a neural network to approximate Q-values

---

## 🎯 Goal

Compare different RL strategies on different environments:

- Solve `FrozenLake-v1` with model-based and model-free methods
- Explore and visualize high-value paths in a custom maze using a random agent

---

## 🛠️ Requirements

- Python 3.9+
- `numpy`
- `matplotlib`
- `gym`

Install all dependencies:

```bash
pip install -r requirements.txt