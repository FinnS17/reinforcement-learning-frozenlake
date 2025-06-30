# Reinforcement Learning on FrozenLake

This project demonstrates different Reinforcement Learning algorithms on the classic FrozenLake-v1 environment from OpenAI Gym.

## 📂 Project Structure

All algorithms are implemented in separate files:

- **Random Agent** – takes random actions
- **Value Iteration** – uses dynamic programming to solve the MDP
- **Q-Learning** – model-free off-policy learning
- **Deep Q-Learning** – uses a neural network to approximate Q-values

## 💡 Goal

Solve the FrozenLake-v1 environment with different strategies and compare their performance.

## 📦 Requirements

- Python 3.9+
- `numpy`
- `matplotlib`
- `gym`

Install dependencies:

```bash
pip install -r requirements.txt