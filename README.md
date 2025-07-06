# Reinforcement Learning Maze & FrozenLake Experiments

This repository contains multiple implementations of basic Reinforcement Learning (RL) agents applied to custom mazes and the OpenAI Gym `FrozenLake-v1` environment. The focus lies on learning optimal paths through value estimation or Q-learning techniques, ranging from random agents to Monte Carlo methods and Q-learning.

## Repository Structure

```
.
├── custom_maze/
│   └── random_agent.py
├── frozenlake/
│   ├── Q-Learning.py
│   └── monte_carlo_value_estimation.py
├── .gitignore
└── README.md
```

## Environments

- **Custom Maze**: An 8x8 grid-based maze with fixed obstacles and a start/goal.
- **FrozenLake-v1**: A standard 8x8 FrozenLake environment (with and without slipperiness), optionally modified via a custom map.

---

## Scripts Overview

### `random_agent.py` (Custom Maze)

Simulates a random agent navigating an 8x8 maze. It:

- Collects visit frequencies for each cell over thousands of episodes
- Rewards later steps more heavily (heuristic)
- Averages these to estimate a value for each state
- Derives the best deterministic path based on those values

**Output**: Visualized maze with the highest value path.

---

### `Q-Learning.py` (FrozenLake)

Implements tabular **Q-Learning** on an 8x8 deterministic FrozenLake map. Key elements:

- Epsilon-greedy exploration strategy
- Value updates via Bellman equation
- Tracks episode rewards and visualizes learning progress

**Output**: Learning curve plot + final demo in `render_mode="human"`.

---

### `monte_carlo_value_estimation.py` (FrozenLake)

Implements **Monte Carlo-style value estimation**:

- Uses a custom 8x8 FrozenLake map
- Updates value estimates based on episode outcomes (success = +100, failure = -100)
- Applies epsilon-greedy policy to balance exploration/exploitation
- Tracks value estimates and visualizes the final learned policy path

**Output**: Episode summaries, final learned value map, and visual policy demo.

---

## Setup & Dependencies

This project uses the following Python packages:

- `numpy`
- `matplotlib`
- `gym`
- `tqdm`

Install them via:

```bash
pip install -r requirements.txt
```

**Note**: `FrozenLake-v1` comes with `gym`.

---

## Run Instructions

```bash
# Custom Maze Random Agent
python custom_maze/random_agent.py

# FrozenLake Q-Learning
python frozenlake/Q-Learning.py

# FrozenLake Monte Carlo Value Estimation
python frozenlake/monte_carlo_value_estimation.py
```

---

## Author

**Finn Stäcker**  
Passionate about Reinforcement Learning, causal inference, and intelligent systems.

---

## License

MIT License
