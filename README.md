# Wall-Following Robot Navigation — MDP + Value Iteration

Streamlit app for the RDMU exercise "MDPs – Wall-following Robot navigation".

## Files
- `app.py` — the Streamlit app (UI only)
- `mdp_core.py` — the maths: data prep, states, transition probabilities, rewards, value iteration, Monte Carlo evaluation
- `data/sensor_readings_24.csv` — Kaggle / UCI SCITOS-G5 dataset (24 sensors + class)
- `requirements.txt`, `.streamlit/config.toml`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Without the UI: `python mdp_core.py` writes `optimal_value_function.csv`.

## How the MDP is built
| Piece | Definition |
|---|---|
| States | SD_front, SD_left, SD_right, SD_back each cut into quantile bins (default 3: Near/Mid/Far) |
| Actions | Move-Forward, Slight-Right-Turn, Sharp-Right-Turn, Slight-Left-Turn |
| Transitions | P(s'\|s,a) counted from consecutive readings (row t → row t+1) |
| Rewards | +1 left wall in target band, −1 wall lost, −5 too close front/left, small cost per turn |
| γ, tolerance | Sidebar (default 0.9 and 1e-6) |
| Output | `optimal_value_function.csv`: V*(s), optimal action, Q-values, visits |

The four simplified distances are the minimum of these sensors (this reproduces the
official `sensor_readings_4` file exactly): front US11–15, left US18–20, right US5–9, back US23–24.

Value iteration keeps the Day 5 notebook update
`Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)` and adds a stop when
`max|V_new − V| < tolerance`. Actions never observed in a state are skipped.
