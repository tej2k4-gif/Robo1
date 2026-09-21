"""
Wall-Following Robot Navigation - MDP + Value Iteration
Single-file Streamlit app for the RDMU exercise (no other .py files needed).
Includes an interactive grid simulator of the robot (with start and end points),
a step-by-step value iteration animation, and the Day 5 notebook programs:
MDP, ADP, Monte Carlo policy search and Hooke-Jeeves.

Run:  streamlit run app.py
Data: put sensor_readings_24.csv next to app.py (or in a data/ folder),
      or upload it from the sidebar.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import json


# ===========================================================================
# Compatibility helpers - work on old and new Streamlit versions
# ===========================================================================

def _st_version():
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


NEW_ST = _st_version() >= (1, 50)
cache_data = getattr(st, "cache_data", None) or st.cache


def show_chart(fig):
    if NEW_ST:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_df(data, height=None, hide_index=True):
    kwargs = {}
    if height is not None:
        kwargs["height"] = height
    if NEW_ST:
        kwargs["width"] = "stretch"
    else:
        kwargs["use_container_width"] = True
    if hide_index and _st_version() >= (1, 23):
        kwargs["hide_index"] = True
    try:
        st.dataframe(data, **kwargs)
    except TypeError:
        st.dataframe(data)


def find_default_data():
    here = Path(__file__).parent
    for p in [here / "data" / "sensor_readings_24.csv", here / "sensor_readings_24.csv",
              Path.cwd() / "data" / "sensor_readings_24.csv", Path.cwd() / "sensor_readings_24.csv"]:
        if p.exists():
            return p
    return None


# ===========================================================================
# MDP maths (data prep, states, transitions, rewards, value iteration)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

ACTIONS = ["Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn"]
SENSOR_COLS = [f"US{i}" for i in range(1, 25)]

# Sensor groups that reproduce the official sensor_readings_4 file exactly
# (checked against the summary statistics in the dataset README).
SD_GROUPS = {
    "SD_front": [11, 12, 13, 14, 15],
    "SD_left": [18, 19, 20],
    "SD_right": [5, 6, 7, 8, 9],
    "SD_back": [23, 24],
}
SD_COLS = list(SD_GROUPS.keys())

# Reference angle of each ultrasound sensor (from the README), in degrees
SENSOR_ANGLES = {1: 180}
SENSOR_ANGLES.update({i: -180 + 15 * (i - 1) for i in range(2, 14)})   # US2..US13
SENSOR_ANGLES.update({i: 15 * (i - 13) for i in range(14, 25)})        # US14..US24


def load_data(path_or_buffer):
    """Read the raw 24-sensor CSV and add the 4 simplified distances.
    Works whether or not the file has a header row."""
    df = pd.read_csv(path_or_buffer, header=None)
    df = df.iloc[:, :25]
    try:
        float(df.iloc[0, 0])
    except (TypeError, ValueError):          # first row is a header -> drop it
        df = df.iloc[1:].reset_index(drop=True)
    df.columns = SENSOR_COLS + ["Class"]
    df[SENSOR_COLS] = df[SENSOR_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["Class"] = df["Class"].astype(str).str.strip()
    df = df[df["Class"].isin(ACTIONS)].reset_index(drop=True)
    for name, sensors in SD_GROUPS.items():
        df[name] = df[[f"US{i}" for i in sensors]].min(axis=1)
    return df


# ---------------------------------------------------------------------------
# 2. States (discretisation)
# ---------------------------------------------------------------------------

BIN_NAMES = {
    2: ["Near", "Far"],
    3: ["Near", "Mid", "Far"],
    4: ["Very near", "Near", "Mid", "Far"],
    5: ["Very near", "Near", "Mid", "Far", "Very far"],
}


def make_bin_edges(df, features, n_bins):
    """Quantile-based edges so every bin holds roughly the same number of readings."""
    edges = {}
    for f in features:
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges[f] = np.unique(np.round(df[f].quantile(qs).values, 3))
    return edges


def to_state(values, features, edges, n_bins):
    """Map one reading (dict or Series of distances) to a state label like 'N|M|F|F'."""
    names = BIN_NAMES[n_bins]
    parts = []
    for f in features:
        idx = int(np.searchsorted(edges[f], values[f], side="right"))
        parts.append(names[min(idx, len(names) - 1)])
    return " | ".join(parts)


def add_states(df, features, edges, n_bins):
    df = df.copy()
    df["State"] = [to_state(row, features, edges, n_bins) for _, row in df[features].iterrows()]
    return df


# ---------------------------------------------------------------------------
# 3 + 4. Transitions and rewards
# ---------------------------------------------------------------------------

def reading_reward(row, p):
    """
    Reward for *arriving* at a reading. p = dict of reward settings.
      + follow_reward   left wall inside the target band (good wall-following)
      - lost_penalty    left wall too far away (robot lost the wall)
      - crash_penalty   something too close in front or on the left
    """
    r = 0.0
    if p["left_low"] <= row["SD_left"] <= p["left_high"]:
        r += p["follow_reward"]
    elif row["SD_left"] > p["left_high"]:
        r -= p["lost_penalty"]
    if row["SD_front"] < p["front_danger"]:
        r -= p["crash_penalty"]
    if row["SD_left"] < p["side_danger"]:
        r -= p["crash_penalty"]
    return r


def build_mdp(df, reward_params, action_costs):
    """
    Estimate T[s, a, s'] and R[s, a] from consecutive rows.
    Returns a dict with states, T, R, counts and a validity mask.
    """
    states = sorted(df["State"].unique())
    s_idx = {s: i for i, s in enumerate(states)}
    a_idx = {a: i for i, a in enumerate(ACTIONS)}
    S, A = len(states), len(ACTIONS)

    counts = np.zeros((S, A, S))
    reward_sum = np.zeros((S, A))

    st = df["State"].map(s_idx).values
    ac = df["Class"].map(a_idx).values
    rewards_next = df.apply(lambda r: reading_reward(r, reward_params), axis=1).values

    for t in range(len(df) - 1):
        s, a, s2 = st[t], ac[t], st[t + 1]
        counts[s, a, s2] += 1
        reward_sum[s, a] += rewards_next[t + 1]

    n_sa = counts.sum(axis=2)                     # how often action a was taken in s
    valid = n_sa > 0                              # only trust what the data has seen

    T = np.zeros_like(counts)
    R = np.zeros((S, A))
    T[valid] = counts[valid] / n_sa[valid][:, None]
    R[valid] = reward_sum[valid] / n_sa[valid]
    R -= np.array([action_costs[a] for a in ACTIONS])[None, :]

    # A state with no observed action (e.g. only seen on the last row) becomes absorbing
    dead = ~valid.any(axis=1)
    for s in np.where(dead)[0]:
        T[s, 0, s] = 1.0
        R[s, 0] = 0.0
        valid[s, 0] = True

    return {"states": states, "T": T, "R": R, "counts": counts,
            "n_sa": n_sa, "valid": valid, "rewards_next": rewards_next}


# ---------------------------------------------------------------------------
# 5. Value iteration  (Day 5 notebook update + convergence tolerance)
# ---------------------------------------------------------------------------

def value_iteration(T, R, gamma=0.9, tolerance=1e-6, max_iterations=10_000, valid=None):
    """
    Same Bellman update as the notebook:
        Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
        V_new[s] = max_a Q_sa[a]
    but it stops as soon as max|V_new - V| < tolerance instead of
    running a fixed 1000 iterations. Actions never seen in a state are skipped.
    """
    S, A = R.shape
    if valid is None:
        valid = np.ones((S, A), dtype=bool)
    V = np.zeros(S)
    history = []
    V_hist = [V.copy()]                            # V_0, V_1, ... for the step-by-step animation
    for i in range(max_iterations):
        V_new = np.zeros(S)
        for s in range(S):
            Q_sa = np.full(A, -np.inf)
            for a in range(A):
                if valid[s, a]:
                    Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
            V_new[s] = np.max(Q_sa)
        delta = np.max(np.abs(V_new - V))
        history.append(delta)
        V = V_new
        V_hist.append(V.copy())
        if delta < tolerance:
            break

    Q = np.where(valid, R + gamma * np.einsum("ijk,k->ij", T, V), -np.inf)
    policy = Q.argmax(axis=1)
    return {"V": V, "Q": Q, "policy": policy, "history": history, "V_hist": V_hist,
            "iterations": len(history), "converged": history[-1] < tolerance}


# ---------------------------------------------------------------------------
# 6. Results table / CSV
# ---------------------------------------------------------------------------

def results_table(mdp, vi, features):
    rows = []
    for i, s in enumerate(mdp["states"]):
        row = {"State": s}
        for f, part in zip(features, s.split(" | ")):
            row[f] = part
        row["Optimal_Value"] = round(float(vi["V"][i]), 6)
        row["Optimal_Action"] = ACTIONS[vi["policy"][i]]
        for a_i, a in enumerate(ACTIONS):
            q = vi["Q"][i, a_i]
            row[f"Q({a})"] = round(float(q), 6) if np.isfinite(q) else np.nan
        row["Data_Most_Common_Action"] = ACTIONS[int(mdp["n_sa"][i].argmax())]
        row["Visits"] = int(mdp["n_sa"][i].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Optimal_Value", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Extra: Monte Carlo policy evaluation (adapted from the notebook's
# MonteCarloPolicySearch class, using the learned MDP as the environment)
# ---------------------------------------------------------------------------

class LearnedRobotEnv:
    """Simulator that samples next states from the estimated T and rewards from R."""

    def __init__(self, mdp, start_dist, seed=0):
        self.T, self.R, self.valid = mdp["T"], mdp["R"], mdp["valid"]
        self.start_dist = start_dist
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self):
        self.state = self.rng.choice(len(self.start_dist), p=self.start_dist)
        return self.state

    def step(self, action):
        s = self.state
        if not self.valid[s, action]:                       # unseen action -> penalty, stay put
            return s, -5.0, False
        s2 = self.rng.choice(self.T.shape[2], p=self.T[s, action])
        self.state = s2
        return s2, self.R[s, action], False


class MonteCarloPolicySearch:
    def __init__(self, env, policy, gamma=0.9, horizon=50):
        self.env, self.policy, self.gamma, self.horizon = env, policy, gamma, horizon

    def generate_episode(self):
        episode, state = [], self.env.reset()
        for _ in range(self.horizon):
            action = self.policy(state)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def evaluate_policy(self, num_episodes=500):
        returns = []
        for _ in range(num_episodes):
            G = 0.0
            for _, _, reward in reversed(self.generate_episode()):
                G = reward + self.gamma * G
            returns.append(G)
        return np.array(returns)


def compare_policies(mdp, vi, gamma, episodes=500, horizon=50, seed=42):
    visits = mdp["n_sa"].sum(axis=1)
    start = visits / visits.sum()
    rng = np.random.default_rng(seed)
    valid = mdp["valid"]
    behaviour = mdp["n_sa"].argmax(axis=1)

    policies = {
        "Optimal (value iteration)": lambda s: int(vi["policy"][s]),
        "Robot's logged behaviour": lambda s: int(behaviour[s]),
        "Random (seen actions)": lambda s: int(rng.choice(np.where(valid[s])[0])),
    }
    out = {}
    for name, pol in policies.items():
        env = LearnedRobotEnv(mdp, start, seed=seed)
        out[name] = MonteCarloPolicySearch(env, pol, gamma, horizon).evaluate_policy(episodes)
    return out


# ---------------------------------------------------------------------------
# Defaults + command-line run
# ---------------------------------------------------------------------------

DEFAULT_REWARDS = {
    "left_low": 0.50, "left_high": 0.90,     # metres: good distance to the wall on the left
    "front_danger": 0.60, "side_danger": 0.40,
    "follow_reward": 1.0, "lost_penalty": 1.0, "crash_penalty": 5.0,
}
DEFAULT_COSTS = {"Move-Forward": 0.0, "Slight-Right-Turn": 0.1,
                 "Sharp-Right-Turn": 0.2, "Slight-Left-Turn": 0.1}


def run_pipeline(df, features=SD_COLS, n_bins=3, gamma=0.9, tolerance=1e-6,
                 reward_params=DEFAULT_REWARDS, action_costs=DEFAULT_COSTS):
    edges = make_bin_edges(df, features, n_bins)
    dfs = add_states(df, features, edges, n_bins)
    mdp = build_mdp(dfs, reward_params, action_costs)
    vi = value_iteration(mdp["T"], mdp["R"], gamma, tolerance, valid=mdp["valid"])
    table = results_table(mdp, vi, features)
    return dfs, edges, mdp, vi, table






# ===========================================================================
# HTML / JavaScript for the interactive components
# ===========================================================================

SIM_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  :root { --ink:#1C2330; --muted:#5B6573; --line:#E3E6EA; --panel:#FFFFFF; --bg:#FBFBF9; --primary:#2F5D8A; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Source Sans Pro","Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;
         color:var(--ink); background:var(--bg); font-size:14px; }
  #app { display:flex; gap:16px; flex-wrap:wrap; padding:4px; }
  #left { flex:1.7 1 520px; min-width:300px; }
  #right { flex:1 1 280px; min-width:260px; display:flex; flex-direction:column; gap:10px; }
  .bar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:8px; }
  button { font:inherit; border:1px solid var(--line); background:var(--panel); color:var(--ink);
           padding:5px 11px; border-radius:6px; cursor:pointer; }
  button:hover { border-color:var(--primary); }
  button:focus-visible, select:focus-visible, input:focus-visible { outline:2px solid var(--primary); outline-offset:1px; }
  button.primary { background:var(--primary); color:#fff; border-color:var(--primary); min-width:82px; }
  button.on { background:#DCE7F2; border-color:var(--primary); }
  select { font:inherit; padding:4px 6px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink); }
  label.inline { display:flex; align-items:center; gap:6px; color:var(--muted); }
  input[type=range] { width:110px; accent-color:var(--primary); }
  canvas#cv { display:block; border-radius:6px; touch-action:none; cursor:crosshair; box-shadow:0 0 0 1px var(--line); }
  .hint { color:var(--muted); font-size:12.5px; margin-top:6px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .card h4 { margin:0 0 6px 0; font-size:13px; color:var(--muted); font-weight:600; }
  #actionPill { display:inline-block; padding:5px 12px; border-radius:999px; color:#fff; font-weight:700; font-size:16px; }
  #stateLbl { margin-top:6px; font-size:13px; }
  #status { font-size:14px; font-weight:600; }
  #status.ok { color:#3E7D5E; } #status.bad { color:#C8553D; }
  .drow { display:grid; grid-template-columns:54px 1fr 96px; align-items:center; gap:6px; margin:3px 0; font-size:12.5px; }
  .dtrack { height:9px; background:#EEF1F4; border-radius:5px; overflow:hidden; }
  .dfill { height:100%; border-radius:5px; }
  .qrow { display:grid; grid-template-columns:118px 1fr 52px; align-items:center; gap:6px; margin:3px 0; font-size:12.5px; }
  .stats { display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; font-size:13px; }
  .stats b { font-size:17px; display:block; }
  #log div { font-size:12px; padding:2px 0; border-bottom:1px dashed var(--line); display:flex; gap:6px; }
  #log div:last-child { border-bottom:none; }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-top:4px; flex:none; }
  details { font-size:13px; color:var(--muted); }
  details .bar { margin-top:6px; }
  .legend { display:flex; flex-wrap:wrap; gap:10px; font-size:12px; color:var(--muted); margin-top:6px; }
  .legend span { display:flex; align-items:center; gap:4px; }
</style></head>
<body>
<div id="app">
  <div id="left">
    <div class="bar">
      <button id="play" class="primary">▶ Play</button>
      <button id="stepBtn">Step</button>
      <button id="reset" title="Send the robot back to the start point">↺ Back to start</button>
      <label class="inline">Speed <input id="speed" type="range" min="1" max="40" value="12"></label>
      <label class="inline">Room <select id="layout"></select></label>
    </div>
    <div class="bar">
      <label class="inline">Policy
        <select id="policy">
          <option value="optimal">Wall-following: optimal (sensor MDP)</option>
          <option value="goal">Go to end point: grid MDP (value iteration)</option>
          <option value="behaviour">Wall-following: robot's logged behaviour</option>
          <option value="random">Random</option>
        </select></label>
    </div>
    <div class="bar">
      <span style="color:var(--muted)">Click on the room to:</span>
      <button id="mWalls" class="on">Draw / erase walls</button>
      <button id="mStart">🟢 Set start</button>
      <button id="mGoal">🏁 Set end point</button>
      <button id="rotL" title="Turn the start heading left 45°">⟲</button>
      <button id="rotR" title="Turn the start heading right 45°">⟳</button>
    </div>
    <canvas id="cv"></canvas>
    <div class="legend" id="legend"></div>
    <div class="hint">Each square is 0.4 m. Green S is the start (the tick shows the starting direction), the flag is the end point.
      The coloured fans are the four 60° sensor arcs; the dot on each fan is the closest reading.</div>
    <details>
      <summary>Uncertainty, motion and display settings</summary>
      <div class="bar">
        <label class="inline">Action slip <input id="slip" type="range" min="0" max="50" value="0"><span id="slipV">0%</span></label>
        <label class="inline">Sensor noise <input id="noise" type="range" min="0" max="30" value="0"><span id="noiseV">0 cm</span></label>
      </div>
      <div class="bar">
        <label class="inline">Step <input id="stepm" type="range" min="4" max="20" value="10"><span id="stepmV">10 cm</span></label>
        <label class="inline">Slight turn <input id="slight" type="range" min="5" max="45" value="20"><span id="slightV">20°</span></label>
        <label class="inline">Sharp turn <input id="sharp" type="range" min="20" max="90" value="45"><span id="sharpV">45°</span></label>
      </div>
      <div class="bar">
        <label class="inline"><input type="checkbox" id="showRays" checked> sensor fans</label>
        <label class="inline"><input type="checkbox" id="showVisits" checked> visited squares</label>
        <label class="inline"><input type="checkbox" id="showTrail" checked> trail</label>
        <label class="inline"><input type="checkbox" id="showGridV" checked> grid MDP values and arrows (end-point policy)</label>
      </div>
    </details>
  </div>

  <div id="right">
    <div class="card"><h4>Status</h4><div id="status">Ready. Press Play.</div></div>
    <div class="card">
      <h4>Next move</h4>
      <span id="actionPill">–</span>
      <div id="stateLbl"></div>
    </div>
    <div class="card"><h4>Sensor readings (simplified distances)</h4><div id="dists"></div></div>
    <div class="card"><h4 id="qTitle">Q-values for this state</h4><div id="qvals"></div></div>
    <div class="card"><h4>Run so far</h4><div class="stats" id="stats"></div></div>
    <div class="card"><h4>Recent moves</h4><div id="log"></div></div>
  </div>
</div>

<script>
(function () {
"use strict";
const P = __PAYLOAD__;
const ACTIONS = P.actions, COLORS = P.colors, ARC_COL = P.arc_colors;
const ARC_OFF = { SD_front: 0, SD_left: 90, SD_right: -90, SD_back: 180 };
const SHORT = { "Move-Forward": "Forward", "Slight-Right-Turn": "Slight right", "Sharp-Right-Turn": "Sharp right", "Slight-Left-Turn": "Slight left" };
const W = 16, H = 12, CELL = 0.4, RAD = 0.15, MAXD = 5.0, MAX_WALL = 1500, MAX_GRID = 600, GOAL_R = 0.9;
const DIRS = [[1, 0, "east"], [1, -1, "north-east"], [0, -1, "north"], [-1, -1, "north-west"],
              [-1, 0, "west"], [-1, 1, "south-west"], [0, 1, "south"], [1, 1, "south-east"]];
const DIR_ARROW = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"];
const GOAL_COLOR = "#2F5D8A", G_GAMMA = 0.99, G_BUMP = 5;
const $ = id => document.getElementById(id);

let grid = [], visits = [], robot = { x: 2.5, y: 9.5, h: 90 }, disp = { x: 2.5, y: 9.5 };
let start = { x: 2, y: 9, h: 90 }, goal = { x: 12, y: 2 };
let trail = [], log = [], stats = null, current = null, status = "ready";
let running = false, bumpFlash = 0, cellPx = 36, dpr = window.devicePixelRatio || 1, mode = "walls";
let gV = null, gPol = null, gReach = null;
const motion = { step: 0.10, slight: 20, sharp: 45 };
let slip = 0, noise = 0;
const cv = $("cv"), ctx = cv.getContext("2d");

// ---------------- rooms ----------------
function blankGrid() {
  grid = []; visits = [];
  for (let y = 0; y < H; y++) {
    const r = [], v = [];
    for (let x = 0; x < W; x++) { r.push(x === 0 || y === 0 || x === W - 1 || y === H - 1 ? 1 : 0); v.push(0); }
    grid.push(r); visits.push(v);
  }
}
function fill(x0, y0, x1, y1) { for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) grid[y][x] = 1; }
const LAYOUTS = {
  "Empty room": { build: () => {}, start: [2, 9, 90], goal: [12, 2] },
  "Room with pillar": { build: () => fill(6, 4, 10, 8), start: [2, 9, 90], goal: [13, 9] },
  "L-shaped room": { build: () => fill(9, 6, 15, 11), start: [2, 9, 90], goal: [12, 2] },
  "Two rooms and a door": { build: () => { for (let y = 1; y < H - 1; y++) if (y < 5 || y > 6) grid[y][8] = 1; }, start: [2, 9, 90], goal: [13, 9] },
  "Cluttered room": { build: () => { fill(4, 3, 6, 5); fill(10, 2, 12, 4); fill(7, 7, 9, 10); fill(12, 7, 14, 9); }, start: [2, 9, 90], goal: [13, 2] },
  "Maze": { build: () => { fill(3, 1, 4, 8); fill(6, 4, 7, 11); fill(9, 1, 10, 8); fill(12, 4, 13, 11); }, start: [1, 10, 90], goal: [14, 1] }
};
Object.keys(LAYOUTS).forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = k; $("layout").appendChild(o); });

function inside(cx, cy) { return cx >= 0 && cy >= 0 && cx < W && cy < H; }
function wallCell(cx, cy) { return !inside(cx, cy) || grid[cy][cx] === 1; }
function isWall(px, py) { return wallCell(Math.floor(px), Math.floor(py)); }
function blocked(x, y) {
  const r = RAD / CELL;
  for (const ax of [-r, 0, r]) for (const ay of [-r, 0, r]) if (isWall(x + ax, y + ay)) return true;
  return false;
}

// ---------------- grid MDP for the end-point policy ----------------
function tryMove(x, y, d) {
  const dx = DIRS[d][0], dy = DIRS[d][1], nx = x + dx, ny = y + dy;
  if (wallCell(nx, ny) || (dx !== 0 && dy !== 0 && (wallCell(x + dx, y) || wallCell(x, y + dy)))) return [x, y, 1, true];
  return [nx, ny, (dx !== 0 && dy !== 0) ? Math.SQRT2 : 1, false];
}
function outcomes(a) { return slip > 0 ? [[a, 1 - slip], [(a + 1) % 8, slip / 2], [(a + 7) % 8, slip / 2]] : [[a, 1]]; }
function qAt(x, y, a) {
  let qv = 0;
  for (const [d, pr] of outcomes(a)) {
    const [nx, ny, cost, bump] = tryMove(x, y, d);
    const nxtV = (nx === goal.x && ny === goal.y) ? 0 : gV[ny][nx];
    qv += pr * (-cost - (bump ? G_BUMP : 0) + G_GAMMA * nxtV);
  }
  return qv;
}
function solveGridMDP() {
  gReach = []; for (let y = 0; y < H; y++) gReach.push(new Array(W).fill(false));
  const q = [[goal.x, goal.y]]; gReach[goal.y][goal.x] = true;
  while (q.length) {
    const [x, y] = q.shift();
    for (let d = 0; d < 8; d++) { const [nx, ny, , b] = tryMove(x, y, d); if (!b && !gReach[ny][nx]) { gReach[ny][nx] = true; q.push([nx, ny]); } }
  }
  gV = []; gPol = [];
  for (let y = 0; y < H; y++) { gV.push(new Array(W).fill(0)); gPol.push(new Array(W).fill(-1)); }
  for (let it = 0; it < 3000; it++) {
    let delta = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (grid[y][x] || !gReach[y][x] || (x === goal.x && y === goal.y)) continue;
      let best = -Infinity, bestA = -1;
      for (let a = 0; a < 8; a++) { const qv = qAt(x, y, a); if (qv > best) { best = qv; bestA = a; } }
      delta = Math.max(delta, Math.abs(best - gV[y][x]));
      gV[y][x] = best; gPol[y][x] = bestA;
    }
    if (delta < 1e-5) break;
  }
}
function gridQ(x, y) { const out = []; for (let a = 0; a < 8; a++) out.push(qAt(x, y, a)); return out; }

// ---------------- sensing and the sensor-MDP policy ----------------
function castRay(x, y, angDeg) {
  const a = angDeg * Math.PI / 180, c = Math.cos(a) / CELL, s = Math.sin(a) / CELL;
  for (let d = 0.02; d < MAXD; d += 0.02) if (isWall(x + d * c, y - d * s)) return d;
  return MAXD;
}
function gauss() { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function sense() {
  const d = {}, rays = {};
  for (const k of P.sd_cols) {
    let m = MAXD, best = null; const pts = [];
    for (let o = -30; o <= 30; o += 5) {
      const ang = robot.h + ARC_OFF[k] + o, dist = castRay(robot.x, robot.y, ang);
      pts.push([ang, dist]); if (dist < m) { m = dist; best = [ang, dist]; }
    }
    let v = m;
    if (noise > 0) v = Math.min(MAXD, Math.max(0.05, v + gauss() * noise));
    d[k] = v; rays[k] = { pts: pts, best: best || [robot.h + ARC_OFF[k], MAXD] };
  }
  return { d: d, rays: rays };
}
function binIndex(f, v) { const e = P.edges[f]; let i = 0; while (i < e.length && v >= e[i]) i++; return Math.min(i, P.bin_names.length - 1); }
function nearestState(idx) {
  let best = null, bd = Infinity;
  for (const s of P.states) { const b = P.bins[s]; let dd = 0; for (let i = 0; i < idx.length; i++) dd += Math.abs(b[i] - idx[i]); if (dd < bd) { bd = dd; best = s; } }
  return best;
}
function readingReward(d) {
  const p = P.rewards; let r = 0;
  if (d.SD_left >= p.left_low && d.SD_left <= p.left_high) r += p.follow_reward;
  else if (d.SD_left > p.left_high) r -= p.lost_penalty;
  if (d.SD_front < p.front_danger) r -= p.crash_penalty;
  if (d.SD_left < p.side_danger) r -= p.crash_penalty;
  return r;
}
function goalMode() { return $("policy").value === "goal"; }
function decide() {
  const s = sense();
  if (goalMode()) {
    const cx = Math.floor(robot.x), cy = Math.floor(robot.y);
    const a = (gPol && inside(cx, cy)) ? gPol[cy][cx] : -1;
    const here = cx === goal.x && cy === goal.y;
    current = { sense: s, goal: true, cx: cx, cy: cy, dir: here ? -1 : a, label: "cell (" + cx + ", " + cy + ")",
                action: here || atGoal() ? "Arrived at the end point" : (a >= 0 ? "Move " + DIRS[a][2] : "No path") };
    return;
  }
  const idx = P.features.map(f => binIndex(f, s.d[f]));
  let label = idx.map(i => P.bin_names[i]).join(" | "), nearest = false;
  if (!Object.prototype.hasOwnProperty.call(P.opt, label)) { label = nearestState(idx); nearest = true; }
  const src = $("policy").value; let a;
  if (src === "behaviour") a = P.beh[label];
  else if (src === "random") { const v = P.valid[label]; a = v[Math.floor(Math.random() * v.length)]; }
  else a = P.opt[label];
  current = { sense: s, goal: false, label: label, nearest: nearest, action: a };
}

// ---------------- one simulation step ----------------
function newStats() { return { steps: 0, bumps: 0, inBand: 0, reward: 0, unknown: 0, slips: 0, dist: 0 }; }
function atGoal() { return Math.hypot(robot.x - (goal.x + 0.5), robot.y - (goal.y + 0.5)) <= GOAL_R; }
function maxSteps() { return goalMode() ? MAX_GRID : MAX_WALL; }
function isDone() { return status === "reached" || status === "timeout" || status === "unreachable"; }
function finish(newStatus) { status = newStatus; setRunning(false); updatePanel(); }
function step() {
  if (isDone()) return;
  if (atGoal()) { finish("reached"); return; }
  if (!current) decide();
  if (current.goal) { if (!stepGoal()) return; } else stepWall();
  stats.steps++;
  const cy = Math.floor(robot.y), cx = Math.floor(robot.x);
  if (inside(cx, cy)) visits[cy][cx]++;
  decide();
  if (!current.goal) {
    stats.reward += readingReward(current.sense.d);
    const L = current.sense.d.SD_left;
    if (L >= P.rewards.left_low && L <= P.rewards.left_high) stats.inBand++;
  }
  if (atGoal()) finish("reached");
  else if (stats.steps >= maxSteps()) finish("timeout");
  else status = "running";
}
function stepWall() {
  let a = current.action, slipped = false;
  if (Math.random() < slip) { a = ACTIONS[Math.floor(Math.random() * ACTIONS.length)]; slipped = true; stats.slips++; }
  if (current.nearest) stats.unknown++;
  let adv = motion.step, h = robot.h;
  if (a === "Slight-Right-Turn") h -= motion.slight;
  else if (a === "Sharp-Right-Turn") { h -= motion.sharp; adv *= 0.3; }
  else if (a === "Slight-Left-Turn") h += motion.slight;
  robot.h = ((h % 360) + 360) % 360;
  const r = robot.h * Math.PI / 180;
  const nx = robot.x + adv * Math.cos(r) / CELL, ny = robot.y - adv * Math.sin(r) / CELL;
  let bumped = false;
  if (blocked(nx, ny)) { bumped = true; stats.bumps++; bumpFlash = 10; } else { robot.x = nx; robot.y = ny; stats.dist += adv; }
  disp.x = robot.x; disp.y = robot.y;
  trail.push({ x: robot.x, y: robot.y, c: COLORS[a] }); if (trail.length > 3000) trail.shift();
  pushLog(SHORT[a], COLORS[a], current.label + (current.nearest ? " (nearest)" : ""), slipped, bumped);
}
function stepGoal() {
  if (current.dir < 0) { finish("unreachable"); return false; }
  let a = current.dir, slipped = false;
  if (Math.random() < slip) { a = (a + (Math.random() < 0.5 ? 1 : 7)) % 8; slipped = true; stats.slips++; }
  const [nx, ny, cost, bump] = tryMove(current.cx, current.cy, a);
  robot.h = a * 45;
  if (bump) { stats.bumps++; bumpFlash = 10; }
  else { robot.x = nx + 0.5; robot.y = ny + 0.5; stats.dist += cost * CELL; }
  trail.push({ x: robot.x, y: robot.y, c: GOAL_COLOR }); if (trail.length > 3000) trail.shift();
  pushLog(DIR_ARROW[a] + " Move " + DIRS[a][2], GOAL_COLOR, current.label, slipped, bump);
  return true;
}
function pushLog(txt, col, state, slipped, bumped) {
  log.unshift({ t: stats.steps + 1, txt: txt, col: col, state: state, slipped: slipped, bumped: bumped });
  if (log.length > 8) log.pop();
}

// ---------------- drawing ----------------
function resize() {
  const wrap = $("left").clientWidth || 600;
  cellPx = Math.max(16, Math.floor(Math.min(wrap, 820) / W));
  dpr = window.devicePixelRatio || 1;
  cv.width = cellPx * W * dpr; cv.height = cellPx * H * dpr;
  cv.style.width = (cellPx * W) + "px"; cv.style.height = (cellPx * H) + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}
function render() {
  const cp = cellPx, cw = cp * W, ch = cp * H;
  ctx.clearRect(0, 0, cw, ch);
  const showG = goalMode() && $("showGridV").checked && gV !== null;
  let vmin = 0;
  if (showG) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (!grid[y][x] && gReach[y][x]) vmin = Math.min(vmin, gV[y][x]);
  let maxV = 1; for (const row of visits) for (const v of row) if (v > maxV) maxV = v;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (grid[y][x]) ctx.fillStyle = "#2B3442";
    else if (showG) {
      if (!gReach[y][x]) ctx.fillStyle = "#F1E4E0";
      else { const t = vmin < 0 ? gV[y][x] / vmin : 0; ctx.fillStyle = "rgba(47,93,138," + (0.04 + 0.30 * (1 - t)).toFixed(3) + ")"; }
    } else if ($("showVisits").checked && visits[y][x] > 0) {
      const a = 0.06 + 0.30 * Math.sqrt(visits[y][x] / maxV); ctx.fillStyle = "rgba(47,93,138," + a.toFixed(3) + ")";
    } else ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(x * cp, y * cp, cp, cp);
  }
  ctx.strokeStyle = "#E6E9ED"; ctx.lineWidth = 1; ctx.beginPath();
  for (let x = 0; x <= W; x++) { ctx.moveTo(x * cp + 0.5, 0); ctx.lineTo(x * cp + 0.5, ch); }
  for (let y = 0; y <= H; y++) { ctx.moveTo(0, y * cp + 0.5); ctx.lineTo(cw, y * cp + 0.5); }
  ctx.stroke();

  if (showG) {
    if (cp >= 22) {
      ctx.fillStyle = "rgba(28,35,48,0.45)"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.font = Math.round(cp * 0.42) + "px system-ui, sans-serif";
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
        if (!grid[y][x] && gReach[y][x] && gPol[y][x] >= 0) ctx.fillText(DIR_ARROW[gPol[y][x]], (x + 0.5) * cp, (y + 0.5) * cp);
    }
    let cx = Math.floor(robot.x), cy = Math.floor(robot.y), n = 0;
    if (inside(cx, cy) && gReach[cy][cx]) {
      ctx.setLineDash([6, 5]); ctx.strokeStyle = "rgba(47,93,138,0.9)"; ctx.lineWidth = 2.5; ctx.beginPath();
      ctx.moveTo((cx + 0.5) * cp, (cy + 0.5) * cp);
      while (!(cx === goal.x && cy === goal.y) && n < 200) {
        const a = gPol[cy][cx]; if (a < 0) break;
        const [nx, ny, , b] = tryMove(cx, cy, a); if (b) break;
        cx = nx; cy = ny; n++; ctx.lineTo((cx + 0.5) * cp, (cy + 0.5) * cp);
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
  }
  if ($("showTrail").checked && trail.length > 1) {
    ctx.lineWidth = 2.2; ctx.lineCap = "round"; const n = trail.length;
    for (let i = 1; i < n; i++) {
      ctx.globalAlpha = 0.15 + 0.85 * (i / n); ctx.strokeStyle = trail[i].c; ctx.beginPath();
      ctx.moveTo(trail[i - 1].x * cp, trail[i - 1].y * cp); ctx.lineTo(trail[i].x * cp, trail[i].y * cp); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  // start marker
  const sx = (start.x + 0.5) * cp, sy = (start.y + 0.5) * cp, sh = start.h * Math.PI / 180;
  ctx.beginPath(); ctx.arc(sx, sy, cp * 0.36, 0, 2 * Math.PI); ctx.fillStyle = "rgba(62,125,94,0.18)"; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = "#3E7D5E"; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sx + cp * 0.36 * Math.cos(sh), sy - cp * 0.36 * Math.sin(sh)); ctx.lineTo(sx + cp * 0.52 * Math.cos(sh), sy - cp * 0.52 * Math.sin(sh)); ctx.stroke();
  ctx.fillStyle = "#3E7D5E"; ctx.font = "bold " + Math.round(cp * 0.34) + "px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("S", sx, sy);
  // end marker: chequered flag
  const gx = goal.x * cp, gy = goal.y * cp, q = cp / 4;
  ctx.fillStyle = "rgba(200,85,61,0.12)"; ctx.fillRect(gx, gy, cp, cp);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
    ctx.fillStyle = (i + j) % 2 ? "#FFFFFF" : "#1C2330"; ctx.fillRect(gx + q * 1.2 + j * q * 0.7, gy + q * 0.7 + i * q * 0.7, q * 0.7, q * 0.7);
  }
  ctx.strokeStyle = "#1C2330"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(gx + q * 1.2, gy + q * 0.6); ctx.lineTo(gx + q * 1.2, gy + cp - q * 0.5); ctx.stroke();
  ctx.strokeStyle = "#C8553D"; ctx.lineWidth = 2; ctx.strokeRect(gx + 1.5, gy + 1.5, cp - 3, cp - 3);

  // robot (smoothed position)
  const k = current && current.goal ? 0.35 : 1;
  disp.x += (robot.x - disp.x) * k; disp.y += (robot.y - disp.y) * k;
  const rx = disp.x * cp, ry = disp.y * cp;
  if (current && $("showRays").checked && !showG) {
    for (const key of P.sd_cols) {
      const rs = current.sense.rays[key], col = ARC_COL[key];
      ctx.beginPath(); ctx.moveTo(rx, ry);
      for (const [ang, d] of rs.pts) { const a = ang * Math.PI / 180; ctx.lineTo(rx + d / CELL * cp * Math.cos(a), ry - d / CELL * cp * Math.sin(a)); }
      ctx.closePath(); ctx.fillStyle = col + "22"; ctx.fill(); ctx.strokeStyle = col + "88"; ctx.lineWidth = 1; ctx.stroke();
      const a = rs.best[0] * Math.PI / 180, d = rs.best[1];
      ctx.beginPath(); ctx.arc(rx + d / CELL * cp * Math.cos(a), ry - d / CELL * cp * Math.sin(a), 3.5, 0, 2 * Math.PI); ctx.fillStyle = col; ctx.fill();
    }
  }
  const rr = RAD / CELL * cp;
  if (bumpFlash > 0) { ctx.beginPath(); ctx.arc(rx, ry, rr + 6, 0, 2 * Math.PI); ctx.strokeStyle = "rgba(200,40,40," + (bumpFlash / 10) + ")"; ctx.lineWidth = 3; ctx.stroke(); }
  if (status === "reached") { ctx.beginPath(); ctx.arc(rx, ry, rr + 9, 0, 2 * Math.PI); ctx.strokeStyle = "#3E7D5E"; ctx.lineWidth = 3; ctx.stroke(); }
  ctx.beginPath(); ctx.arc(rx, ry, rr, 0, 2 * Math.PI);
  ctx.fillStyle = current ? (current.goal ? GOAL_COLOR : COLORS[current.action]) : "#2F5D8A"; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = "#FFFFFF"; ctx.stroke();
  const hr = robot.h * Math.PI / 180;
  ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(rx + (rr + 7) * Math.cos(hr), ry - (rr + 7) * Math.sin(hr));
  ctx.strokeStyle = "#1C2330"; ctx.lineWidth = 2.5; ctx.stroke();

  if (isDone()) {
    const msg = status === "reached" ? "End point reached in " + stats.steps + " steps" :
                status === "timeout" ? "Stopped after " + stats.steps + " steps without reaching the end" : "The end point cannot be reached";
    ctx.font = "bold 15px system-ui, sans-serif"; const tw = ctx.measureText(msg).width + 28;
    ctx.fillStyle = status === "reached" ? "rgba(62,125,94,0.94)" : "rgba(200,85,61,0.94)";
    ctx.fillRect((cw - tw) / 2, 10, tw, 32); ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(msg, cw / 2, 26);
  }
}

// ---------------- side panel ----------------
function fmt(v, d) { return (v === null || v === undefined || !isFinite(v)) ? "–" : Number(v).toFixed(d); }
function updatePanel() {
  if (!current || !stats) return;
  const st = $("status");
  const distTxt = stats.dist.toFixed(1) + " m";
  if (status === "reached") { st.className = "ok"; st.textContent = "End point reached in " + stats.steps + " steps (" + distTxt + ", " + stats.bumps + " bumps)."; }
  else if (status === "timeout") { st.className = "bad"; st.textContent = "Stopped after " + stats.steps + " steps. " +
      (goalMode() ? "Too much slip?" : "The wall-follower only reaches end points close to the wall it follows. Try the 'Go to end point' policy, or move the flag next to a wall."); }
  else if (status === "unreachable") { st.className = "bad"; st.textContent = "No path from here to the end point. Erase a wall or move the flag."; }
  else if (goalMode() && gReach && !gReach[start.y][start.x]) { st.className = "bad"; st.textContent = "The end point is walled off from the start."; }
  else { st.className = ""; st.textContent = running ? "Driving… " + stats.steps + " steps" : (stats.steps ? "Paused." : "Ready. Press Play."); }

  const pill = $("actionPill");
  pill.textContent = current.goal ? (current.dir >= 0 ? DIR_ARROW[current.dir] + " " + current.action : current.action) : current.action;
  pill.style.background = current.goal ? GOAL_COLOR : COLORS[current.action];
  if (current.goal) {
    const v = (gV && inside(current.cx, current.cy)) ? gV[current.cy][current.cx] : null;
    $("stateLbl").innerHTML = "State: <b>" + current.label + "</b><br><span style='color:#5B6573'>V(s) = " + fmt(v, 2) +
      " (roughly " + (v !== null ? Math.max(0, -v).toFixed(1) : "–") + " squares of travel left)</span>";
  } else {
    $("stateLbl").innerHTML = "State: <b>" + current.label + "</b>" +
      (current.nearest ? "<br><span style='color:#C8553D'>This exact reading never appears in the data, so the nearest known state is used.</span>" : "") +
      "<br><span style='color:#5B6573'>V*(s) = " + fmt(P.V[current.label], 3) + "</span>";
  }
  let h = "";
  for (const k of P.sd_cols) {
    const v = current.sense.d[k], used = !current.goal && P.features.indexOf(k) >= 0;
    const bin = used ? P.bin_names[binIndex(k, v)] : (current.goal ? "" : "not used");
    h += "<div class='drow' style='opacity:" + (used || current.goal ? 1 : 0.45) + "'><span>" + k.replace("SD_", "") + "</span>" +
      "<div class='dtrack'><div class='dfill' style='width:" + Math.min(100, v / MAXD * 100).toFixed(1) + "%;background:" + ARC_COL[k] + "'></div></div>" +
      "<span>" + v.toFixed(2) + " m" + (bin ? ", " + bin : "") + "</span></div>";
  }
  $("dists").innerHTML = h;

  h = "";
  if (current.goal) {
    $("qTitle").textContent = "Q-values of the grid MDP (8 moves)";
    const ok = gV && inside(current.cx, current.cy) && gReach[current.cy][current.cx] && !(current.cx === goal.x && current.cy === goal.y);
    if (ok) {
      const q = gridQ(current.cx, current.cy);
      const order = q.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      const qmin = order[order.length - 1][0], qmax = order[0][0];
      for (const [v, i] of order) {
        const w = qmax > qmin ? 10 + 90 * (v - qmin) / (qmax - qmin) : 100, best = i === current.dir;
        h += "<div class='qrow'><span style='font-weight:" + (best ? 700 : 400) + "'>" + DIR_ARROW[i] + " " + DIRS[i][2] + (best ? " ★" : "") + "</span>" +
          "<div class='dtrack'><div class='dfill' style='width:" + w.toFixed(1) + "%;background:" + GOAL_COLOR + "'></div></div><span>" + v.toFixed(2) + "</span></div>";
      }
    } else h = "<span style='color:#5B6573'>No moves to rank here.</span>";
  } else {
    $("qTitle").textContent = "Q-values for this state (sensor MDP)";
    const q = P.Q[current.label] || [], fin = q.filter(x => x !== null);
    const qmin = fin.length ? Math.min.apply(null, fin) : 0, qmax = fin.length ? Math.max.apply(null, fin) : 1;
    ACTIONS.forEach((a, i) => {
      const v = q[i], ok = v !== null && v !== undefined;
      const w = ok ? (qmax > qmin ? 15 + 85 * (v - qmin) / (qmax - qmin) : 100) : 0, isOpt = a === P.opt[current.label];
      h += "<div class='qrow'><span style='font-weight:" + (isOpt ? 700 : 400) + "'>" + SHORT[a] + (isOpt ? " ★" : "") + "</span>" +
        "<div class='dtrack'><div class='dfill' style='width:" + w.toFixed(1) + "%;background:" + COLORS[a] + "'></div></div>" +
        "<span>" + (ok ? v.toFixed(2) : "unseen") + "</span></div>";
    });
  }
  $("qvals").innerHTML = h;
  const band = stats.steps ? (100 * stats.inBand / stats.steps) : 0;
  $("stats").innerHTML =
    "<div><b>" + stats.steps + "</b>steps</div>" +
    "<div><b>" + distTxt + "</b>distance travelled</div>" +
    "<div><b>" + stats.bumps + "</b>bumps into walls</div>" +
    "<div><b>" + stats.slips + "</b>slipped actions</div>" +
    (current.goal ? "" :
      "<div><b>" + band.toFixed(0) + "%</b>time in wall band</div>" +
      "<div><b>" + stats.reward.toFixed(1) + "</b>total reward</div>");
  $("log").innerHTML = log.map(e => "<div><span class='dot' style='background:" + e.col + "'></span><span>#" + e.t + " " + e.txt +
    (e.slipped ? " <i>(slip)</i>" : "") + (e.bumped ? " <b style='color:#C8553D'>bump</b>" : "") +
    "<br><span style='color:#5B6573'>" + e.state + "</span></span></div>").join("") || "<span style='color:#5B6573'>Press Play or Step.</span>";
}

// ---------------- reset / controls ----------------
function toStart() {
  robot = { x: start.x + 0.5, y: start.y + 0.5, h: start.h }; disp = { x: robot.x, y: robot.y };
  trail = []; log = []; stats = newStats(); status = "ready";
  for (const row of visits) row.fill(0);
  solveGridMDP(); decide(); render(); updatePanel();
}
function loadLayout() {
  blankGrid(); const L = LAYOUTS[$("layout").value]; L.build();
  start = { x: L.start[0], y: L.start[1], h: L.start[2] }; goal = { x: L.goal[0], y: L.goal[1] };
  toStart();
}
function setRunning(v) { running = v; $("play").textContent = running ? "⏸ Pause" : "▶ Play"; }
function setMode(m) {
  mode = m; $("mWalls").classList.toggle("on", m === "walls"); $("mStart").classList.toggle("on", m === "start"); $("mGoal").classList.toggle("on", m === "goal");
}
$("play").onclick = () => {
  if (!running && isDone()) toStart();
  setRunning(!running); updatePanel();
};
$("stepBtn").onclick = () => { setRunning(false); step(); render(); updatePanel(); };
$("reset").onclick = () => { setRunning(false); toStart(); };
$("layout").onchange = () => { setRunning(false); loadLayout(); };
$("policy").onchange = () => {
  if (goalMode()) { robot.x = Math.floor(robot.x) + 0.5; robot.y = Math.floor(robot.y) + 0.5; disp.x = robot.x; disp.y = robot.y; }
  if (!running && !isDone()) status = stats.steps ? "paused" : "ready";
  decide(); render(); updatePanel();
};
$("mWalls").onclick = () => setMode("walls");
$("mStart").onclick = () => setMode("start");
$("mGoal").onclick = () => setMode("goal");
$("rotL").onclick = () => { start.h = (start.h + 45) % 360; if (!stats.steps) toStart(); else render(); };
$("rotR").onclick = () => { start.h = (start.h + 315) % 360; if (!stats.steps) toStart(); else render(); };
function bindRange(id, lbl, fn, after) {
  const el = $(id);
  el.oninput = () => { $(lbl).textContent = fn(Number(el.value)); if (after) after(); };
  $(lbl).textContent = fn(Number(el.value));
}
bindRange("slip", "slipV", v => { slip = v / 100; return v + "%"; }, () => { solveGridMDP(); decide(); render(); updatePanel(); });
bindRange("noise", "noiseV", v => { noise = v / 100; return v + " cm"; });
bindRange("stepm", "stepmV", v => { motion.step = v / 100; return v + " cm"; });
bindRange("slight", "slightV", v => { motion.slight = v; return v + "°"; });
bindRange("sharp", "sharpV", v => { motion.sharp = v; return v + "°"; });
["showRays", "showVisits", "showTrail", "showGridV"].forEach(id => $(id).onchange = render);

let painting = null;
function cellFromEvent(e) { const rect = cv.getBoundingClientRect(); return [Math.floor((e.clientX - rect.left) / cellPx), Math.floor((e.clientY - rect.top) / cellPx)]; }
function interior(cx, cy) { return cx > 0 && cy > 0 && cx < W - 1 && cy < H - 1; }
function paintAt(cx, cy) {
  if (!interior(cx, cy)) return;
  if (painting === 1) {
    if (cx === goal.x && cy === goal.y) return;
    if (cx === start.x && cy === start.y) return;
    if (cx === Math.floor(robot.x) && cy === Math.floor(robot.y)) return;
  }
  grid[cy][cx] = painting;
}
cv.addEventListener("pointerdown", e => {
  const [cx, cy] = cellFromEvent(e);
  if (!interior(cx, cy)) return;
  if (mode === "start") {
    if (grid[cy][cx]) return;
    start.x = cx; start.y = cy; setRunning(false); toStart(); return;
  }
  if (mode === "goal") {
    if (grid[cy][cx]) return;
    goal.x = cx; goal.y = cy; solveGridMDP();
    if (isDone()) status = "paused";
    decide(); render(); updatePanel(); return;
  }
  painting = grid[cy][cx] ? 0 : 1; paintAt(cx, cy);
  try { cv.setPointerCapture(e.pointerId); } catch (err) { /* older browsers */ }
  render();
});
cv.addEventListener("pointermove", e => {
  if (painting === null) return;
  const [cx, cy] = cellFromEvent(e);
  paintAt(cx, cy); render();
});
window.addEventListener("pointerup", () => {
  if (painting !== null) { painting = null; solveGridMDP(); if (status === "unreachable") status = "paused"; decide(); render(); updatePanel(); }
});

$("legend").innerHTML = ACTIONS.map(a => "<span><span class='dot' style='margin:0;background:" + COLORS[a] + "'></span>" + a + "</span>").join("") +
  "<span><span class='dot' style='margin:0;background:" + GOAL_COLOR + "'></span>Grid move (end-point policy)</span>";

let acc = 0, lastT = null;
function loop(t) {
  if (lastT === null) lastT = t;
  const dt = Math.min(0.1, (t - lastT) / 1000); lastT = t;
  if (running) {
    const spd = Number($("speed").value) * (goalMode() ? 0.35 : 1);
    acc += dt * spd;
    let n = 0, moved = false;
    while (acc >= 1 && n < 40 && running) { step(); acc -= 1; n++; moved = true; }
    if (moved) updatePanel();
  } else acc = 0;
  if (bumpFlash > 0) bumpFlash--;
  render();
  requestAnimationFrame(loop);
}
window.__simTest = { visits: function () { return visits; }, run: function (n) { let i = 0; while (i < n && !isDone()) { step(); i++; } updatePanel(); return { status: status, steps: stats.steps, bumps: stats.bumps }; } };
window.addEventListener("resize", resize);
loadLayout(); resize(); requestAnimationFrame(loop);
})();
</script>
</body></html>
"""

VI_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  :root { --ink:#1C2330; --muted:#5B6573; --line:#E3E6EA; --panel:#FFFFFF; --bg:#FBFBF9; --primary:#2F5D8A; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Source Sans Pro","Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;
         color:var(--ink); background:var(--bg); font-size:14px; }
  #app { display:flex; gap:16px; flex-wrap:wrap; padding:4px; }
  #left { flex:1.5 1 480px; min-width:300px; }
  #right { flex:1 1 320px; min-width:280px; display:flex; flex-direction:column; gap:10px; }
  .bar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:8px; }
  button { font:inherit; border:1px solid var(--line); background:var(--panel); color:var(--ink);
           padding:5px 11px; border-radius:6px; cursor:pointer; }
  button:hover { border-color:var(--primary); }
  button:focus-visible, input:focus-visible, select:focus-visible { outline:2px solid var(--primary); outline-offset:1px; }
  button.primary { background:var(--primary); color:#fff; border-color:var(--primary); min-width:82px; }
  select { font:inherit; padding:4px 6px; border:1px solid var(--line); border-radius:6px; background:#fff; }
  input[type=range] { accent-color:var(--primary); }
  #frame { flex:1; min-width:160px; }
  #iterBig { font-size:30px; font-weight:700; line-height:1; }
  .sub { color:var(--muted); font-size:12.5px; }
  canvas { display:block; }
  #gridCv { cursor:pointer; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .card h4 { margin:0 0 6px 0; font-size:13px; color:var(--muted); font-weight:600; }
  .code { font-family:"Source Code Pro",Menlo,Consolas,monospace; font-size:12.5px; line-height:1.55; }
  .code div { padding:0 6px; border-radius:4px; white-space:pre; }
  .code div.on { background:#DCE7F2; }
  table { border-collapse:collapse; width:100%; font-size:12.5px; }
  td, th { padding:3px 4px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
  th { color:var(--muted); font-weight:600; }
  tr.best td { background:#EEF3F8; font-weight:700; }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:4px; }
  #tip { position:absolute; pointer-events:none; background:#1C2330; color:#fff; font-size:12px; padding:5px 8px; border-radius:5px; display:none; z-index:5; }
  .row { display:flex; gap:14px; align-items:flex-end; flex-wrap:wrap; margin-bottom:8px; }
  .scale { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); margin-top:6px; }
  #scaleCv { border-radius:3px; }
</style></head>
<body>
<div id="app">
  <div id="left">
    <div class="row">
      <div><div class="sub">Iteration k</div><div id="iterBig">0</div></div>
      <div><div class="sub">Largest change Δ</div><div id="deltaTxt" style="font-size:18px;font-weight:600">–</div></div>
      <div><div class="sub">Greedy moves changed</div><div id="changedTxt" style="font-size:18px;font-weight:600">–</div></div>
    </div>
    <div class="bar">
      <button id="first" title="First iteration">⏮</button>
      <button id="prev" title="Previous">◀</button>
      <button id="play" class="primary">▶ Play</button>
      <button id="next" title="Next">▶|</button>
      <button id="last" title="Converged">⏭</button>
      <input id="frame" type="range" min="0" max="0" value="0">
      <select id="speed"><option value="2">Slow</option><option value="6" selected>Normal</option><option value="15">Fast</option></select>
    </div>
    <div id="gridWrap" style="position:relative">
      <canvas id="gridCv"></canvas>
      <div id="tip"></div>
    </div>
    <div class="scale"><span id="vminTxt"></span><canvas id="scaleCv" width="160" height="10"></canvas><span id="vmaxTxt"></span>
      <span style="margin-left:10px">Arrow = best move under V<sub>k</sub>. Grey = combination never seen in the data. Click a square to see its update.</span></div>
    <div class="card" style="margin-top:10px"><h4>Convergence: Δ per iteration (log scale)</h4><canvas id="convCv"></canvas></div>
  </div>
  <div id="right">
    <div class="card">
      <h4>Algorithm</h4>
      <div class="code" id="code"></div>
    </div>
    <div class="card">
      <h4 id="backupTitle">Bellman update for the selected state</h4>
      <div id="backup"></div>
    </div>
  </div>
</div>
<script>
(function () {
"use strict";
const P = __PAYLOAD__;
const $ = id => document.getElementById(id);
const S = P.states.length, A = P.actions.length;
const ARROW = { "Move-Forward": "↑", "Slight-Right-Turn": "↗", "Sharp-Right-Turn": "→", "Slight-Left-Turn": "↖" };
const SHORT = { "Move-Forward": "Forward", "Slight-Right-Turn": "Slight right", "Sharp-Right-Turn": "Sharp right", "Slight-Left-Turn": "Slight left" };
const stateIndex = {}; P.states.forEach((s, i) => stateIndex[s] = i);

// ---------- layout of the state-space grid ----------
const nF = P.features.length, split = Math.floor(nF / 2);
const rowF = P.features.slice(0, split), colF = P.features.slice(split);
const nb = P.nbins;               // bins per feature (array aligned with features)
function combos(feats) {
  let out = [[]];
  feats.forEach(f => { const k = nb[P.features.indexOf(f)]; const nxt = []; out.forEach(c => { for (let i = 0; i < k; i++) nxt.push(c.concat([i])); }); out = nxt; });
  return out;
}
const rows = combos(rowF), cols = combos(colF);
const cellOf = {};                // state index -> [r, c]
const atCell = {};                // "r,c" -> state index
P.states.forEach((s, i) => {
  const b = P.bins[s];
  const rk = b.slice(0, split).join(","), ck = b.slice(split).join(",");
  const r = rows.findIndex(x => x.join(",") === rk), c = cols.findIndex(x => x.join(",") === ck);
  cellOf[i] = [r, c]; atCell[r + "," + c] = i;
});
function abbrev(name) { return name.split(" ").map(w => w[0].toUpperCase()).join(""); }
function comboLabel(feats, combo) { return combo.map((b, j) => abbrev(P.bin_names[b])).join("/"); }

// ---------- value iteration maths in the browser ----------
function qValues(s, V) {
  const q = new Array(A).fill(null);
  for (let a = 0; a < A; a++) {
    if (!P.valid[s][a]) continue;
    let ev = 0; for (const [s2, p] of P.T[s][a]) ev += p * V[s2];
    q[a] = P.R[s][a] + P.gamma * ev;
  }
  return q;
}
function greedy(V) {
  const pol = new Array(S);
  for (let s = 0; s < S; s++) { const q = qValues(s, V); let best = -1, bv = -Infinity; q.forEach((v, a) => { if (v !== null && v > bv) { bv = v; best = a; } }); pol[s] = best; }
  return pol;
}
const frames = P.frames;           // [{k, V, delta}]
const policies = frames.map(f => greedy(f.V));

// ---------- colours ----------
const stops = [[200, 85, 61], [242, 232, 207], [47, 93, 138]];
function colorFor(v) {
  const t = P.vmax > P.vmin ? Math.max(0, Math.min(1, (v - P.vmin) / (P.vmax - P.vmin))) : 0.5;
  const seg = t < 0.5 ? 0 : 1, u = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const c = stops[seg].map((x, i) => Math.round(x + (stops[seg + 1][i] - x) * u));
  return "rgb(" + c.join(",") + ")";
}
function textColorFor(v) {
  const t = P.vmax > P.vmin ? (v - P.vmin) / (P.vmax - P.vmin) : 0.5;
  return (t < 0.18 || t > 0.72) ? "#FFFFFF" : "#1C2330";
}
(function drawScale() {
  const c = $("scaleCv").getContext("2d");
  for (let x = 0; x < 160; x++) { c.fillStyle = colorFor(P.vmin + (P.vmax - P.vmin) * x / 159); c.fillRect(x, 0, 1, 10); }
  $("vminTxt").textContent = P.vmin.toFixed(1); $("vmaxTxt").textContent = P.vmax.toFixed(1);
})();

// ---------- state ----------
let fi = 0, playing = false, selected = P.default_state, geom = null;
$("frame").max = frames.length - 1;

// ---------- drawing the grid ----------
const gcv = $("gridCv"), g = gcv.getContext("2d");
function drawGrid() {
  const wrapW = $("left").clientWidth || 560, dpr = window.devicePixelRatio || 1;
  const labW = rowF.length ? 26 + 13 * rowF.length : 8, labH = colF.length ? 16 + 14 * Math.max(1, colF.length) : 8;
  const nR = rows.length, nC = cols.length;
  let cs = Math.floor((Math.min(wrapW, 760) - labW - 4) / nC);
  cs = Math.max(10, Math.min(cs, 58));
  const w = labW + cs * nC + 4, h = labH + cs * nR + 4;
  gcv.width = w * dpr; gcv.height = h * dpr; gcv.style.width = w + "px"; gcv.style.height = h + "px";
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  geom = { labW: labW, labH: labH, cs: cs };
  const f = frames[fi], pol = policies[fi], prevPol = fi > 0 ? policies[fi - 1] : null;
  g.font = "11px system-ui, sans-serif"; g.fillStyle = "#5B6573"; g.textBaseline = "middle";
  // axis titles
  g.textAlign = "left";
  g.fillText((rowF.length ? "rows: " + rowF.map(x => x.replace("SD_", "")).join(" / ") + "    " : "") +
             "columns: " + colF.map(x => x.replace("SD_", "")).join(" / "), 2, 7);
  if (cs >= 16) {
    g.textAlign = "center";
    cols.forEach((c, j) => g.fillText(comboLabel(colF, c), labW + j * cs + cs / 2, labH - 7));
    g.textAlign = "right";
    rows.forEach((r, i) => { if (rowF.length) g.fillText(comboLabel(rowF, r), labW - 4, labH + i * cs + cs / 2); });
  }
  for (let r = 0; r < nR; r++) for (let c = 0; c < nC; c++) {
    const x = labW + c * cs, y = labH + r * cs, s = atCell[r + "," + c];
    if (s === undefined) {
      g.fillStyle = "#EEF0F2"; g.fillRect(x + 1, y + 1, cs - 2, cs - 2);
      g.strokeStyle = "#DADDE1"; g.lineWidth = 1; g.beginPath(); g.moveTo(x + 3, y + cs - 3); g.lineTo(x + cs - 3, y + 3); g.stroke();
      continue;
    }
    const v = f.V[s];
    g.fillStyle = colorFor(v); g.fillRect(x + 1, y + 1, cs - 2, cs - 2);
    if (prevPol && prevPol[s] !== pol[s]) { g.strokeStyle = "#E0A43A"; g.lineWidth = 2.5; g.strokeRect(x + 2.5, y + 2.5, cs - 5, cs - 5); }
    if (pol[s] >= 0 && cs >= 14) {
      g.fillStyle = textColorFor(v); g.textAlign = "center";
      g.font = "bold " + Math.round(cs * 0.42) + "px system-ui, sans-serif";
      g.fillText(ARROW[P.actions[pol[s]]], x + cs / 2, y + cs / 2 + (cs >= 34 ? -5 : 0));
      if (cs >= 34) { g.font = "10px system-ui, sans-serif"; g.fillText(v.toFixed(1), x + cs / 2, y + cs - 8); }
    }
    if (s === selected) { g.strokeStyle = "#1C2330"; g.lineWidth = 3; g.strokeRect(x + 1.5, y + 1.5, cs - 3, cs - 3); }
  }
}

// ---------- convergence chart ----------
const ccv = $("convCv"), cc = ccv.getContext("2d");
function drawConv() {
  const w = Math.max(260, ($("left").clientWidth || 560) - 26), h = 150, dpr = window.devicePixelRatio || 1;
  ccv.width = w * dpr; ccv.height = h * dpr; ccv.style.width = w + "px"; ccv.style.height = h + "px";
  cc.setTransform(dpr, 0, 0, dpr, 0, 0); cc.clearRect(0, 0, w, h);
  const pts = P.conv; if (!pts.length) return;
  const L = 46, Rm = 10, T = 8, B = 22, n = P.iterations;
  const ys = pts.map(p => Math.log10(Math.max(p[1], 1e-14))).concat([Math.log10(P.tol)]);
  const ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
  const X = k => L + (w - L - Rm) * (n > 1 ? (k - 1) / (n - 1) : 0);
  const Y = v => T + (h - T - B) * (ymax > ymin ? (ymax - Math.log10(Math.max(v, 1e-14))) / (ymax - ymin) : 0.5);
  cc.strokeStyle = "#E3E6EA"; cc.lineWidth = 1; cc.font = "10px system-ui, sans-serif"; cc.fillStyle = "#5B6573";
  cc.textAlign = "right"; cc.textBaseline = "middle";
  for (let e = Math.ceil(ymin); e <= Math.floor(ymax); e++) {
    const y = Y(Math.pow(10, e)); cc.beginPath(); cc.moveTo(L, y); cc.lineTo(w - Rm, y); cc.stroke(); cc.fillText("1e" + e, L - 4, y);
  }
  cc.textAlign = "center"; cc.fillText("1", L, h - 8); cc.fillText(String(n), w - Rm, h - 8);
  cc.setLineDash([5, 4]); cc.strokeStyle = "#C8553D"; cc.beginPath(); cc.moveTo(L, Y(P.tol)); cc.lineTo(w - Rm, Y(P.tol)); cc.stroke(); cc.setLineDash([]);
  cc.fillStyle = "#C8553D"; cc.textAlign = "left"; cc.fillText("threshold θ", L + 4, Y(P.tol) - 8);
  cc.strokeStyle = "#2F5D8A"; cc.lineWidth = 2; cc.beginPath();
  pts.forEach((p, i) => { const x = X(p[0]), y = Y(p[1]); if (i === 0) cc.moveTo(x, y); else cc.lineTo(x, y); }); cc.stroke();
  const k = frames[fi].k;
  if (k >= 1) {
    const x = X(k), y = Y(frames[fi].delta);
    cc.strokeStyle = "rgba(28,35,48,0.35)"; cc.lineWidth = 1; cc.beginPath(); cc.moveTo(x, T); cc.lineTo(x, h - B); cc.stroke();
    cc.beginPath(); cc.arc(x, y, 5, 0, 2 * Math.PI); cc.fillStyle = "#E0A43A"; cc.fill(); cc.strokeStyle = "#fff"; cc.lineWidth = 2; cc.stroke();
  }
}

// ---------- pseudocode ----------
const CODE = [
  "V(s) ← 0 for every state s",
  "repeat for k = 1, 2, 3, …",
  "  for each state s:",
  "    for each action a seen in s:",
  "      Q(s,a) ← R(s,a) + γ Σ P(s'|s,a) V(s')",
  "    V_new(s) ← max over a of Q(s,a)",
  "  Δ ← max |V_new(s) − V(s)| ;  V ← V_new",
  "until Δ < θ          (θ = " + P.tol.toExponential(0) + ", γ = " + P.gamma + ")",
  "π*(s) ← action with the largest Q(s,a)"
];
let codePulse = 0;
function drawCode() {
  const k = frames[fi].k, last = fi === frames.length - 1;
  let on;
  if (k === 0) on = [0];
  else if (last && P.converged) on = [7, 8];
  else { const cyc = [[2, 3, 4], [5], [6]]; on = playing ? cyc[codePulse % 3] : [2, 3, 4, 5, 6]; }
  $("code").innerHTML = CODE.map((l, i) => "<div class='" + (on.indexOf(i) >= 0 ? "on" : "") + "'>" + (i + 1) + "  " + l.replace(/</g, "&lt;") + "</div>").join("");
}

// ---------- Bellman backup table ----------
function fmt(v, d) { return (v === null || v === undefined || !isFinite(v)) ? "–" : Number(v).toFixed(d); }
function drawBackup() {
  const f = frames[fi], s = selected, V = f.V, q = qValues(s, V);
  let best = -1, bv = -Infinity; q.forEach((v, a) => { if (v !== null && v > bv) { bv = v; best = a; } });
  $("backupTitle").textContent = "Bellman update for " + P.states[s];
  let h = "<div class='sub' style='margin-bottom:6px'>Uses V<sub>" + f.k + "</sub> to compute V<sub>" + (f.k + 1) + "</sub>. " +
          "Current V<sub>" + f.k + "</sub>(s) = <b>" + fmt(V[s], 3) + "</b></div>";
  h += "<table><tr><th>Action</th><th>R(s,a)</th><th>Where it leads: P × V<sub>" + f.k + "</sub>(s')</th><th>Q</th></tr>";
  P.actions.forEach((a, ai) => {
    if (!P.valid[s][ai]) {
      h += "<tr style='color:#9AA3AE'><td><span class='dot' style='background:" + P.colors[a] + "'></span>" + SHORT[a] + "</td><td colspan='3'>never taken here in the data</td></tr>";
      return;
    }
    const tr = P.T[s][ai].slice().sort((x, y) => y[1] - x[1]);
    const shown = tr.slice(0, 3).map(([s2, p]) => p.toFixed(2) + " × " + V[s2].toFixed(2)).join(" + ");
    const more = tr.length > 3 ? " + " + (tr.length - 3) + " more" : "";
    h += "<tr class='" + (ai === best ? "best" : "") + "'><td><span class='dot' style='background:" + P.colors[a] + "'></span>" + SHORT[a] + (ai === best ? " ★" : "") +
         "</td><td>" + fmt(P.R[s][ai], 2) + "</td><td>γ·(" + shown + more + ")</td><td>" + fmt(q[ai], 2) + "</td></tr>";
  });
  h += "</table>";
  h += "<div style='margin-top:8px'>V<sub>" + (f.k + 1) + "</sub>(s) = max Q = <b>" + fmt(bv, 3) + "</b>" +
       (best >= 0 ? "  →  best move: <b style='color:" + P.colors[P.actions[best]] + "'>" + P.actions[best] + "</b>" : "") + "</div>";
  h += "<div class='sub' style='margin-top:4px'>Visits in the data: " + P.visits[s] + ". Final V*(s) = " + fmt(P.vfinal[s], 3) + "</div>";
  $("backup").innerHTML = h;
}

function drawAll() {
  const f = frames[fi];
  $("iterBig").textContent = f.k;
  $("deltaTxt").textContent = f.k === 0 ? "–" : f.delta.toExponential(2);
  let ch = "–";
  if (fi > 0) { let n = 0; for (let s = 0; s < S; s++) if (policies[fi][s] !== policies[fi - 1][s]) n++; ch = String(n); }
  $("changedTxt").textContent = ch;
  $("frame").value = fi;
  drawGrid(); drawConv(); drawCode(); drawBackup();
}
function setPlaying(v) { playing = v; $("play").textContent = playing ? "⏸ Pause" : "▶ Play"; drawCode(); }
$("play").onclick = () => { if (!playing && fi === frames.length - 1) fi = 0; setPlaying(!playing); drawAll(); };
$("first").onclick = () => { setPlaying(false); fi = 0; drawAll(); };
$("last").onclick = () => { setPlaying(false); fi = frames.length - 1; drawAll(); };
$("prev").onclick = () => { setPlaying(false); fi = Math.max(0, fi - 1); drawAll(); };
$("next").onclick = () => { setPlaying(false); fi = Math.min(frames.length - 1, fi + 1); drawAll(); };
$("frame").oninput = e => { setPlaying(false); fi = Number(e.target.value); drawAll(); };

function cellAt(e) {
  if (!geom) return undefined;
  const rect = gcv.getBoundingClientRect(), x = e.clientX - rect.left - geom.labW, y = e.clientY - rect.top - geom.labH;
  if (x < 0 || y < 0) return undefined;
  const c = Math.floor(x / geom.cs), r = Math.floor(y / geom.cs);
  return atCell[r + "," + c];
}
gcv.addEventListener("click", e => { const s = cellAt(e); if (s !== undefined) { selected = s; drawAll(); } });
gcv.addEventListener("mousemove", e => {
  const s = cellAt(e), tip = $("tip");
  if (s === undefined) { tip.style.display = "none"; return; }
  const pol = policies[fi][s];
  tip.innerHTML = P.states[s] + "<br>V<sub>" + frames[fi].k + "</sub> = " + frames[fi].V[s].toFixed(3) + (pol >= 0 ? "<br>best: " + P.actions[pol] : "");
  const rect = $("gridWrap").getBoundingClientRect();
  tip.style.left = (e.clientX - rect.left + 12) + "px"; tip.style.top = (e.clientY - rect.top + 12) + "px"; tip.style.display = "block";
});
gcv.addEventListener("mouseleave", () => { $("tip").style.display = "none"; });

let acc = 0, lastT = null;
function loop(t) {
  if (lastT === null) lastT = t;
  const dt = Math.min(0.1, (t - lastT) / 1000); lastT = t;
  if (playing) {
    acc += dt * Number($("speed").value);
    if (acc >= 1) {
      acc = 0; codePulse++;
      if (fi < frames.length - 1) { fi++; drawAll(); } else { setPlaying(false); drawAll(); }
    }
  }
  requestAnimationFrame(loop);
}
window.addEventListener("resize", drawAll);
drawAll(); requestAnimationFrame(loop);
})();
</script>
</body></html>
"""


# ===========================================================================
# Day 5 notebook programs: MDP, ADP, Monte Carlo policy search, Hooke-Jeeves
# ===========================================================================

def edit_table(df, key):
    """Editable table that works across Streamlit versions (falls back to read-only)."""
    editor = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
    if editor is None:
        show_df(df, hide_index=False)
        return df
    try:
        if NEW_ST:
            return editor(df, key=key, width="stretch")
        return editor(df, key=key, use_container_width=True)
    except TypeError:
        return editor(df, key=key)


def notebook_code(code):
    with st.expander("Code from the Day 5 notebook"):
        st.code(code, language="python")


# ---------------------------------------------------------------------------
# MDP: the notebook's 3-state, 2-action example
# ---------------------------------------------------------------------------

NB_STATES = ["s1", "s2", "s3"]
NB_ACTIONS = ["a1", "a2"]
NB_R = np.array([[5, 10], [2, 3], [8, 1]], dtype=float)
NB_T = np.array([
    [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]],
    [[0.3, 0.4, 0.3], [0.5, 0.3, 0.2]],
    [[0.4, 0.4, 0.2], [0.2, 0.5, 0.3]],
])
NB_ACTION_COLORS = {"a1": "#2F5D8A", "a2": "#E0A43A"}


def nb_value_iteration(T, R, gamma, iterations=1000, tolerance=None):
    """The notebook's value_iteration, plus a record of every V_k (and an optional early stop)."""
    S, A = R.shape
    V = np.zeros(S)
    hist, deltas = [V.copy()], []
    for i in range(iterations):
        V_new = np.zeros(S)
        for s in range(S):
            Q_sa = np.zeros(A)
            for a in range(A):
                Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
            V_new[s] = np.max(Q_sa)
        deltas.append(float(np.max(np.abs(V_new - V))))
        V = V_new
        hist.append(V.copy())
        if tolerance is not None and deltas[-1] < tolerance:
            break
    Q = R + gamma * np.einsum("ijk,k->ij", T, V)
    return V, Q, hist, deltas


def clean_transitions(T):
    """Clip negatives and make every row sum to 1. Returns (T, list of fixed rows)."""
    T = np.clip(np.nan_to_num(T.astype(float)), 0, None)
    fixed = []
    for s in range(T.shape[0]):
        for a in range(T.shape[1]):
            tot = T[s, a].sum()
            if tot <= 0:
                T[s, a] = 1.0 / T.shape[2]
                fixed.append((s, a))
            elif abs(tot - 1) > 1e-6:
                T[s, a] = T[s, a] / tot
                fixed.append((s, a))
    return T, fixed


def mdp_graph(T, V, policy, show):
    lines = ['digraph MDP {', 'rankdir=LR; bgcolor="transparent";',
             'node [shape=circle, style=filled, fillcolor="#EEF3F8", color="#2F5D8A", fontname="Helvetica", fontsize=12];',
             'edge [fontname="Helvetica", fontsize=10];']
    for s, name in enumerate(NB_STATES):
        lines.append(f'{name} [label="{name}\\nV*={V[s]:.1f}\\nbest: {NB_ACTIONS[policy[s]]}"];')
    for s in range(len(NB_STATES)):
        for a, act in enumerate(NB_ACTIONS):
            is_opt = policy[s] == a
            if show == "Best action only" and not is_opt:
                continue
            for s2 in range(len(NB_STATES)):
                p = T[s, a, s2]
                if p < 0.005:
                    continue
                style = "solid" if is_opt else "dashed"
                lines.append(f'{NB_STATES[s]} -> {NB_STATES[s2]} [label="{act}: {p:.2f}", color="{NB_ACTION_COLORS[act]}", '
                             f'fontcolor="{NB_ACTION_COLORS[act]}", penwidth={0.6 + 4 * p:.2f}, style={style}];')
    lines.append("}")
    return "\n".join(lines)


def render_mdp_tab():
    st.subheader("Markov Decision Process: the notebook example")
    st.markdown(
        '<div class="note">Three states, two actions. Edit any reward or probability and value iteration re-runs '
        'instantly. Each transition row is automatically rescaled to sum to 1.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.1])
    with left:
        st.markdown("**Rewards R(s, a)**")
        r_df = edit_table(pd.DataFrame(NB_R, index=NB_STATES, columns=NB_ACTIONS), key="mdp_rewards")
        st.markdown("**Transition probabilities P(s' | s, a)**")
        idx = [f"{s}, {a}" for s in NB_STATES for a in NB_ACTIONS]
        t_df = edit_table(pd.DataFrame(NB_T.reshape(6, 3), index=idx, columns=NB_STATES), key="mdp_trans")
        gamma_nb = st.slider("Discount factor γ", 0.0, 0.99, 0.9, 0.01, key="mdp_gamma")
        stop = st.radio("Stopping rule", ["Fixed 1000 iterations (as in the notebook)", "Stop at a tolerance"],
                        key="mdp_stop")
        tol_nb = None
        if stop.startswith("Stop"):
            tol_nb = float(st.select_slider("Tolerance", options=["1e-02", "1e-03", "1e-04", "1e-06", "1e-08"],
                                            value="1e-06", key="mdp_tol"))

    try:
        R = pd.DataFrame(r_df).apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float).reshape(3, 2)
        T_raw = pd.DataFrame(t_df).apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(float).reshape(3, 2, 3)
    except Exception:
        st.error("The tables could not be read, so the notebook values are used.")
        R, T_raw = NB_R.copy(), NB_T.copy()
    T, fixed = clean_transitions(T_raw)
    V, Q, hist, deltas = nb_value_iteration(T, R, gamma_nb, 1000, tol_nb)
    policy = Q.argmax(axis=1)

    with right:
        if fixed:
            st.caption("Rescaled to sum to 1: " + ", ".join(f"({NB_STATES[s]}, {NB_ACTIONS[a]})" for s, a in fixed))
        m = st.columns(3)
        for s in range(3):
            m[s].metric(f"V*({NB_STATES[s]})", f"{V[s]:.2f}", f"best: {NB_ACTIONS[policy[s]]}", delta_color="off")
        show = st.radio("Diagram shows", ["All actions", "Best action only"], horizontal=True, key="mdp_show")
        try:
            st.graphviz_chart(mdp_graph(T, V, policy, show))
        except Exception:
            st.info("The state diagram needs a newer Streamlit; the tables below show the same information.")
        st.caption(f"Stopped after {len(deltas)} iterations. Solid arrows are the best action in each state; "
                   "thicker arrows are more likely transitions.")

    st.markdown("#### Watch the values converge")
    kmax = len(hist) - 1
    k = st.slider("Iteration k", 0, min(kmax, 150), min(5, kmax), key="mdp_k")
    a, b = st.columns([1.2, 1])
    with a:
        show_k = min(kmax, 150)
        vh = pd.DataFrame(np.array(hist[: show_k + 1]), columns=NB_STATES)
        vh["k"] = np.arange(show_k + 1)
        fig = px.line(vh.melt(id_vars="k", var_name="State", value_name="V"), x="k", y="V", color="State",
                      color_discrete_sequence=["#2F5D8A", "#E0A43A", "#5B9279"])
        fig.add_vline(x=k, line_dash="dot", line_color="#1C2330")
        fig.update_layout(height=340, margin=dict(t=10), xaxis_title="Iteration k", yaxis_title="V_k(s)")
        show_chart(fig)
    with b:
        Vk = hist[k]
        rows = []
        for s in range(3):
            qs = [R[s, act] + gamma_nb * float(np.dot(T[s, act], Vk)) for act in range(2)]
            for act in range(2):
                terms = " + ".join(f"{T[s, act, j]:.2f}×{Vk[j]:.2f}" for j in range(3))
                rows.append({"State": NB_STATES[s], "Action": NB_ACTIONS[act],
                             "R + γ Σ P·V_k": f"{R[s, act]:.1f} + {gamma_nb:.2f}·({terms})",
                             "Q": round(qs[act], 3), "Best": "★" if act == int(np.argmax(qs)) else ""})
        st.markdown(f"**Bellman update at k = {k}** (gives V<sub>{k + 1}</sub>)", unsafe_allow_html=True)
        show_df(pd.DataFrame(rows))

    st.markdown("#### Final Q-values and policy")
    qdf = pd.DataFrame(Q, index=NB_STATES, columns=[f"Q(s, {x})" for x in NB_ACTIONS]).round(3)
    qdf["Optimal action"] = [NB_ACTIONS[p] for p in policy]
    qdf["V*(s)"] = V.round(3)
    show_df(qdf, hide_index=False)
    notebook_code('''states = ['s1', 's2', 's3']
actions = ['a1', 'a2']
R = np.array([[5, 10], [2, 3], [8, 1]])
T = np.array([[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]],
              [[0.3, 0.4, 0.3], [0.5, 0.3, 0.2]],
              [[0.4, 0.4, 0.2], [0.2, 0.5, 0.3]]])
gamma = 0.9

def value_iteration(T, R, gamma, iterations=1000):
    V = np.zeros(len(states))
    for i in range(iterations):
        V_new = np.zeros(len(states))
        for s in range(len(states)):
            Q_sa = np.zeros(len(actions))
            for a in range(len(actions)):
                Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
            V_new[s] = np.max(Q_sa)
        V = V_new
    return V''')


# ---------------------------------------------------------------------------
# ADP: approximate dynamic programming
# ---------------------------------------------------------------------------

def sampled_value_iteration(T, R, valid, gamma, n_samples, iterations, alpha_mode, alpha_const, seed, V_exact, pol_exact):
    """Approximate VI: the expectation Σ P(s'|s,a) V(s') is replaced by an average over sampled next states."""
    rng = np.random.default_rng(seed)
    S, A = R.shape
    # sparse sampling tables: only the next states that can actually happen
    K = max(1, int((T > 0).sum(axis=2).max()))
    nz_idx = np.zeros((S, A, K), dtype=int)
    nz_cdf = np.ones((S, A, K))
    for s in range(S):
        for a in range(A):
            nz = np.nonzero(T[s, a] > 0)[0]
            if len(nz):
                nz_idx[s, a, :len(nz)] = nz
                nz_idx[s, a, len(nz):] = nz[-1]
                c = np.cumsum(T[s, a, nz])
                nz_cdf[s, a, :len(nz)] = c / c[-1]
    V = np.zeros(S)
    hist, err, agree = [V.copy()], [], []
    for k in range(1, iterations + 1):
        u = rng.random((S, A, n_samples))
        pos = np.minimum((u[..., None] > nz_cdf[:, :, None, :]).sum(axis=-1), K - 1)
        nxt = np.take_along_axis(nz_idx, pos.reshape(S, A, -1), axis=2)
        est = V[nxt].mean(axis=2)
        Qk = np.where(valid, R + gamma * est, -np.inf)
        target = Qk.max(axis=1)
        if alpha_mode == "Constant":
            alpha = alpha_const
        elif alpha_mode == "Decaying":
            alpha = 1.0 / (1.0 + (1.0 - gamma) * k)
        else:
            alpha = 1.0                                # plain sampled backups
        V = (1 - alpha) * V + alpha * target
        hist.append(V.copy())
        err.append(float(np.max(np.abs(V - V_exact))))
        greedy = np.where(valid, R + gamma * np.einsum("ijk,k->ij", T, V), -np.inf).argmax(axis=1)
        agree.append(float((greedy == pol_exact).mean()))
    return np.array(hist), err, agree


def exact_vi(T, R, valid, gamma, tol=1e-10, max_iter=100_000):
    V = np.zeros(R.shape[0])
    for _ in range(max_iter):
        V_new = np.where(valid, R + gamma * np.einsum("ijk,k->ij", T, V), -np.inf).max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    pol = np.where(valid, R + gamma * np.einsum("ijk,k->ij", T, V), -np.inf).argmax(axis=1)
    return V, pol


def fitted_value_iteration(T, R, valid, gamma, Phi, weights, iterations, V_exact, pol_exact):
    """ADP with a linear value function V(s) ≈ Φ(s)·w, refitted by weighted least squares every iteration."""
    w = np.zeros(Phi.shape[1])
    sw = np.sqrt(np.maximum(weights, 1e-9))
    err, agree, deltas = [], [], []
    V = Phi @ w
    for _ in range(iterations):
        Qk = np.where(valid, R + gamma * np.einsum("ijk,k->ij", T, V), -np.inf)
        y = Qk.max(axis=1)
        w_new, *_ = np.linalg.lstsq(Phi * sw[:, None], y * sw, rcond=None)
        V_new = Phi @ w_new
        deltas.append(float(np.max(np.abs(V_new - V))))
        w, V = w_new, V_new
        err.append(float(np.max(np.abs(V - V_exact))))
        greedy = np.where(valid, R + gamma * np.einsum("ijk,k->ij", T, V), -np.inf).argmax(axis=1)
        agree.append(float((greedy == pol_exact).mean()))
        if not np.all(np.isfinite(V)):
            break
    return w, V, err, agree, deltas


@cache_data(show_spinner="Running approximate dynamic programming…")
def run_sampled_adp(problem_key, _T, _R, _valid, gamma, n_list, iterations, alpha_mode, alpha_const, seed):
    V_exact, pol_exact = exact_vi(_T, _R, _valid, gamma)
    runs = {}
    for n in n_list:
        runs[n] = sampled_value_iteration(_T, _R, _valid, gamma, n, iterations, alpha_mode, alpha_const, seed, V_exact, pol_exact)
    return V_exact, pol_exact, runs


def render_adp_tab(mdp, features, n_bins, gamma_robot, robot_key):
    st.subheader("Approximate Dynamic Programming")
    st.markdown(
        '<div class="note">Exact value iteration needs the full transition model and one value per state. '
        'ADP gives up some accuracy to avoid that: <b>sampled backups</b> estimate the expected next value from a few '
        'simulated next states, and a <b>linear value function</b> stores a handful of weights instead of a table. '
        'Both are compared with the exact answer.</div>', unsafe_allow_html=True)
    method = st.radio("ADP method", ["Sampled backups (simulation-based)", "Linear value function (fitted value iteration)"],
                      horizontal=True, key="adp_method")

    if method.startswith("Sampled"):
        c1, c2, c3 = st.columns(3)
        problem = c1.selectbox("Problem", ["Notebook 3-state MDP", "Robot MDP (sidebar settings)"], key="adp_problem")
        n_samples = c2.select_slider("Samples per backup", options=[1, 2, 5, 10, 25, 50, 100], value=5, key="adp_n")
        iterations = c3.slider("Iterations", 20, 1000, 150, 10, key="adp_iter")
        c4, c5, c6 = st.columns(3)
        alpha_mode = c4.radio("Step size α", ["α = 1 (plain sampled backups)", "Constant α", "Decaying α = 1/(1 + (1−γ)k)"],
                              key="adp_alpha_mode")
        alpha_const = c5.slider("Constant α", 0.05, 1.0, 0.5, 0.05, key="adp_alpha",
                                disabled=not alpha_mode.startswith("Constant"))
        seed = c6.number_input("Random seed", 0, 9999, 7, 1, key="adp_seed")
        mode = "Constant" if alpha_mode.startswith("Constant") else ("Decaying" if alpha_mode.startswith("Decaying") else "Plain")
        if problem.startswith("Notebook"):
            T, R = NB_T, NB_R
            valid = np.ones_like(R, dtype=bool)
            g, labels, key = 0.9, NB_STATES, ("nb",)
        else:
            T, R, valid = mdp["T"], mdp["R"], mdp["valid"]
            g, labels, key = gamma_robot, mdp["states"], robot_key
        n_list = sorted({1, 25, int(n_samples)})
        V_exact, pol_exact, runs = run_sampled_adp(key, T, R, valid, g, tuple(n_list), iterations, mode, alpha_const, int(seed))
        hist, err, agree = runs[int(n_samples)]

        m = st.columns(3)
        m[0].metric("Max error |V − V*| (last 20 iterations)", f"{float(np.mean(err[-20:])):.3f}")
        m[1].metric("Greedy policy = exact policy", f"{agree[-1]:.0%}")
        m[2].metric("Samples used per iteration", f"{len(labels) * R.shape[1] * n_samples:,}")
        a, b = st.columns(2)
        with a:
            if problem.startswith("Notebook"):
                d = pd.DataFrame(hist, columns=labels)
                d["Iteration"] = np.arange(len(hist))
                fig = px.line(d.melt(id_vars="Iteration", var_name="State", value_name="V"), x="Iteration", y="V",
                              color="State", color_discrete_sequence=["#2F5D8A", "#E0A43A", "#5B9279"])
                for i, s in enumerate(labels):
                    fig.add_hline(y=V_exact[i], line_dash="dash", line_color="#9AA3AE",
                                  annotation_text=f"exact {s}", annotation_position="right")
                fig.update_layout(height=360, margin=dict(t=10, r=60), yaxis_title="Approximate V(s)")
                st.markdown(f"**Approximate values with {n_samples} sample(s) per backup** (dashed = exact)")
            else:
                fig = px.scatter(x=V_exact, y=hist[-1], labels={"x": "Exact V*(s)", "y": "Approximate V(s)"},
                                 hover_name=labels)
                lo, hi = float(min(V_exact.min(), hist[-1].min())), float(max(V_exact.max(), hist[-1].max()))
                fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="#9AA3AE"))
                fig.update_traces(marker=dict(color="#2F5D8A", size=8, opacity=0.75))
                fig.update_layout(height=360, margin=dict(t=10))
                st.markdown("**Approximate vs exact value for each state** (dashed line = perfect)")
            show_chart(fig)
        with b:
            rows = []
            for n in n_list:
                for i, e in enumerate(runs[n][1]):
                    rows.append({"Iteration": i + 1, "Max error": max(e, 1e-12), "Samples per backup": str(n)})
            fig = px.line(pd.DataFrame(rows), x="Iteration", y="Max error", color="Samples per backup", log_y=True,
                          color_discrete_sequence=["#C8553D", "#2F5D8A", "#5B9279", "#8C7AA9"])
            fig.update_layout(height=360, margin=dict(t=10))
            st.markdown("**Error against the exact answer**")
            show_chart(fig)
        st.caption("More samples give a less noisy estimate of the expected next value, so the error settles at a lower "
                   "floor (roughly halving each time the samples go up four times). A smaller or decaying α averages the "
                   "noise out but takes longer to get there. With γ close to 1 the values need more iterations to settle.")
    else:
        c1, c2 = st.columns(2)
        iterations = c1.slider("Iterations", 10, 400, 150, 10, key="fvi_iter")
        weighting = c2.radio("Fit weighting", ["By how often the state was visited", "Equal weight per state"],
                             key="fvi_weight")
        T, R, valid = mdp["T"], mdp["R"], mdp["valid"]
        names = BIN_NAMES[n_bins]
        bins = np.array([[names.index(p) for p in s.split(" | ")] for s in mdp["states"]])
        cols, col_names = [np.ones(len(bins))], ["Intercept"]
        for j, f in enumerate(features):
            for bidx in range(bins[:, j].max() + 1):
                cols.append((bins[:, j] == bidx).astype(float))
                col_names.append(f"{f.replace('SD_', '')} = {names[bidx]}")
        Phi = np.column_stack(cols)
        wts = mdp["n_sa"].sum(axis=1) if weighting.startswith("By") else np.ones(len(bins))
        V_exact, pol_exact = exact_vi(T, R, valid, gamma_robot)
        w, V_hat, err, agree, deltas = fitted_value_iteration(T, R, valid, gamma_robot, Phi, wts, iterations, V_exact, pol_exact)

        m = st.columns(4)
        m[0].metric("Numbers stored", f"{Phi.shape[1]} weights", f"vs {len(bins)} table entries", delta_color="off")
        m[1].metric("Max error |V − V*|", f"{err[-1]:.3f}")
        m[2].metric("Mean error", f"{float(np.mean(np.abs(V_hat - V_exact))):.3f}")
        m[3].metric("Greedy policy = exact policy", f"{agree[-1]:.0%}")
        a, b = st.columns(2)
        with a:
            fig = px.scatter(x=V_exact, y=V_hat, size=np.maximum(mdp["n_sa"].sum(axis=1), 1), hover_name=mdp["states"],
                             labels={"x": "Exact V*(s)", "y": "Linear approximation Φ(s)·w"})
            lo, hi = float(min(V_exact.min(), V_hat.min())), float(max(V_exact.max(), V_hat.max()))
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="#9AA3AE"))
            fig.update_traces(marker=dict(color="#2F5D8A", opacity=0.7))
            fig.update_layout(height=380, margin=dict(t=10))
            st.markdown("**Approximate vs exact value** (bubble size = visits in the data)")
            show_chart(fig)
        with b:
            wdf = pd.DataFrame({"Feature": col_names[1:], "Weight": w[1:]})
            fig = px.bar(wdf, x="Weight", y="Feature", orientation="h", color="Weight",
                         color_continuous_scale=["#C8553D", "#F2E8CF", "#2F5D8A"])
            fig.update_layout(height=380, margin=dict(t=10), coloraxis_showscale=False, yaxis_title="")
            st.markdown(f"**What each bin is worth** (intercept = {w[0]:.2f})")
            show_chart(fig)
        d = pd.DataFrame({"Iteration": np.arange(1, len(err) + 1), "Max error vs exact": err,
                          "Policy agreement": agree})
        fig = px.line(d, x="Iteration", y=["Max error vs exact"], color_discrete_sequence=["#C8553D"])
        fig.update_layout(height=260, margin=dict(t=10), yaxis_title="Max error", legend_title="")
        show_chart(fig)
        st.caption("V(s) is modelled as an intercept plus one weight per distance bin, so the value of a state is the "
                   "sum of what each of its readings is worth. It can't capture interactions (a near front wall matters "
                   "more when the left wall is also near), which is where the remaining error comes from.")
    notebook_code('''# "Approximate Dynamic Programming" cell of the notebook
V = np.zeros(len(states))

def value_iteration(T, R, gamma, iterations=1000):
    V = np.zeros(len(states))
    for i in range(iterations):
        V_new = np.zeros(len(states))
        for s in range(len(states)):
            Q_sa = np.zeros(len(actions))
            for a in range(len(actions)):
                Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
            V_new[s] = np.max(Q_sa)
        V = V_new
    return V

# In this app the exact expectation np.dot(T[s][a], V) is replaced by
# np.mean(V[sampled_next_states])  (sampled backups), or V is replaced by
# Phi @ w refitted with least squares each iteration (linear approximation).''')


# ---------------------------------------------------------------------------
# Monte Carlo policy search (notebook classes, with a few extras)
# ---------------------------------------------------------------------------

class GridEnvironment:
    def __init__(self, grid_size=(4, 4), start_state=(0, 0), goal_state=(3, 3), traps=None, max_steps=500):
        self.grid_size = grid_size
        self.start_state = start_state
        self.goal_state = goal_state
        self.current_state = start_state
        self.actions = [0, 1, 2, 3]            # Up, Down, Left, Right
        self.current_pos = start_state
        self.goal = goal_state
        self.traps = list(traps or [])
        self.max_steps = max_steps
        self.t = 0

    def reset(self):
        self.current_state = self.start_state
        self.current_pos = self.start_state
        self.t = 0
        return self.current_state

    def step(self, action):
        x, y = self.current_pos
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.grid_size[1] - 1, y + 1)
        self.current_pos = (x, y)
        self.t += 1
        reward, done, info = -1, False, "running"
        if self.current_pos == self.goal:
            reward, done, info = 10, True, "goal"
        elif self.current_pos in self.traps:
            reward, done, info = -10, True, "trap"
        elif self.t >= self.max_steps:
            done, info = True, "timeout"
        return self.current_pos, reward, done, {"end": info}


class GridMonteCarloPolicySearch:
    """The notebook's MonteCarloPolicySearch for the grid world (renamed so it doesn't clash with the robot version)."""

    def __init__(self, env, policy, gamma=0.99):
        self.env = env
        self.policy = policy
        self.gamma = gamma

    def generate_episode(self):
        episode, state, end = [], self.env.reset(), "running"
        while True:
            action = self.policy(state)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                end = info["end"]
                episode.append((next_state, None, 0))
                break
            state = next_state
        return episode, end

    def evaluate_policy(self, num_episodes=1000):
        returns, lengths, ends = [], [], []
        for _ in range(num_episodes):
            episode, end = self.generate_episode()
            G = 0.0
            for t in reversed(range(len(episode) - 1)):
                G = episode[t][2] + self.gamma * G
            returns.append(G)
            lengths.append(len(episode) - 1)
            ends.append(end)
        return np.array(returns), np.array(lengths), ends


def make_policy(kind, epsilon, goal, rng):
    if kind == "Random (notebook)":
        return lambda state: int(rng.choice([0, 1, 2, 3]))

    def eps_greedy(state):
        if rng.random() < epsilon:
            return int(rng.choice([0, 1, 2, 3]))
        x, y = state
        moves = []
        if goal[0] < x: moves.append(0)
        if goal[0] > x: moves.append(1)
        if goal[1] < y: moves.append(2)
        if goal[1] > y: moves.append(3)
        return int(rng.choice(moves)) if moves else int(rng.choice([0, 1, 2, 3]))
    return eps_greedy


def make_traps(n, count, seed, start, goal):
    rng = np.random.default_rng(seed)
    cells = [(i, j) for i in range(n) for j in range(n) if (i, j) not in (start, goal)]
    count = min(count, len(cells))
    idx = rng.choice(len(cells), size=count, replace=False) if count else []
    return [cells[i] for i in idx]


@cache_data(show_spinner="Simulating episodes…")
def run_monte_carlo(n, n_traps, trap_seed, kind, epsilon, episodes, gamma, max_steps, seed):
    start, goal = (0, 0), (n - 1, n - 1)
    traps = make_traps(n, n_traps, trap_seed, start, goal)
    rng = np.random.default_rng(seed)
    env = GridEnvironment((n, n), start, goal, traps, max_steps)
    mc = GridMonteCarloPolicySearch(env, make_policy(kind, epsilon, goal, rng), gamma)
    returns, lengths, ends = mc.evaluate_policy(episodes)
    visits = np.zeros((n, n))
    sample_eps = []
    rng2 = np.random.default_rng(seed + 1)
    mc2 = GridMonteCarloPolicySearch(GridEnvironment((n, n), start, goal, traps, max_steps),
                                 make_policy(kind, epsilon, goal, rng2), gamma)
    for i in range(min(episodes, 300)):
        ep, end = mc2.generate_episode()
        for (s, _, _) in ep:
            visits[s[0], s[1]] += 1
        if i < 20:
            sample_eps.append(([s for (s, _, _) in ep], end))
    return traps, returns, lengths, ends, visits, sample_eps


def episode_animation(n, traps, path, title):
    goal = (n - 1, n - 1)
    base = np.zeros((n, n))
    for t in traps:
        base[t[0], t[1]] = -1
    base[goal[0], goal[1]] = 1
    path = path[:300]
    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    heat = go.Heatmap(z=base, x=list(range(n)), y=list(range(n)), showscale=False, hoverinfo="skip",
                      colorscale=[[0, "#E7B8AC"], [0.5, "#FFFFFF"], [1, "#BCD6C5"]], zmin=-1, zmax=1, xgap=2, ygap=2)
    fig = go.Figure(
        data=[heat,
              go.Scatter(x=xs[:1], y=ys[:1], mode="lines", line=dict(color="#2F5D8A", width=3), name="path"),
              go.Scatter(x=xs[:1], y=ys[:1], mode="markers", marker=dict(size=22, color="#2F5D8A", line=dict(color="white", width=2)), name="agent")],
        frames=[go.Frame(data=[heat,
                               go.Scatter(x=xs[: i + 1], y=ys[: i + 1], mode="lines", line=dict(color="#2F5D8A", width=3)),
                               go.Scatter(x=[xs[i]], y=[ys[i]], mode="markers", marker=dict(size=22, color="#2F5D8A", line=dict(color="white", width=2)))],
                         name=str(i)) for i in range(len(xs))])
    ann = [dict(x=0, y=0, text="S", showarrow=False, font=dict(size=16, color="#3E7D5E")),
           dict(x=goal[1], y=goal[0], text="G", showarrow=False, font=dict(size=16, color="#3E7D5E"))]
    ann += [dict(x=t[1], y=t[0], text="✕", showarrow=False, font=dict(size=16, color="#C8553D")) for t in traps]
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)), height=520, margin=dict(t=70, b=90, l=10, r=10), showlegend=False,
        annotations=ann,
        xaxis=dict(range=[-0.5, n - 0.5], dtick=1, title="Y", constrain="domain"),
        yaxis=dict(range=[n - 0.5, -0.5], dtick=1, title="X", scaleanchor="x", constrain="domain"),
        updatemenus=[dict(type="buttons", showactive=False, x=1, y=1.02, xanchor="right", yanchor="bottom", direction="left",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=180, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0, len=1, y=-0.12, yanchor="top", pad=dict(t=10), currentvalue=dict(prefix="Step "),
                      steps=[dict(method="animate", label=str(i),
                                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for i in range(len(xs))])])
    return fig


def render_mc_tab():
    st.subheader("Monte Carlo policy search")
    st.markdown(
        '<div class="note">The notebook\'s grid world: start at S, reach G for +10, every step costs −1 and traps (✕) '
        'cost −10 and end the episode. Monte Carlo estimates how good a policy is by playing many episodes and '
        'averaging the discounted return.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    n = c1.slider("Grid size", 4, 10, 4, key="mc_n")
    n_traps = c2.slider("Traps", 0, 10, 0, key="mc_traps")
    trap_seed = c3.number_input("Trap layout seed", 0, 999, 3, key="mc_trapseed")
    episodes = c4.select_slider("Episodes", options=[100, 250, 500, 1000, 2000, 5000], value=1000, key="mc_eps")
    c5, c6, c7, c8 = st.columns(4)
    kind = c5.selectbox("Policy", ["Random (notebook)", "ε-greedy towards the goal"], key="mc_policy")
    epsilon = c6.slider("ε (chance of a random move)", 0.0, 1.0, 0.3, 0.05, key="mc_epsilon",
                        disabled=kind.startswith("Random"))
    gamma_mc = c7.slider("Discount γ", 0.5, 1.0, 0.99, 0.01, key="mc_gamma")
    max_steps = c8.slider("Max steps per episode", 20, 1000, 500, 20, key="mc_max")
    seed = st.number_input("Random seed", 0, 9999, 42, key="mc_seed")

    traps, returns, lengths, ends, visits, sample_eps = run_monte_carlo(
        n, n_traps, int(trap_seed), kind, epsilon, episodes, gamma_mc, max_steps, int(seed))
    se = returns.std(ddof=1) / np.sqrt(len(returns)) if len(returns) > 1 else 0.0
    m = st.columns(4)
    m[0].metric("Estimated value of the policy", f"{returns.mean():.2f}", f"± {1.96 * se:.2f} (95%)", delta_color="off")
    m[1].metric("Average episode length", f"{lengths.mean():.1f} steps")
    m[2].metric("Episodes reaching the goal", f"{ends.count('goal') / len(ends):.0%}")
    m[3].metric("Episodes ending in a trap", f"{ends.count('trap') / len(ends):.0%}")

    a, b = st.columns([1, 1])
    with a:
        pick = st.slider("Sample episode to replay", 1, len(sample_eps), 1, key="mc_pick")
        path, end = sample_eps[pick - 1]
        show_chart(episode_animation(n, traps, path, f"Agent's movement in the grid: {len(path) - 1} steps, ended at {end}"))
    with b:
        running = np.cumsum(returns) / np.arange(1, len(returns) + 1)
        csd = pd.Series(returns).expanding().std().fillna(0).values
        band = 1.96 * csd / np.sqrt(np.arange(1, len(returns) + 1))
        ep = np.arange(1, len(returns) + 1)
        fig = go.Figure([
            go.Scatter(x=np.concatenate([ep, ep[::-1]]), y=np.concatenate([running + band, (running - band)[::-1]]),
                       fill="toself", fillcolor="rgba(47,93,138,0.15)", line=dict(width=0), hoverinfo="skip", name="95% band"),
            go.Scatter(x=ep, y=running, line=dict(color="#2F5D8A", width=2), name="running mean")])
        fig.update_layout(height=250, margin=dict(t=30, b=10), title=dict(text="Estimate settles as episodes are added", font=dict(size=14)),
                          xaxis_title="Episodes", yaxis_title="Mean return", showlegend=False)
        show_chart(fig)
        fig = px.histogram(x=returns, nbins=40, color_discrete_sequence=["#8C7AA9"], labels={"x": "Discounted return"})
        fig.update_layout(height=220, margin=dict(t=30, b=10), title=dict(text="Distribution of returns", font=dict(size=14)),
                          yaxis_title="Episodes")
        show_chart(fig)
    fig = px.imshow(visits, color_continuous_scale="Blues", labels=dict(x="Y", y="X", color="Visits"), text_auto=".0f")
    fig.update_layout(height=360, margin=dict(t=30, b=10), title=dict(text="Where the agent spends its time (first 300 episodes)", font=dict(size=14)))
    show_chart(fig)
    notebook_code('''class MonteCarloPolicySearch:
    def __init__(self, env, policy, gamma=0.99):
        self.env = env
        self.policy = policy
        self.gamma = gamma

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        while True:
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def evaluate_policy(self, num_episodes=1000):
        returns = []
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
            returns.append(G)
        return np.mean(returns)

def random_policy(state): return np.random.choice([0, 1, 2, 3])
env = GridEnvironment()
mcps = MonteCarloPolicySearch(env, random_policy)
print("Policy evaluation:", mcps.evaluate_policy(num_episodes=1000))''')


# ---------------------------------------------------------------------------
# Hooke-Jeeves pattern search
# ---------------------------------------------------------------------------

HJ_FUNCS = {
    "Sphere x² + y² (notebook)": (lambda x: x[0] ** 2 + x[1] ** 2, (-3, 3, -3, 3), [(0, 0)], False),
    "Booth": (lambda x: (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2, (-4, 6, -4, 6), [(1, 3)], False),
    "Matyas": (lambda x: 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1], (-5, 5, -5, 5), [(0, 0)], False),
    "Himmelblau (four minima)": (lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2, (-5, 5, -5, 5),
                                 [(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)], True),
    "Rosenbrock (curved valley)": (lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, (-2, 2, -1, 3), [(1, 1)], True),
}


def hooke_jeeves_traced(func, x0, step_size=0.5, epsilon=1e-6, max_iter=1000, check_pattern=False):
    """The notebook's hooke_jeeves, recording every probe and move so it can be animated."""
    x = np.array(x0, dtype=float)
    n = len(x)
    delta = step_size
    iter_count = 0
    probes, rows, frames = [], [], []
    n_evals = [0]

    def f(v):
        n_evals[0] += 1
        return func(v)

    def explore(x, delta):
        for i in range(n):
            f_val = f(x)
            x[i] += delta
            probes.append(x.copy())
            if f(x) < f_val:
                continue
            x[i] -= 2 * delta
            probes.append(x.copy())
            if f(x) < f_val:
                continue
            x[i] += delta
        return x

    while delta > epsilon and iter_count < max_iter:
        iter_count += 1
        x_old = np.copy(x)
        x = explore(x, delta)
        if np.array_equal(x, x_old):
            move = "no improvement: halve step"
            delta_used = delta
            delta /= 2
        else:
            delta_used = delta
            pattern = x + (x - x_old)
            if check_pattern and not (func(pattern) < func(x)):
                move = "explore moved (pattern rejected)"
            else:
                move = "explore moved + pattern move"
                x = pattern
        rows.append({"Iteration": iter_count, "x": x[0], "y": x[1], "f(x)": float(func(x)), "Step size": delta_used, "Move": move})
        frames.append((x_old.copy(), x.copy(), delta_used, move))
    return x, rows, np.array(probes) if probes else np.zeros((0, 2)), frames, n_evals[0]


def hj_figure(func, rng_box, minima, log_scale, rows, probes, frames, x0):
    x_min, x_max, y_min, y_max = rng_box
    gx = np.linspace(x_min, x_max, 160)
    gy = np.linspace(y_min, y_max, 160)
    XX, YY = np.meshgrid(gx, gy)
    ZZ = func([XX, YY])
    Zp = np.log10(ZZ - ZZ.min() + 1) if log_scale else ZZ
    path_x = [x0[0]] + [r["x"] for r in rows]
    path_y = [x0[1]] + [r["y"] for r in rows]
    contour = go.Contour(x=gx, y=gy, z=Zp, colorscale=[[0, "#2F5D8A"], [0.5, "#F2E8CF"], [1, "#C8553D"]], showscale=False,
                         contours=dict(showlines=True), line=dict(width=0.5, color="rgba(28,35,48,0.25)"), hoverinfo="skip", opacity=0.85)
    mins = go.Scatter(x=[m[0] for m in minima], y=[m[1] for m in minima], mode="markers",
                      marker=dict(symbol="star", size=14, color="#FFFFFF", line=dict(color="#1C2330", width=1.5)), name="true minimum")
    probe_tr = go.Scatter(x=probes[:, 0] if len(probes) else [], y=probes[:, 1] if len(probes) else [], mode="markers",
                          marker=dict(size=4, color="rgba(28,35,48,0.35)"), name="probes")

    def cross(center, d):
        cx, cy = center
        return go.Scatter(x=[cx - d, cx + d, None, cx, cx], y=[cy, cy, None, cy - d, cy + d], mode="lines",
                          line=dict(color="#E0A43A", width=3), name="probe cross")

    n_fr = len(frames)
    keep = list(range(n_fr)) if n_fr <= 120 else sorted(set(np.linspace(0, n_fr - 1, 120).astype(int).tolist()))
    fr = []
    for i in keep:
        start_pt, end_pt, d, move = frames[i]
        fr.append(go.Frame(name=str(i), data=[
            contour, probe_tr, mins,
            go.Scatter(x=path_x[: i + 2], y=path_y[: i + 2], mode="lines+markers", line=dict(color="#1C2330", width=2),
                       marker=dict(size=6, color="#1C2330"), name="path"),
            cross(start_pt, d),
            go.Scatter(x=[end_pt[0]], y=[end_pt[1]], mode="markers", marker=dict(size=14, color="#E0A43A", line=dict(color="#1C2330", width=2)), name="current")],
            layout=go.Layout(title=dict(text=f"Iteration {i + 1}: {move}, step {d:.3g}", font=dict(size=14)))))
    first = [contour, probe_tr, mins,
             go.Scatter(x=path_x, y=path_y, mode="lines+markers", line=dict(color="#1C2330", width=2), marker=dict(size=6, color="#1C2330"), name="path"),
             cross(frames[-1][1] if frames else x0, frames[-1][2] if frames else 0),
             go.Scatter(x=[path_x[-1]], y=[path_y[-1]], mode="markers", marker=dict(size=14, color="#E0A43A", line=dict(color="#1C2330", width=2)), name="current")]
    fig = go.Figure(data=first, frames=fr)
    fig.update_layout(
        height=620, margin=dict(t=70, b=90, l=10, r=10), showlegend=False,
        title=dict(text="Full search path (press Play to replay it)", font=dict(size=14)),
        xaxis=dict(range=[x_min, x_max], title="x", constrain="domain"),
        yaxis=dict(range=[y_min, y_max], title="y", scaleanchor="x", constrain="domain"),
        updatemenus=[dict(type="buttons", showactive=False, x=1, y=1.02, xanchor="right", yanchor="bottom", direction="left",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=250, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0, len=1, y=-0.1, yanchor="top", pad=dict(t=10), currentvalue=dict(prefix="Iteration "),
                      steps=[dict(method="animate", label=str(i + 1),
                                  args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for i in keep])])
    return fig


def render_hj_tab():
    st.subheader("Hooke-Jeeves pattern search")
    st.markdown(
        '<div class="note">A derivative-free optimiser. From the current point it probes each coordinate by ± step '
        '(the orange cross). If a probe improves f, it takes a bigger <b>pattern move</b> in that direction; if nothing '
        'improves, it halves the step. It stops when the step is smaller than ε.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    fname = c1.selectbox("Function to minimise", list(HJ_FUNCS.keys()), key="hj_func")
    func, box, minima, log_scale = HJ_FUNCS[fname]
    x0 = c2.slider("Start x", float(box[0]), float(box[1]), float(np.clip(1.0 if "Sphere" in fname else box[0] + 0.3 * (box[1] - box[0]), box[0], box[1])), 0.1, key=f"hj_x0_{fname}")
    y0 = c3.slider("Start y", float(box[2]), float(box[3]), float(np.clip(1.0 if "Sphere" in fname else box[2] + 0.8 * (box[3] - box[2]), box[2], box[3])), 0.1, key=f"hj_y0_{fname}")
    c4, c5, c6, c7 = st.columns(4)
    step_size = c4.slider("Initial step size", 0.05, 2.0, 0.5, 0.05, key="hj_step")
    epsilon = float(c5.select_slider("Stop when step < ε", options=["1e-02", "1e-03", "1e-04", "1e-06", "1e-08"],
                                     value="1e-06", key="hj_eps"))
    max_iter = c6.slider("Max iterations", 10, 3000, 1000, 10, key="hj_maxiter")
    check_pattern = c7.checkbox("Only keep a pattern move if it improves f", value=True, key="hj_check",
                                help="Untick to run exactly as the notebook, which always keeps the pattern move. "
                                     "From some starting points that makes it bounce back and forth without ever "
                                     "shrinking its step.")

    x_best, rows, probes, frames, n_evals = hooke_jeeves_traced(func, [x0, y0], step_size, epsilon, max_iter, check_pattern)
    if not check_pattern and len(rows) >= max_iter and rows[-1]["Step size"] > epsilon:
        st.warning("The notebook version hit the iteration limit without shrinking its step: the pattern move keeps "
                   "overshooting and the search bounces back and forth. Tick 'Only keep a pattern move if it improves f' "
                   "to fix it.")
    f_best = float(func(x_best))
    dist = min(float(np.hypot(x_best[0] - m[0], x_best[1] - m[1])) for m in minima)
    m = st.columns(4)
    m[0].metric("Optimized parameters", f"({x_best[0]:.4f}, {x_best[1]:.4f})")
    m[1].metric("f at the result", f"{f_best:.2e}")
    m[2].metric("Iterations / function calls", f"{len(rows)} / {n_evals}")
    m[3].metric("Distance to nearest true minimum", f"{dist:.2e}")
    if not rows:
        st.info("The step size is already below ε, so the search stops before its first move.")
        return
    show_chart(hj_figure(func, box, minima, log_scale, rows, probes, frames, [x0, y0]))
    d = pd.DataFrame(rows)
    a, b = st.columns(2)
    with a:
        fig = px.line(d, x="Iteration", y=np.maximum(d["f(x)"] - min(0.0, d["f(x)"].min()), 1e-16), log_y=True)
        fig.update_traces(line_color="#2F5D8A")
        fig.update_layout(height=280, margin=dict(t=30), yaxis_title="f(x)", title=dict(text="Objective value", font=dict(size=14)))
        show_chart(fig)
    with b:
        fig = px.line(d, x="Iteration", y="Step size", log_y=True)
        fig.update_traces(line_color="#E0A43A")
        fig.add_hline(y=epsilon, line_dash="dash", line_color="#C8553D", annotation_text="ε")
        fig.update_layout(height=280, margin=dict(t=30), title=dict(text="Step size (halves when stuck)", font=dict(size=14)))
        show_chart(fig)
    with st.expander("Iteration table"):
        show_df(d.round(6), height=320)
    notebook_code('''def hooke_jeeves(func, x0, step_size=0.5, epsilon=1e-6, max_iter=1000):
    x = np.array(x0)
    n = len(x)
    delta = step_size
    iter_count = 0

    def explore(x, delta):
        for i in range(n):
            f_val = func(x)
            x[i] += delta
            if func(x) < f_val:
                continue
            x[i] -= 2 * delta
            if func(x) < f_val:
                continue
            x[i] += delta
        return x

    while delta > epsilon and iter_count < max_iter:
        iter_count += 1
        x_old = np.copy(x)
        x = explore(x, delta)
        if np.array_equal(x, x_old):
            delta /= 2
        else:
            x = x + (x - x_old)
    return x

def objective_function(x):
    return x[0]**2 + x[1]**2
result = hooke_jeeves(objective_function, [1.0, 1.0])
print("Optimized parameters:", result)''')


# ===========================================================================
# Payloads for the two interactive (HTML/JavaScript) components
# ===========================================================================

ACTION_COLORS = {
    "Move-Forward": "#2F5D8A",
    "Slight-Right-Turn": "#E0A43A",
    "Sharp-Right-Turn": "#C8553D",
    "Slight-Left-Turn": "#5B9279",
}
ARC_COLORS = {"SD_front": "#2F5D8A", "SD_left": "#5B9279", "SD_right": "#E0A43A", "SD_back": "#8C7AA9"}


def _num(x, nd=4):
    """JSON-safe float (None for inf / nan)."""
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _bins_of(states, n_bins):
    names = BIN_NAMES[n_bins]
    return {s: [names.index(p) for p in s.split(" | ")] for s in states}


def sim_payload(mdp, vi, features, edges, n_bins, rewards):
    states = mdp["states"]
    payload = {
        "actions": ACTIONS,
        "colors": ACTION_COLORS,
        "arc_colors": ARC_COLORS,
        "sd_cols": SD_COLS,
        "features": list(features),
        "bin_names": BIN_NAMES[n_bins],
        "edges": {f: [float(e) for e in edges[f]] for f in features},
        "states": states,
        "bins": _bins_of(states, n_bins),
        "opt": {s: ACTIONS[int(vi["policy"][i])] for i, s in enumerate(states)},
        "beh": {s: ACTIONS[int(mdp["n_sa"][i].argmax())] if mdp["n_sa"][i].sum() > 0 else ACTIONS[0]
                for i, s in enumerate(states)},
        "valid": {s: [ACTIONS[a] for a in range(len(ACTIONS)) if mdp["valid"][i, a]] for i, s in enumerate(states)},
        "Q": {s: [_num(q, 3) for q in vi["Q"][i]] for i, s in enumerate(states)},
        "V": {s: _num(vi["V"][i], 3) for i, s in enumerate(states)},
        "rewards": {k: float(v) for k, v in rewards.items()},
    }
    return json.dumps(payload)


def vi_payload(mdp, vi, features, edges, n_bins, gamma, tolerance):
    states = mdp["states"]
    S, A = mdp["R"].shape
    V_hist = vi["V_hist"]
    n = len(V_hist) - 1
    if n <= 150:
        ks = list(range(n + 1))
    else:
        ks = set(range(0, 31)) | set(np.unique(np.round(np.geomspace(1, n, 120)).astype(int)).tolist()) | {n}
        ks = sorted(k for k in ks if 0 <= k <= n)
    hist = vi["history"]
    frames = [{"k": int(k), "V": [round(float(v), 4) for v in V_hist[k]],
               "delta": float(hist[k - 1]) if k >= 1 else None} for k in ks]
    allv = np.concatenate([np.asarray(V_hist[k]) for k in ks])
    conv_idx = np.unique(np.linspace(1, n, min(n, 400)).astype(int)) if n >= 1 else []
    T = []
    for s in range(S):
        row = []
        for a in range(A):
            if mdp["valid"][s, a]:
                nz = np.nonzero(mdp["T"][s, a])[0]
                row.append([[int(j), round(float(mdp["T"][s, a, j]), 4)] for j in nz])
            else:
                row.append([])
        T.append(row)
    visits = mdp["n_sa"].sum(axis=1)
    payload = {
        "actions": ACTIONS,
        "colors": ACTION_COLORS,
        "features": list(features),
        "bin_names": BIN_NAMES[n_bins],
        "nbins": [len(edges[f]) + 1 for f in features],
        "states": states,
        "bins": _bins_of(states, n_bins),
        "T": T,
        "R": [[round(float(r), 4) for r in row] for row in mdp["R"]],
        "valid": [[bool(v) for v in row] for row in mdp["valid"]],
        "gamma": float(gamma),
        "tol": float(tolerance),
        "frames": frames,
        "conv": [[int(k), float(hist[k - 1])] for k in conv_idx],
        "iterations": int(n),
        "converged": bool(vi["converged"]),
        "vmin": float(allv.min()),
        "vmax": float(allv.max()),
        "visits": [int(v) for v in visits],
        "vfinal": [round(float(v), 4) for v in vi["V"]],
        "default_state": int(np.argmax(visits)),
    }
    return json.dumps(payload)


def render_component(template, payload_json, height):
    html = template.replace("__PAYLOAD__", payload_json)
    components.html(html, height=height, scrolling=True)


# ===========================================================================
# Streamlit UI
# ===========================================================================

st.set_page_config(page_title="Wall-Following Robot MDP", page_icon="🤖", layout="wide")

HERE = Path(__file__).parent
DEFAULT_DATA = find_default_data()
OUTPUT_CSV = HERE / "optimal_value_function.csv"

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1280px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .note {background:#EEF3F8; border-left:4px solid #2F5D8A; padding:0.7rem 1rem;
           border-radius:4px; font-size:0.95rem; margin-bottom:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@cache_data(show_spinner=False)
def get_data(file_bytes, default_path):
    if file_bytes is not None:
        return load_data(io.BytesIO(file_bytes))
    return load_data(default_path)


@cache_data(show_spinner="Building the MDP and running value iteration…")
def get_results(df, features, n_bins, gamma, tolerance, rewards, costs):
    return run_pipeline(df, list(features), n_bins, gamma, tolerance, dict(rewards), dict(costs))


@cache_data(show_spinner="Simulating episodes…")
def get_mc(_mdp, _vi, key, gamma, episodes, horizon):
    return compare_policies(_mdp, _vi, gamma, episodes, horizon)


# ---------------------------------------------------------------------------
# Sidebar: the MDP inputs listed on the slide
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("MDP inputs")
    st.caption("Every change here rebuilds the MDP, re-runs value iteration and "
               "updates the robot simulator and the animation.")
    upload = st.file_uploader("Sensor file (optional)", type=["csv", "data"],
                              help="Raw 24-sensor file. Uses sensor_readings_24.csv next to app.py if empty.")
    if upload is None and DEFAULT_DATA is None:
        st.error("sensor_readings_24.csv not found. Put it next to app.py (or in a data/ folder), "
                 "or upload it above.")
        st.stop()
    try:
        df = get_data(upload.getvalue() if upload else None, str(DEFAULT_DATA) if DEFAULT_DATA else None)
    except Exception as e:
        st.error(f"Could not read the sensor file: {e}")
        st.stop()
    if len(df) < 10:
        st.error("The sensor file has too few valid rows.")
        st.stop()

    st.subheader("States")
    features = st.multiselect("Distances used for the state", SD_COLS, default=SD_COLS,
                              help="Front + Left alone matches the 2-sensor version of the dataset.")
    if len(features) < 1:
        st.warning("Pick at least one distance.")
        st.stop()
    n_bins = st.select_slider("Bins per distance", options=[2, 3, 4, 5], value=3)
    st.caption(f"Up to {n_bins ** len(features)} possible states.")

    st.subheader("Value iteration")
    gamma = st.slider("Discount factor γ", 0.50, 0.99, 0.90, 0.01)
    tolerance = float(st.select_slider("Convergence threshold θ",
                                       options=["1e-02", "1e-03", "1e-04", "1e-05", "1e-06", "1e-07", "1e-08"],
                                       value="1e-06"))

    with st.expander("Rewards"):
        d = DEFAULT_REWARDS
        left_low, left_high = st.slider("Target left-wall band (m)", 0.3, 2.0,
                                        (d["left_low"], d["left_high"]), 0.05)
        front_danger = st.slider("Front danger distance (m)", 0.3, 1.5, d["front_danger"], 0.05)
        side_danger = st.slider("Left danger distance (m)", 0.3, 1.0, d["side_danger"], 0.05)
        follow_reward = st.number_input("Reward: inside wall band", 0.0, 10.0, d["follow_reward"], 0.5)
        lost_penalty = st.number_input("Penalty: wall lost", 0.0, 10.0, d["lost_penalty"], 0.5)
        crash_penalty = st.number_input("Penalty: too close", 0.0, 50.0, d["crash_penalty"], 1.0)

    with st.expander("Action costs"):
        costs = {a: st.number_input(a, 0.0, 2.0, DEFAULT_COSTS[a], 0.05, key=f"c_{a}") for a in ACTIONS}

rewards = dict(left_low=left_low, left_high=left_high, front_danger=front_danger, side_danger=side_danger,
               follow_reward=follow_reward, lost_penalty=lost_penalty, crash_penalty=crash_penalty)
features = [f for f in SD_COLS if f in features]      # fixed order

dfs, edges, mdp, vi, table = get_results(df, tuple(features), n_bins, gamma, tolerance,
                                         tuple(rewards.items()), tuple(costs.items()))
try:
    table.to_csv(OUTPUT_CSV, index=False)                # the slide's required output file
    saved_ok = True
except Exception:
    saved_ok = False

policy_map = dict(zip(table["State"], table["Optimal_Action"]))
dfs["Optimal"] = dfs["State"].map(policy_map)
row_agree = float((dfs["Optimal"] == dfs["Class"]).mean())


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Wall-Following Robot Navigation")
st.write("A Markov Decision Process learned from the SCITOS-G5 robot's ultrasound log, solved with "
         "Value Iteration, then used to drive a robot around a room you can redraw.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Readings", f"{len(df):,}")
c2.metric("States in the MDP", len(mdp["states"]))
c3.metric("Iterations to converge", vi["iterations"], "converged" if vi["converged"] else "not converged",
          delta_color="normal" if vi["converged"] else "inverse")
c4.metric("Agrees with logged moves", f"{row_agree:.0%}")

tabs = st.tabs(["🤖 Robot simulator", "🧮 Value iteration", "📐 MDP", "🔁 ADP", "🎲 Monte Carlo",
                "🧭 Hooke-Jeeves", "📊 Robot data & results"])
with tabs[6]:
    sub = st.tabs(["The task", "Sensors", "MDP model", "Results & CSV", "Policy check"])


# ---------------------------------------------------------------------------
# Tab: robot simulator
# ---------------------------------------------------------------------------

with tabs[0]:
    st.markdown(
        '<div class="note">Put down a <b>start</b> (🟢) and an <b>end point</b> (🏁), then press Play. '
        '<b>Wall-following</b> uses the optimal policy learned from the sensor data: it reads its four 60° sensor '
        'arcs, turns them into an MDP state and looks up the best move. <b>Go to end point</b> solves a second MDP '
        'on the grid itself with value iteration and follows the arrows to the flag. Draw walls, add slip or noise, '
        'or change the sidebar and watch the behaviour change.</div>',
        unsafe_allow_html=True)
    try:
        render_component(SIM_HTML, sim_payload(mdp, vi, features, edges, n_bins, rewards), height=930)
    except Exception as e:
        st.error(f"The simulator could not be drawn: {e}")
    with st.expander("How the simulator maps onto the MDP"):
        st.markdown("""
- **Room:** 16 × 12 squares of 0.4 m each (6.4 m × 4.8 m), close to the size of the room in the original experiment.
- **Sensing:** 13 rays are cast in each 60° arc; the shortest one is the simplified distance, exactly how SD_front, SD_left, SD_right and SD_back are defined.
- **State:** each distance is put into the same bins used to build the MDP (see the MDP model tab).
- **Action:** read from the optimal policy table. Forward moves one step, a slight turn rotates a little and moves, a sharp right turn rotates more and barely moves.
- **Unseen readings:** if a combination of bins never appears in the data, the nearest known state is used and counted.
- **Action slip** executes a random action some of the time, which is the uncertainty an MDP is designed to handle.
- **Reward** shown in the panel uses the same reward settings as the sidebar.
- **End point:** the run stops when the robot gets within 0.36 m of the flag. The data-learned wall-follower only reaches end points near the wall it follows, and in tight spaces (the maze, the cluttered room) it can get stuck in a loop. That is an honest limitation of a policy learned from one room's log with coarse bins.
- **Go to end point (grid MDP):** states are the free squares, actions are the 8 compass moves, each step costs 1 (1.41 diagonally), bumping a wall costs 5, γ = 0.99. With action slip the robot veers 45° left or right with that probability, and the MDP plans for it. Value iteration is re-run in the browser every time a wall, the flag or the slip changes.
""")


# ---------------------------------------------------------------------------
# Tab: value iteration animation
# ---------------------------------------------------------------------------

with tabs[1]:
    st.markdown(
        '<div class="note">Each square is one state (a combination of distance bins). Press Play to watch the '
        'values spread from the rewarding states to the rest, the arrows settle into the optimal policy and Δ '
        'fall below the threshold. Click any square to see its Bellman update with real numbers.</div>',
        unsafe_allow_html=True)
    try:
        render_component(VI_HTML, vi_payload(mdp, vi, features, edges, n_bins, gamma, tolerance), height=1000)
    except Exception as e:
        st.error(f"The animation could not be drawn: {e}")
    with st.expander("Python code used for value iteration (same update as the Day 5 notebook)"):
        st.code('''def value_iteration(T, R, gamma=0.9, tolerance=1e-6, max_iterations=10_000, valid=None):
    S, A = R.shape
    V = np.zeros(S)
    history = []
    for i in range(max_iterations):
        V_new = np.zeros(S)
        for s in range(S):
            Q_sa = np.full(A, -np.inf)
            for a in range(A):
                if valid[s, a]:                                  # only actions seen in the data
                    Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
            V_new[s] = np.max(Q_sa)
        delta = np.max(np.abs(V_new - V))
        history.append(delta)
        V = V_new
        if delta < tolerance:                                    # convergence threshold
            break
    return V, history''', language="python")


# ---------------------------------------------------------------------------
# Tabs: Day 5 notebook programs
# ---------------------------------------------------------------------------

robot_key = (tuple(features), n_bins, gamma, tolerance, tuple(rewards.items()), tuple(costs.items()))
program_tabs = [(tabs[2], "MDP", lambda: render_mdp_tab()),
                (tabs[3], "ADP", lambda: render_adp_tab(mdp, features, n_bins, gamma, robot_key)),
                (tabs[4], "Monte Carlo", lambda: render_mc_tab()),
                (tabs[5], "Hooke-Jeeves", lambda: render_hj_tab())]
for container, name, fn in program_tabs:
    with container:
        try:
            fn()
        except Exception as e:                      # keep the rest of the app alive whatever happens
            st.error(f"The {name} section hit a problem: {e}")

# ---------------------------------------------------------------------------
# Tab: the task
# ---------------------------------------------------------------------------

with sub[0]:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("How the exercise maps onto an MDP")
        st.markdown(f"""
| MDP piece | In this app |
|---|---|
| **States** | Each simplified distance ({", ".join(features)}) is cut into {n_bins} bins ({", ".join(BIN_NAMES[n_bins])}). A state is one combination, e.g. `{mdp["states"][0]}`. |
| **Actions** | The 4 movement classes: {", ".join(ACTIONS)}. |
| **Transition probabilities** | Counted from the log: the robot records 9 readings a second, so reading *t* taken with action *a* leads to reading *t+1*. P(s'∣s,a) = count(s,a,s') / count(s,a). |
| **Rewards** | + when the left wall sits in the target band, − when the wall is lost, a large − when something is too close in front or on the left, minus a small cost per turn. |
| **Discount γ** | {gamma} |
| **Convergence threshold θ** | {tolerance:.0e}: stop when no state value changes by more than this. |
| **Output** | `optimal_value_function.csv` with V*(s), the best action and all Q-values. |
""")
    with right:
        st.subheader("Class balance in the log")
        counts = df["Class"].value_counts().reindex(ACTIONS).fillna(0)
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                     color_discrete_map=ACTION_COLORS, labels={"x": "", "y": "Readings"})
        fig.update_layout(showlegend=False, height=330, margin=dict(t=10, b=10))
        show_chart(fig)
        st.subheader("Data preview")
        show_df(df[SD_COLS + ["Class"]].head(8))


# ---------------------------------------------------------------------------
# Tab: sensors
# ---------------------------------------------------------------------------

with sub[1]:
    st.subheader("What the real robot saw")
    st.caption("24 ultrasound sensors ring the robot. The four simplified distances are the minimum "
               "reading inside each coloured group (grouping chosen to match the official 4-sensor file).")
    a, b = st.columns([1, 1])
    with a:
        i = st.slider("Reading number", 0, len(df) - 1, min(1200, len(df) - 1))
        row = df.iloc[i]
        sensor_group = {s: g for g, ss in SD_GROUPS.items() for s in ss}
        theta = [SENSOR_ANGLES[k] for k in range(1, 25)]
        r = [row[f"US{k}"] for k in range(1, 25)]
        colors = [ARC_COLORS.get(sensor_group.get(k), "#B8BEC6") for k in range(1, 25)]
        fig = go.Figure(go.Barpolar(r=r, theta=theta, width=[13] * 24, marker_color=colors,
                                    text=[f"US{k}" for k in range(1, 25)],
                                    hovertemplate="%{text}: %{r:.2f} m<extra></extra>"))
        fig.update_layout(height=430, margin=dict(t=20, b=20),
                          polar=dict(radialaxis=dict(range=[0, 5.1], ticksuffix=" m"),
                                     angularaxis=dict(direction="clockwise", rotation=90)))
        show_chart(fig)
    with b:
        st.markdown(f"**Logged action:** {row['Class']}  \n"
                    f"**State:** `{dfs.iloc[i]['State']}`  \n"
                    f"**Optimal action:** {dfs.iloc[i]['Optimal']}")
        show_df(pd.DataFrame({
            "Distance": SD_COLS,
            "Sensors": [", ".join(f"US{s}" for s in SD_GROUPS[c]) for c in SD_COLS],
            "Value (m)": [round(float(row[c]), 3) for c in SD_COLS],
        }))
        feat = st.selectbox("Distance by action", SD_COLS, index=1)
        fig = px.box(df, x="Class", y=feat, color="Class", color_discrete_map=ACTION_COLORS,
                     category_orders={"Class": ACTIONS})
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10), xaxis_title="")
        show_chart(fig)


# ---------------------------------------------------------------------------
# Tab: MDP model
# ---------------------------------------------------------------------------

with sub[2]:
    st.subheader("States")
    edge_rows = []
    for f in features:
        e = list(edges[f])
        names = BIN_NAMES[n_bins][: len(e) + 1]
        bounds = ["min"] + [f"{x:.3f}" for x in e] + ["max"]
        edge_rows.append({"Distance": f, **{nm: f"{bounds[j]} – {bounds[j + 1]} m" for j, nm in enumerate(names)}})
    show_df(pd.DataFrame(edge_rows))
    st.caption("Bins are quantiles, so each holds about the same share of readings. "
               f"{len(mdp['states'])} of {n_bins ** len(features)} possible combinations appear in the data.")

    st.subheader("Transition probabilities")
    act = st.selectbox("Action", ACTIONS)
    a_i = ACTIONS.index(act)
    seen = np.where(mdp["n_sa"][:, a_i] > 0)[0]
    if len(seen) == 0:
        st.info("This action never appears in the data.")
    else:
        top = seen[np.argsort(-mdp["n_sa"][seen, a_i])][:25]
        cols_ = np.where(mdp["T"][top, a_i].sum(axis=0) > 0)[0]
        fig = px.imshow(mdp["T"][np.ix_(top, [a_i], cols_)][:, 0, :],
                        x=[mdp["states"][k] for k in cols_], y=[mdp["states"][k] for k in top],
                        color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="Next state s'", y="State s", color="P(s'|s,a)"))
        fig.update_layout(height=620, margin=dict(t=10))
        show_chart(fig)
        st.caption(f"Rows: the {len(top)} states where '{act}' was used most. Most probability sits on the "
                   "diagonal because the robot samples 9 times a second and rarely changes state in one step.")

    st.subheader("Rewards R(s, a)")
    R_show = pd.DataFrame(np.where(mdp["n_sa"] > 0, mdp["R"], np.nan), index=mdp["states"], columns=ACTIONS)
    try:
        styled = R_show.style.format("{:.2f}", na_rep="not seen").background_gradient(cmap="RdYlGn", axis=None)
        show_df(styled, height=380, hide_index=False)
    except Exception:
        show_df(R_show.round(2), height=380, hide_index=False)


# ---------------------------------------------------------------------------
# Tab: results
# ---------------------------------------------------------------------------

with sub[3]:
    a, b = st.columns([1, 1])
    with a:
        st.subheader("Convergence")
        hist = pd.DataFrame({"Iteration": np.arange(1, len(vi["history"]) + 1),
                             "Largest value change": np.maximum(vi["history"], 1e-16)})
        fig = px.line(hist, x="Iteration", y="Largest value change", log_y=True)
        fig.add_hline(y=tolerance, line_dash="dash", line_color="#C8553D",
                      annotation_text="threshold", annotation_position="top right")
        fig.update_layout(height=360, margin=dict(t=10))
        show_chart(fig)
    with b:
        st.subheader("Optimal value by state")
        fig = px.bar(table, x="Optimal_Value", y="State", color="Optimal_Action", orientation="h",
                     color_discrete_map=ACTION_COLORS, category_orders={"Optimal_Action": ACTIONS})
        fig.update_layout(height=360, margin=dict(t=10), yaxis=dict(showticklabels=False, title="States"),
                          legend=dict(orientation="h", y=-0.2, title=""))
        show_chart(fig)

    st.subheader("optimal_value_function.csv")
    show_df(table, height=420)
    st.download_button("Download optimal_value_function.csv", table.to_csv(index=False).encode(),
                       "optimal_value_function.csv", "text/csv")
    if saved_ok:
        st.caption(f"Also saved next to app.py as {OUTPUT_CSV.name}.")


# ---------------------------------------------------------------------------
# Tab: policy check
# ---------------------------------------------------------------------------

with sub[4]:
    st.subheader("Does the optimal policy match what the robot actually did?")
    state_agree = float((table["Optimal_Action"] == table["Data_Most_Common_Action"]).mean())
    m1, m2 = st.columns(2)
    m1.metric("Readings where optimal = logged action", f"{row_agree:.1%}")
    m2.metric("States where optimal = most common logged action", f"{state_agree:.1%}")

    cm = pd.crosstab(dfs["Class"], dfs["Optimal"]).reindex(index=ACTIONS, columns=ACTIONS, fill_value=0)
    fig = px.imshow(cm.values, x=ACTIONS, y=ACTIONS, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Optimal action", y="Logged action", color="Readings"))
    fig.update_layout(height=420, margin=dict(t=10))
    show_chart(fig)
    st.caption("The logged actions came from the robot's own controller, not from our reward, so full "
               "agreement isn't expected. Where they differ, the MDP thinks another move pays off more "
               "under the rewards set in the sidebar.")

    st.subheader("Monte Carlo policy evaluation")
    st.caption("Adapted from the notebook's MonteCarloPolicySearch: episodes are simulated on the learned "
               "MDP and the discounted return is averaged.")
    x1, x2 = st.columns(2)
    episodes = x1.slider("Episodes per policy", 100, 2000, 500, 100)
    horizon = x2.slider("Steps per episode", 10, 200, 50, 10)
    key = (tuple(features), n_bins, gamma, tolerance, tuple(rewards.items()), tuple(costs.items()))
    mc = get_mc(mdp, vi, key, gamma, episodes, horizon)
    mc_df = pd.DataFrame([{"Policy": k, "Mean return": float(v.mean()), "Std": float(v.std())} for k, v in mc.items()])
    fig = px.bar(mc_df, x="Policy", y="Mean return", error_y="Std", color="Policy",
                 color_discrete_sequence=["#2F5D8A", "#8C7AA9", "#B8BEC6"])
    fig.update_layout(showlegend=False, height=360, margin=dict(t=10), xaxis_title="")
    show_chart(fig)
