# QD-CTF Search Simulator

Interactive visualization of Quality-Diversity (QD) driven exploration with fitness sharing for CTF agent trajectory search.

## Overview

This simulator demonstrates how MAP-Elites with fitness sharing can outperform greedy fitness-only search when navigating deceptive fitness landscapes — modeled here as a CTF flag-capture problem.

### Two Search Modes

- **QD + Fitness Sharing** — Uses a MAP-Elites archive with fitness sharing to maintain population diversity. Balances exploitation (mutating from the archive best) with exploration (selecting parents via shared fitness). Avoids getting trapped in local optima by encouraging behavioral diversity across the grid.
- **Raw Fitness Only** — Greedy selection based purely on fitness. Tends to converge prematurely on deceptive ridges and often fails to reach the flag.

### Key Concepts

- **Behavior Space Archive**: A 2D grid (configurable from 20×20 to 100×100) where each cell stores the highest-fitness individual found at that behavioral coordinate. The axes represent exploration breadth and exploitation depth.
- **Fitness Landscape**: A procedurally generated landscape with a global optimum at the FLAG position, a gentle gradient, and 2–4 deceptive ridges that act as local optima traps.
- **Fitness Sharing (σ)**: Controls the niche radius. Individuals in crowded regions get penalized, pushing the search toward unexplored areas.
- **Exploit/Explore Ratio**: Controls what fraction of offspring are generated from the archive best (tight mutations) vs. sharing-selected parents (wider mutations).

## Running

```bash
cd playground
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## Controls

| Control | Description |
|---|---|
| **Step** | Advance one generation |
| **Run / Pause** | Auto-step at ~5 generations/sec |
| **Reset** | Regenerate the landscape and restart |
| **Selection Mode** | Toggle between QD+Sharing and Raw Fitness |
| **Sharing Radius σ** | Adjust niche radius (0.05–0.60) |
| **Exploit/Explore** | Balance between greedy exploitation and diverse exploration |
| **Grid Size** | Archive resolution (20, 40, 60, or 100) |

## Benchmark

The built-in benchmark runs 100 independent trials (max 200 steps each) under the current settings and reports success rate, average/median/min/max steps to flag capture.
