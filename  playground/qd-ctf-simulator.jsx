import { useState, useCallback, useRef, useEffect } from "react";

// ─── Constants ───
const GRID_DEFAULT = 20;
const SHARING_RADIUS_DEFAULT = 0.25;
const POP_SIZE = 6;
const MUTATION_SPREAD = 0.12;

// ─── Color Palette ───
const C = {
  bg: "#0a0e17",
  panel: "#111827",
  border: "#1e293b",
  accent: "#22d3ee",
  accentDim: "#0e7490",
  warm: "#f59e0b",
  danger: "#ef4444",
  success: "#10b981",
  text: "#e2e8f0",
  textDim: "#64748b",
  grid: "#1e293b",
  cellEmpty: "#0f172a",
  cellFill: (f) => {
    const h = 170 + f * 30;
    const s = 60 + f * 30;
    const l = 15 + f * 35;
    return `hsl(${h}, ${s}%, ${l}%)`;
  },
};

// ─── Utility ───
const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const dist = (a, b) => Math.sqrt((a.bx - b.bx) ** 2 + (a.by - b.by) ** 2);
const rng = () => Math.random();
const gaussian = () => {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

// ─── Simulated CTF fitness landscape ───
const FLAG_POS = { x: 0.85, y: 0.88 };

function generateRidges() {
  const count = 2 + Math.floor(rng() * 3); // 2-4 ridges
  return Array.from({ length: count }, () => ({
    cx: 0.15 + rng() * 0.6,
    cy: 0.2 + rng() * 0.5,
    amp: 0.25 + rng() * 0.25,
    sigX: 0.06 + rng() * 0.15,
    sigY: 0.005 + rng() * 0.008,
    angle: rng() * Math.PI, // rotation
  }));
}

function computeFitness(bx, by, ridges) {
  const d = Math.sqrt((bx - FLAG_POS.x) ** 2 + (by - FLAG_POS.y) ** 2);
  const flagComponent = Math.exp(-d * d / 0.08);
  const gradComponent = 0.12 * (bx + by) * 0.5;
  let ridgeComponent = 0;
  for (const r of ridges) {
    const dx = bx - r.cx, dy = by - r.cy;
    const cos = Math.cos(r.angle), sin = Math.sin(r.angle);
    const rx = cos * dx + sin * dy;
    const ry = -sin * dx + cos * dy;
    ridgeComponent += r.amp * Math.exp(-(rx * rx) / r.sigX - (ry * ry) / r.sigY);
  }
  // Flag base fitness at flag position ≈ 1.0 + gradient
  const flagMax = 1.0 + 0.12 * (FLAG_POS.x + FLAG_POS.y) * 0.5;
  // Cap total so ridges alone can't exceed ~70% of flag max
  const f = flagComponent + gradComponent + Math.min(ridgeComponent, flagMax * 0.7);
  return clamp(f / flagMax, 0, 1);
}

function createIndividual(bx, by, ridges) {
  const fitness = computeFitness(bx, by, ridges);
  return { bx, by, fitness, id: Math.random().toString(36).slice(2, 8) };
}

function mutate(parent, ridges, spread = MUTATION_SPREAD) {
  const bx = clamp(parent.bx + gaussian() * spread);
  const by = clamp(parent.by + gaussian() * spread);
  return createIndividual(bx, by, ridges);
}

// ─── Archive (MAP-Elites style grid) ───
function createArchive() {
  const cells = {};
  return cells;
}

function cellKey(bx, by, grid) {
  const cx = Math.min(grid - 1, Math.floor(bx * grid));
  const cy = Math.min(grid - 1, Math.floor(by * grid));
  return `${cx},${cy}`;
}

function insertIntoArchive(archive, ind, grid) {
  const key = cellKey(ind.bx, ind.by, grid);
  if (!archive[key] || ind.fitness > archive[key].fitness) {
    archive[key] = { ...ind };
    return true;
  }
  return false;
}

// ─── Selection ───
function computeSharedFitness(population, sigma) {
  return population.map((ind) => {
    const nicheCount = population.reduce((sum, other) => {
      const d = dist(ind, other);
      return sum + (d < sigma ? 1 - d / sigma : 0);
    }, 0);
    return { ...ind, sharedFitness: ind.fitness / Math.max(nicheCount, 0.01) };
  });
}

function selectParent(popWithShared) {
  const total = popWithShared.reduce((s, p) => s + p.sharedFitness, 0);
  if (total === 0) return popWithShared[Math.floor(rng() * popWithShared.length)];
  let r = rng() * total;
  for (const p of popWithShared) {
    r -= p.sharedFitness;
    if (r <= 0) return p;
  }
  return popWithShared[popWithShared.length - 1];
}

function selectParentNoSharing(population) {
  const total = population.reduce((s, p) => s + p.fitness, 0);
  if (total === 0) return population[Math.floor(rng() * population.length)];
  let r = rng() * total;
  for (const p of population) {
    r -= p.fitness;
    if (r <= 0) return p;
  }
  return population[population.length - 1];
}

// ─── Components ───

function BehaviorSpaceGrid({ archive, population, highlight, sigma, showSharing, ridges, grid }) {
  const svgW = 640, svgH = 640;
  const pad = 32;
  const gW = svgW - pad * 2, gH = svgH - pad * 2;
  const cellW = gW / grid, cellH = gH / grid;
  const [hover, setHover] = useState(null);

  const archiveEntries = Object.entries(archive);

  return (
    <svg width={svgW} height={svgH} style={{ display: "block" }} onMouseLeave={() => setHover(null)}>
      <defs>
        <radialGradient id="flagGlow">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Background */}
      <rect width={svgW} height={svgH} fill={C.bg} rx="8" />

      {/* Grid cells - fitness landscape heatmap + archive overlay */}
      {Array.from({ length: grid }, (_, cx) =>
        Array.from({ length: grid }, (_, cy) => {
          const key = `${cx},${cy}`;
          const entry = archive[key];
          const x = pad + cx * cellW;
          const y = pad + (grid - 1 - cy) * cellH;
          // Compute landscape fitness at cell center
          const centerBx = (cx + 0.5) / grid;
          const centerBy = (cy + 0.5) / grid;
          const landscapeF = computeFitness(centerBx, centerBy, ridges);
          // Clean: transparent dark → blue glow
          const alpha = 0.08 + landscapeF * 0.7;
          const landscapeColor = `rgba(34, 140, 230, ${alpha.toFixed(3)})`;
          const isHovered = hover && hover.cx === cx && hover.cy === cy;
          return (
            <g key={key} onMouseEnter={() => setHover({ cx, cy, x, y, fitness: landscapeF, archived: !!entry, archiveFit: entry?.fitness })}>
              <rect
                x={x + 0.5}
                y={y + 0.5}
                width={cellW - 1}
                height={cellH - 1}
                fill={landscapeColor}
                stroke={isHovered ? "rgba(255,255,255,0.6)" : C.grid}
                strokeWidth={isHovered ? "1.5" : "0.3"}
                rx="1"
              />
              {entry && (
                <rect
                  x={x + 1}
                  y={y + 1}
                  width={cellW - 2}
                  height={cellH - 2}
                  fill="rgba(34,211,238,0.25)"
                  stroke="rgba(34,211,238,0.6)"
                  strokeWidth="0.8"
                  rx="1"
                />
              )}
            </g>
          );
        })
      )}

      {/* Hover tooltip */}
      {hover && (() => {
        const tx = hover.x + cellW / 2;
        const ty = hover.y;
        const flipX = tx > svgW - 120;
        const flipY = ty < 60;
        const ax = flipX ? tx - 8 : tx + 8;
        const ay = flipY ? ty + cellH + 8 : ty - 8;
        return (
          <g style={{ pointerEvents: "none" }}>
            <rect
              x={flipX ? ax - 108 : ax}
              y={flipY ? ay : ay - 44}
              width={108}
              height={hover.archived ? 44 : 28}
              rx="4"
              fill="rgba(15,23,42,0.92)"
              stroke="rgba(100,180,255,0.3)"
              strokeWidth="1"
            />
            <text
              x={flipX ? ax - 54 : ax + 54}
              y={flipY ? ay + 16 : ay - 28}
              textAnchor="middle"
              fill="#e2e8f0"
              fontSize="10"
              fontFamily="monospace"
            >
              fitness: {hover.fitness.toFixed(3)}
            </text>
            {hover.archived && (
              <text
                x={flipX ? ax - 54 : ax + 54}
                y={flipY ? ay + 32 : ay - 12}
                textAnchor="middle"
                fill={C.accent}
                fontSize="10"
                fontFamily="monospace"
              >
                archived: {hover.archiveFit.toFixed(3)}
              </text>
            )}
          </g>
        );
      })()}

      {/* Flag position */}
      <g>
        <circle
          cx={pad + FLAG_POS.x * gW}
          cy={pad + (1 - FLAG_POS.y) * gH}
          r={30}
          fill="url(#flagGlow)"
        />
        <text
          x={pad + FLAG_POS.x * gW}
          y={pad + (1 - FLAG_POS.y) * gH + 4}
          textAnchor="middle"
          fill={C.accent}
          fontSize="11"
          fontFamily="monospace"
          opacity={0.7}
        >
          FLAG
        </text>
      </g>

      {/* Sharing radius visualization */}
      {showSharing && highlight && (
        <circle
          cx={pad + highlight.bx * gW}
          cy={pad + (1 - highlight.by) * gH}
          r={sigma * gW}
          fill="none"
          stroke={C.warm}
          strokeWidth="1.5"
          strokeDasharray="4 3"
          opacity="0.6"
        />
      )}

      {/* Population dots */}
      {population.map((ind, i) => {
        const cx = pad + ind.bx * gW;
        const cy = pad + (1 - ind.by) * gH;
        const isHL = highlight && ind.id === highlight.id;
        return (
          <g key={ind.id}>
            {isHL && (
              <circle cx={cx} cy={cy} r="10" fill={C.warm} opacity="0.2" />
            )}
            <circle
              cx={cx}
              cy={cy}
              r={isHL ? 5 : 3.5}
              fill={isHL ? C.warm : C.accent}
              stroke={isHL ? C.warm : "none"}
              strokeWidth="2"
              opacity={isHL ? 1 : 0.9}
            />
          </g>
        );
      })}

      {/* Axes */}
      <text x={svgW / 2} y={svgH - 6} textAnchor="middle" fill={C.textDim} fontSize="10" fontFamily="monospace">
        Exploration Breadth →
      </text>
      <text
        x={10}
        y={svgH / 2}
        textAnchor="middle"
        fill={C.textDim}
        fontSize="10"
        fontFamily="monospace"
        transform={`rotate(-90, 10, ${svgH / 2})`}
      >
        Exploitation Depth →
      </text>
    </svg>
  );
}

function StatsPanel({ archive, population, gen, mode, grid }) {
  const filled = Object.keys(archive).length;
  const coverage = ((filled / (grid * grid)) * 100).toFixed(1);
  const avgFit = population.length
    ? (population.reduce((s, p) => s + p.fitness, 0) / population.length).toFixed(3)
    : "—";
  const bestFit = population.length
    ? Math.max(...population.map((p) => p.fitness)).toFixed(3)
    : "—";
  const archiveBest = Object.values(archive).length
    ? Math.max(...Object.values(archive).map((a) => a.fitness)).toFixed(3)
    : "—";

  const stats = [
    { label: "Generation", value: gen },
    { label: "Archive Cells", value: `${filled} / ${grid * grid}` },
    { label: "Coverage", value: `${coverage}%` },
    { label: "Pop Avg Fitness", value: avgFit },
    { label: "Pop Best Fitness", value: bestFit },
    { label: "Archive Best", value: archiveBest },
    { label: "Mode", value: mode === "sharing" ? "QD + Sharing" : "No Sharing" },
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 16px" }}>
      {stats.map((s) => (
        <div key={s.label} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: `1px solid ${C.border}` }}>
          <span style={{ color: C.textDim, fontSize: 11, fontFamily: "monospace" }}>{s.label}</span>
          <span style={{ color: C.text, fontSize: 12, fontFamily: "monospace", fontWeight: 600 }}>{s.value}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main App ───
export default function QDSimulator() {
  const [gridSize, setGridSize] = useState(GRID_DEFAULT);

  const [state, setState] = useState(() => {
    const ridges = generateRidges();
    const pop = Array.from({ length: POP_SIZE }, () =>
      createIndividual(0.1 + rng() * 0.15, 0.1 + rng() * 0.15, ridges)
    );
    const archive = createArchive();
    pop.forEach((ind) => insertIntoArchive(archive, ind, GRID_DEFAULT));
    return { population: pop, archive, gen: 0, ridges, flagFound: false, log: ["Initialized population near origin."] };
  });

  const [mode, setMode] = useState("sharing");
  const [sigma, setSigma] = useState(SHARING_RADIUS_DEFAULT);
  const [highlight, setHighlight] = useState(null);
  const [running, setRunning] = useState(false);
  const [exploitRatio, setExploitRatio] = useState(0.3);
  const speed = 200;
  const runRef = useRef(false);

  const step = useCallback(() => {
    setState((prev) => {
      const { population, archive, gen, ridges, flagFound } = prev;
      if (flagFound) return prev;

      const newArchive = { ...archive };
      const newLog = [];
      const flagCell = cellKey(FLAG_POS.x, FLAG_POS.y, gridSize);

      if (mode === "sharing") {
        // ─── MAP-Elites + exploit/explore balance ───
        const archiveInds = Object.values(newArchive);
        if (archiveInds.length === 0) return prev;

        const withShared = computeSharedFitness(archiveInds, sigma);

        // Find archive best for exploitation
        const archiveBest = archiveInds.reduce((a, b) => a.fitness > b.fitness ? a : b);

        // Split children: some exploit (from best), some explore (from sharing-selected)
        const numChildren = POP_SIZE * 2;
        const numExploit = Math.round(numChildren * exploitRatio);
        const numExplore = numChildren - numExploit;

        const children = [];
        const parents = [];

        // Exploitation: mutate from archive best with tighter spread
        for (let i = 0; i < numExploit; i++) {
          parents.push(archiveBest);
          children.push(mutate(archiveBest, ridges, MUTATION_SPREAD * 0.5));
        }

        // Exploration: mutate from sharing-selected archive parents
        for (let i = 0; i < numExplore; i++) {
          const parent = selectParent(withShared);
          parents.push(parent);
          children.push(mutate(parent, ridges));
        }

        let inserted = 0;
        children.forEach((c) => {
          if (insertIntoArchive(newArchive, c, gridSize)) inserted++;
        });

        // Check if flag cell is now occupied
        const found = !!newArchive[flagCell];

        // Population = a snapshot of current archive elites for display
        const newPop = Object.values(newArchive)
          .sort((a, b) => b.fitness - a.fitness)
          .slice(0, POP_SIZE)
          .map((ind) => {
            const match = withShared.find((ws) => ws.id === ind.id);
            return match ? { ...ind, sharedFitness: match.sharedFitness } : ind;
          });

        if (found && !prev.flagFound) {
          newLog.push(`🚩 FLAG CAPTURED at gen ${gen + 1}!`);
        }
        newLog.push(`Gen ${gen + 1}: ${numExploit} exploit + ${numExplore} explore → ${inserted} new cells [best=${archiveBest.fitness.toFixed(3)}]`);

        return {
          population: newPop,
          archive: newArchive,
          gen: gen + 1,
          ridges,
          flagFound: found,
          log: [...newLog, ...prev.log].slice(0, 50),
          _highlight: archiveBest,
        };
      } else {
        // ─── Raw fitness: greedy selection from population ───
        const parent = selectParentNoSharing(population);
        const children = Array.from({ length: POP_SIZE }, () => mutate(parent, ridges));
        children.forEach((c) => insertIntoArchive(newArchive, c, gridSize));
        const combined = [...population, ...children]
          .sort((a, b) => b.fitness - a.fitness)
          .slice(0, POP_SIZE);

        const found = !!newArchive[flagCell];

        if (found && !prev.flagFound) {
          newLog.push(`🚩 FLAG CAPTURED at gen ${gen + 1}!`);
        }
        newLog.push(`Gen ${gen + 1}: Parent at (${parent.bx.toFixed(2)}, ${parent.by.toFixed(2)}) [raw_f=${parent.fitness.toFixed(3)}]`);

        return {
          population: combined,
          archive: newArchive,
          gen: gen + 1,
          ridges,
          flagFound: found,
          log: [...newLog, ...prev.log].slice(0, 50),
          _highlight: parent,
        };
      }
    });
  }, [mode, sigma, gridSize, exploitRatio]);

  useEffect(() => {
    if (state._highlight) setHighlight(state._highlight);
  }, [state._highlight]);

  useEffect(() => {
    if (state.flagFound) setRunning(false);
  }, [state.flagFound]);

  useEffect(() => {
    runRef.current = running;
    if (!running) return;
    const interval = setInterval(() => {
      if (runRef.current) step();
      else clearInterval(interval);
    }, speed);
    return () => clearInterval(interval);
  }, [running, speed, step]);

  const reset = () => {
    setRunning(false);
    setHighlight(null);
    const ridges = generateRidges();
    const pop = Array.from({ length: POP_SIZE }, () =>
      createIndividual(0.1 + rng() * 0.15, 0.1 + rng() * 0.15, ridges)
    );
    const archive = createArchive();
    pop.forEach((ind) => insertIntoArchive(archive, ind, gridSize));
    setState({ population: pop, archive, gen: 0, ridges, flagFound: false, log: ["Reset. New landscape generated."] });
  };

  const clear = () => {
    setRunning(false);
    setHighlight(null);
    const ridges = state.ridges;
    const pop = Array.from({ length: POP_SIZE }, () =>
      createIndividual(0.1 + rng() * 0.15, 0.1 + rng() * 0.15, ridges)
    );
    const archive = createArchive();
    pop.forEach((ind) => insertIntoArchive(archive, ind, gridSize));
    setState({ population: pop, archive, gen: 0, ridges, flagFound: false, log: ["Cleared. Same landscape, fresh start."] });
  };

  const [bench, setBench] = useState(null); // { running, progress, results }

  const runBenchmark = useCallback(() => {
    setRunning(false);
    setBench({ running: true, progress: 0, results: null });

    const MAX_STEPS = 200;
    const NUM_RUNS = 100;

    // Run async to not block UI
    let runIdx = 0;
    const stepsArr = [];
    const foundArr = [];

    const runOne = () => {
      const ridges = generateRidges();
      const flagCell = cellKey(FLAG_POS.x, FLAG_POS.y, gridSize);
      let found = false;
      let steps = MAX_STEPS;

      if (mode === "sharing") {
        // QD mode
        let archive = createArchive();
        const initPop = Array.from({ length: POP_SIZE }, () =>
          createIndividual(0.1 + rng() * 0.15, 0.1 + rng() * 0.15, ridges)
        );
        initPop.forEach((ind) => insertIntoArchive(archive, ind, gridSize));

        for (let g = 0; g < MAX_STEPS; g++) {
          const archiveInds = Object.values(archive);
          if (archiveInds.length === 0) break;
          const withShared = computeSharedFitness(archiveInds, sigma);
          const archiveBest = archiveInds.reduce((a, b) => a.fitness > b.fitness ? a : b);
          const numChildren = POP_SIZE * 2;
          const nExploit = Math.round(numChildren * exploitRatio);
          const nExplore = numChildren - nExploit;
          for (let i = 0; i < nExploit; i++) {
            insertIntoArchive(archive, mutate(archiveBest, ridges, MUTATION_SPREAD * 0.5), gridSize);
          }
          for (let i = 0; i < nExplore; i++) {
            const parent = selectParent(withShared);
            insertIntoArchive(archive, mutate(parent, ridges), gridSize);
          }
          if (archive[flagCell]) { steps = g + 1; found = true; break; }
        }
      } else {
        // Raw fitness mode
        let pop = Array.from({ length: POP_SIZE }, () =>
          createIndividual(0.1 + rng() * 0.15, 0.1 + rng() * 0.15, ridges)
        );
        let archive = createArchive();
        pop.forEach((ind) => insertIntoArchive(archive, ind, gridSize));

        for (let g = 0; g < MAX_STEPS; g++) {
          const parent = selectParentNoSharing(pop);
          const children = Array.from({ length: POP_SIZE }, () => mutate(parent, ridges));
          children.forEach((c) => insertIntoArchive(archive, c, gridSize));
          pop = [...pop, ...children].sort((a, b) => b.fitness - a.fitness).slice(0, POP_SIZE);
          if (archive[flagCell]) { steps = g + 1; found = true; break; }
        }
      }

      stepsArr.push(steps);
      foundArr.push(found);
      runIdx++;
      setBench((prev) => ({ ...prev, progress: runIdx }));

      if (runIdx < NUM_RUNS) {
        setTimeout(runOne, 0);
      } else {
        const numFound = foundArr.filter(Boolean).length;
        const avgSteps = (stepsArr.reduce((a, b) => a + b, 0) / NUM_RUNS).toFixed(1);
        const minSteps = Math.min(...stepsArr);
        const maxSteps = Math.max(...stepsArr);
        const median = [...stepsArr].sort((a, b) => a - b)[Math.floor(NUM_RUNS / 2)];
        setBench({
          running: false,
          progress: NUM_RUNS,
          results: { numFound, avgSteps, minSteps, maxSteps, median, total: NUM_RUNS },
        });
      }
    };

    setTimeout(runOne, 0);
  }, [mode, sigma, gridSize, exploitRatio]);

  const btnStyle = (active) => ({
    padding: "6px 14px",
    fontSize: 12,
    fontFamily: "monospace",
    fontWeight: 600,
    border: `1px solid ${active ? C.accent : C.border}`,
    borderRadius: 4,
    background: active ? C.accentDim + "33" : C.panel,
    color: active ? C.accent : C.textDim,
    cursor: "pointer",
    transition: "all 0.15s",
  });

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.text, fontFamily: "'JetBrains Mono', 'Fira Code', monospace", padding: 20 }}>
      {/* Header */}
      <div style={{ marginBottom: 16, borderBottom: `1px solid ${C.border}`, paddingBottom: 12 }}>
        <h1 style={{ margin: 0, fontSize: 18, color: C.accent, fontWeight: 700, letterSpacing: "0.05em" }}>
          QD-CTF SEARCH SIMULATOR
        </h1>
        <p style={{ margin: "4px 0 0", fontSize: 11, color: C.textDim }}>
          Quality-Diversity driven exploration with fitness sharing for CTF agent trajectory search
        </p>
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        {/* Left: Behavior Space */}
        <div>
          <div style={{ fontSize: 11, color: C.textDim, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.1em" }}>
            Behavior Space Archive
          </div>
          <BehaviorSpaceGrid
            archive={state.archive}
            population={state.population}
            highlight={highlight}
            sigma={sigma}
            showSharing={mode === "sharing"}
            ridges={state.ridges}
            grid={gridSize}
          />
        </div>

        {/* Right: Controls + Stats */}
        <div style={{ flex: 1, minWidth: 280, display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Mode Toggle */}
          <div>
            <div style={{ fontSize: 11, color: C.textDim, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.1em" }}>
              Selection Mode
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button style={btnStyle(mode === "sharing")} onClick={() => setMode("sharing")}>
                QD + Fitness Sharing
              </button>
              <button style={btnStyle(mode === "none")} onClick={() => setMode("none")}>
                Raw Fitness Only
              </button>
            </div>
          </div>

          {/* Sigma slider */}
          {mode === "sharing" && (
            <div>
              <div style={{ fontSize: 11, color: C.textDim, marginBottom: 4 }}>
                Sharing Radius σ = {sigma.toFixed(2)}
              </div>
              <input
                type="range"
                min="0.05"
                max="0.6"
                step="0.01"
                value={sigma}
                onChange={(e) => setSigma(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: C.accent }}
              />
            </div>
          )}

          {/* Exploit/Explore ratio */}
          {mode === "sharing" && (
            <div>
              <div style={{ fontSize: 11, color: C.textDim, marginBottom: 4 }}>
                Exploit / Explore = {Math.round(exploitRatio * 100)}% / {Math.round((1 - exploitRatio) * 100)}%
              </div>
              <input
                type="range"
                min="0"
                max="0.8"
                step="0.05"
                value={exploitRatio}
                onChange={(e) => setExploitRatio(parseFloat(e.target.value))}
                style={{ width: "100%", accentColor: C.warm }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: C.textDim, marginTop: 2 }}>
                <span>← pure explore</span>
                <span>exploit →</span>
              </div>
            </div>
          )}

          {/* Grid Size */}
          <div>
            <div style={{ fontSize: 11, color: C.textDim, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.1em" }}>
              Grid Size: {gridSize}×{gridSize}
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {[20, 40, 60, 100].map((g) => (
                <button
                  key={g}
                  style={btnStyle(gridSize === g)}
                  onClick={() => {
                    setGridSize(g);
                    setRunning(false);
                    setHighlight(null);
                    const ridges = generateRidges();
                    const pop = Array.from({ length: POP_SIZE }, () =>
                      createIndividual(0.1 + rng() * 0.15, 0.1 + rng() * 0.15, ridges)
                    );
                    const archive = createArchive();
                    pop.forEach((ind) => insertIntoArchive(archive, ind, g));
                    setState({ population: pop, archive, gen: 0, ridges, flagFound: false, log: [`Grid set to ${g}×${g}. New landscape generated.`] });
                  }}
                >
                  {g}
                </button>
              ))}
            </div>
          </div>

          {/* Control buttons */}
          <div style={{ display: "flex", gap: 8 }}>
            <button
              style={btnStyle(false)}
              onClick={step}
            >
              Step
            </button>
            <button
              style={btnStyle(running)}
              onClick={() => setRunning(!running)}
            >
              {running ? "⏸ Pause" : "▶ Run"}
            </button>
            <button
              style={{ ...btnStyle(false), borderColor: C.danger + "66", color: C.danger }}
              onClick={reset}
            >
              Reset
            </button>
            <button
              style={{ ...btnStyle(false), borderColor: C.warm + "66", color: C.warm }}
              onClick={clear}
            >
              Clear
            </button>
          </div>

          {/* Stats */}
          <div style={{ background: C.panel, borderRadius: 6, padding: 12, border: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 11, color: C.textDim, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.1em" }}>
              Statistics
            </div>
            <StatsPanel archive={state.archive} population={state.population} gen={state.gen} mode={mode} grid={gridSize} />
          </div>

          {/* Benchmark */}
          <div style={{ background: C.panel, borderRadius: 6, padding: 12, border: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 11, color: C.textDim, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.1em" }}>
              Benchmark (100 runs, max 200 steps)
            </div>
            <button
              style={{ ...btnStyle(false), borderColor: bench?.running ? C.textDim : C.warm + "88", color: bench?.running ? C.textDim : C.warm, width: "100%", marginBottom: 8 }}
              onClick={runBenchmark}
              disabled={bench?.running}
            >
              {bench?.running ? `Running... ${bench.progress}/100` : `Run Benchmark (${mode === "sharing" ? "QD + Sharing" : "Raw Fitness"})`}
            </button>
            {bench?.running && (
              <div style={{ height: 4, background: C.cellEmpty, borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: `${bench.progress}%`, height: "100%", background: C.warm, borderRadius: 2, transition: "width 0.1s" }} />
              </div>
            )}
            {bench?.results && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px", marginTop: 4 }}>
                {[
                  { label: "Success Rate", value: `${bench.results.numFound} / ${bench.results.total}` },
                  { label: "Avg Steps", value: bench.results.avgSteps },
                  { label: "Median Steps", value: bench.results.median },
                  { label: "Min Steps", value: bench.results.minSteps },
                  { label: "Max Steps", value: bench.results.maxSteps },
                ].map((s) => (
                  <div key={s.label} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: `1px solid ${C.border}` }}>
                    <span style={{ color: C.textDim, fontSize: 10, fontFamily: "monospace" }}>{s.label}</span>
                    <span style={{ color: C.warm, fontSize: 11, fontFamily: "monospace", fontWeight: 600 }}>{s.value}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
