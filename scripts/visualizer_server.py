"""Serve an interactive three-body endpoint visualizer.

The visualizer compares a finite-time endpoint prediction against an
independent numerical playback trajectory.  It is meant for exploration, not as
a substitute for the proof package.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from three_body_symmetry.dynamics import (  # noqa: E402
    center_of_mass,
    energy,
    integrate,
    linear_momentum,
    pack_state,
)
from three_body_symmetry.general_solution import evaluate_unrestricted_solution  # noqa: E402
from three_body_symmetry.series import construct_taylor_solution, integrate_reference  # noqa: E402


DEFAULT_PRESET = {
    "masses": [1.0, 1.0, 1.0],
    "positions": [
        [1.0, 0.0],
        [-0.5, 0.8660254037844386],
        [-0.5, -0.8660254037844386],
    ],
    "velocities": [
        [0.0, 0.7598356856515925],
        [-0.6580370064762462, -0.37991784282579627],
        [0.6580370064762462, -0.37991784282579627],
    ],
    "target_time": 2.0,
    "samples": 500,
    "prediction_method": "taylor",
    "order": 24,
}


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Three-Body Endpoint Visualizer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101319;
      --panel: #181d25;
      --line: #303744;
      --text: #eff4fb;
      --muted: #a8b3c4;
      --accent: #45d0a2;
      --warn: #f2b84b;
      --bad: #ff6b6b;
      --blue: #5aa9ff;
      --magenta: #d986ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow: auto;
    }
    section {
      display: grid;
      grid-template-rows: auto minmax(360px, 1fr) auto;
      min-width: 0;
    }
    h1 {
      margin: 0 0 16px;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }
    h2 {
      margin: 18px 0 8px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0;
    }
    label {
      display: block;
      margin: 10px 0 5px;
      color: var(--muted);
      font-size: 13px;
    }
    input, select, textarea, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f131a;
      color: var(--text);
      font: inherit;
    }
    input, select {
      height: 36px;
      padding: 0 10px;
    }
    textarea {
      min-height: 90px;
      padding: 9px 10px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.35;
    }
    button {
      height: 38px;
      padding: 0 12px;
      cursor: pointer;
      background: var(--accent);
      color: #06120e;
      border-color: transparent;
      font-weight: 700;
    }
    button.secondary {
      background: #202733;
      color: var(--text);
      border-color: var(--line);
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      padding: 12px 16px;
      background: #121720;
    }
    .toolbar button {
      width: auto;
      min-width: 96px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 1px;
      background: var(--line);
      border-top: 1px solid var(--line);
    }
    .stat {
      background: #121720;
      padding: 12px 14px;
      min-height: 68px;
    }
    .stat div:first-child {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .stat div:last-child {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 14px;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      background: #090c11;
    }
    .status {
      margin-top: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      white-space: pre-wrap;
      font-size: 12px;
      min-height: 44px;
    }
    .legend {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }
    .dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 5px;
      vertical-align: -1px;
    }
    .ok { color: var(--accent); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      section { min-height: 70vh; }
      .stats { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
<main>
  <aside>
    <h1>Three-Body Endpoint Visualizer</h1>
    <label>Masses</label>
    <input id="masses" value="1,1,1">
    <label>Positions JSON</label>
    <textarea id="positions"></textarea>
    <label>Velocities JSON</label>
    <textarea id="velocities"></textarea>
    <div class="grid2">
      <div>
        <label>Target time</label>
        <input id="targetTime" type="number" step="0.01" value="2">
      </div>
      <div>
        <label>Samples</label>
        <input id="samples" type="number" min="2" max="2000" step="1" value="500">
      </div>
    </div>
    <div class="grid2">
      <div>
        <label>Predictor</label>
        <select id="method">
          <option value="taylor">Taylor atlas local</option>
          <option value="validated_atlas">Validated atlas</option>
          <option value="compactified_sundman">Compactified Sundman</option>
          <option value="reference">Reference endpoint</option>
        </select>
      </div>
      <div>
        <label>Order</label>
        <input id="order" type="number" min="2" max="64" step="1" value="24">
      </div>
    </div>
    <h2>Commands</h2>
    <div class="grid2">
      <button id="solveBtn">Solve</button>
      <button class="secondary" id="presetBtn">Reset Preset</button>
    </div>
    <button class="secondary" id="figureBtn" style="margin-top:10px;">Figure-Eight Preset</button>
    <div id="status" class="status">Ready.</div>
  </aside>
  <section>
    <div class="toolbar">
      <div class="legend">
        <span><span class="dot" style="background:#45d0a2"></span>body 1</span>
        <span><span class="dot" style="background:#5aa9ff"></span>body 2</span>
        <span><span class="dot" style="background:#d986ff"></span>body 3</span>
        <span>ring = predicted endpoint</span>
      </div>
      <button class="secondary" id="playBtn">Play</button>
    </div>
    <canvas id="scene"></canvas>
    <div class="stats">
      <div class="stat"><div>Max position error</div><div id="posErr">-</div></div>
      <div class="stat"><div>Max velocity error</div><div id="velErr">-</div></div>
      <div class="stat"><div>Prediction status</div><div id="predStatus">-</div></div>
      <div class="stat"><div>Energy drift</div><div id="energyDrift">-</div></div>
    </div>
  </section>
</main>
<script>
const defaultPreset = __DEFAULT_PRESET__;
const figureEightPreset = {
  masses: [1, 1, 1],
  positions: [[0.97000436, -0.24308753], [-0.97000436, 0.24308753], [0, 0]],
  velocities: [[0.466203685, 0.43236573], [0.466203685, 0.43236573], [-0.93240737, -0.86473146]],
  target_time: 6.32591398,
  samples: 900,
  prediction_method: "reference",
  order: 24
};
const colors = ["#45d0a2", "#5aa9ff", "#d986ff"];
let result = null;
let frame = 0;
let playing = false;
let lastTick = 0;

function setPreset(preset) {
  document.getElementById("masses").value = preset.masses.join(",");
  document.getElementById("positions").value = JSON.stringify(preset.positions, null, 2);
  document.getElementById("velocities").value = JSON.stringify(preset.velocities, null, 2);
  document.getElementById("targetTime").value = preset.target_time;
  document.getElementById("samples").value = preset.samples;
  document.getElementById("method").value = preset.prediction_method;
  document.getElementById("order").value = preset.order;
}

function payload() {
  return {
    masses: document.getElementById("masses").value.split(",").map(Number),
    positions: JSON.parse(document.getElementById("positions").value),
    velocities: JSON.parse(document.getElementById("velocities").value),
    target_time: Number(document.getElementById("targetTime").value),
    samples: Number(document.getElementById("samples").value),
    prediction_method: document.getElementById("method").value,
    order: Number(document.getElementById("order").value)
  };
}

async function solve() {
  const status = document.getElementById("status");
  const requestPayload = payload();
  status.textContent = "Solving...";
  document.getElementById("solveBtn").disabled = true;
  try {
    const response = await fetch("/api/solve", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(requestPayload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed");
    result = data;
    frame = 0;
    playing = false;
    updateStats();
    draw();
    status.textContent = [
      `method: ${data.prediction.method}`,
      `prediction: ${data.prediction.status}`,
      `proof_certified: ${data.prediction.proof_certified}`,
      `chart_count: ${data.prediction.chart_count ?? "-"}`,
      `elapsed_ms: ${data.prediction.elapsed_ms}`
    ].join("\n");
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      try {
        result = solveInBrowser(requestPayload, error.message);
        frame = 0;
        playing = false;
        updateStats();
        draw();
        status.textContent = [
          "method: browser RK4 fallback",
          "prediction: fallback",
          "server: unreachable",
          "restart: python3 scripts/visualizer_server.py --port 8765"
        ].join("\n");
      } catch (fallbackError) {
        status.textContent = `Server is not reachable and browser fallback failed: ${fallbackError.message}`;
      }
    } else {
      status.textContent = error.message;
    }
  } finally {
    document.getElementById("solveBtn").disabled = false;
  }
}

function updateStats() {
  if (!result) return;
  document.getElementById("posErr").textContent = fmt(result.comparison.max_position_error);
  document.getElementById("velErr").textContent = fmt(result.comparison.max_velocity_error);
  const pred = result.prediction;
  const cls = pred.status === "certified" || pred.status === "computed" ? "ok" : "warn";
  document.getElementById("predStatus").innerHTML = `<span class="${cls}">${pred.status}</span>`;
  document.getElementById("energyDrift").textContent = fmt(result.simulation.energy_drift);
}

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return n === 0 ? "0" : n.toExponential(3);
}

function solveInBrowser(input, reason) {
  const masses = input.masses.map(Number);
  const positions = input.positions.map((row) => row.map(Number));
  const velocities = input.velocities.map((row) => row.map(Number));
  const dimension = validateBrowserInput(masses, positions, velocities, input.samples);
  const targetTime = Number(input.target_time);
  const samples = Number(input.samples);
  const state0 = packBrowserState(positions, velocities);
  const playback = integrateBrowser(state0, masses, dimension, targetTime, samples);
  const endpointSteps = Math.max(128, Math.min(24000, Math.ceil(Math.abs(targetTime) * 900)));
  const predictedState = integrateBrowserEndpoint(state0, masses, dimension, targetTime, endpointSteps);
  const predictedPositions = unpackBrowserPositions(predictedState, dimension);
  const predictedVelocities = unpackBrowserVelocities(predictedState, dimension);
  const comparison = compareBrowserStates(predictedState, playback.final_state, dimension);
  return {
    input: {
      masses,
      positions,
      velocities,
      target_time: targetTime,
      samples,
      dimension
    },
    prediction: {
      method: `${input.prediction_method}_browser_rk4_fallback`,
      status: "fallback",
      proof_certified: false,
      positions: predictedPositions,
      velocities: predictedVelocities,
      state: predictedState,
      state_interval: null,
      max_interval_radius: 0,
      missing_obligations: ["python_api_unreachable"],
      chart_count: null,
      route: null,
      note: `browser RK4 fallback because fetch failed: ${reason}`
    },
    simulation: playback,
    comparison
  };
}

function validateBrowserInput(masses, positions, velocities, samples) {
  if (masses.length !== 3 || masses.some((m) => !Number.isFinite(m) || m <= 0)) {
    throw new Error("masses must be three positive numbers");
  }
  if (positions.length !== 3 || velocities.length !== 3) {
    throw new Error("positions and velocities must contain three bodies");
  }
  const dimension = positions[0]?.length;
  if (![2, 3].includes(dimension)) throw new Error("positions must be 2D or 3D");
  for (let i = 0; i < 3; i++) {
    if (positions[i].length !== dimension || velocities[i].length !== dimension) {
      throw new Error("positions and velocities must have matching dimensions");
    }
    for (let d = 0; d < dimension; d++) {
      if (!Number.isFinite(positions[i][d]) || !Number.isFinite(velocities[i][d])) {
        throw new Error("positions and velocities must be finite");
      }
    }
  }
  if (!Number.isInteger(samples) || samples < 2 || samples > 2000) {
    throw new Error("samples must be between 2 and 2000");
  }
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      let r2 = 0;
      for (let d = 0; d < dimension; d++) r2 += (positions[i][d] - positions[j][d]) ** 2;
      if (r2 === 0) throw new Error("initial positions must be noncollision");
    }
  }
  return dimension;
}

function packBrowserState(positions, velocities) {
  return positions.flat().concat(velocities.flat());
}

function unpackBrowserPositions(state, dimension) {
  const positions = [];
  for (let i = 0; i < 3; i++) {
    positions.push(state.slice(i * dimension, (i + 1) * dimension));
  }
  return positions;
}

function unpackBrowserVelocities(state, dimension) {
  const offset = 3 * dimension;
  const velocities = [];
  for (let i = 0; i < 3; i++) {
    velocities.push(state.slice(offset + i * dimension, offset + (i + 1) * dimension));
  }
  return velocities;
}

function integrateBrowser(initialState, masses, dimension, targetTime, samples) {
  const times = [];
  const positions = [];
  const velocities = [];
  let state = initialState.slice();
  const h = samples === 1 ? 0 : targetTime / (samples - 1);
  const initialEnergy = browserEnergy(state, masses, dimension);
  for (let i = 0; i < samples; i++) {
    times.push(i * h);
    positions.push(unpackBrowserPositions(state, dimension));
    velocities.push(unpackBrowserVelocities(state, dimension));
    if (i < samples - 1) state = rk4BrowserStep(state, masses, dimension, h);
  }
  const finalEnergy = browserEnergy(state, masses, dimension);
  return {
    times,
    positions,
    velocities,
    final_state: state,
    final_positions: positions[positions.length - 1],
    final_velocities: velocities[velocities.length - 1],
    energy_initial: initialEnergy,
    energy_final: finalEnergy,
    energy_drift: finalEnergy - initialEnergy,
    center_of_mass_initial: browserCenterOfMass(initialState, masses, dimension),
    center_of_mass_final: browserCenterOfMass(state, masses, dimension),
    linear_momentum_initial: browserLinearMomentum(initialState, masses, dimension),
    linear_momentum_final: browserLinearMomentum(state, masses, dimension)
  };
}

function integrateBrowserEndpoint(initialState, masses, dimension, targetTime, steps) {
  let state = initialState.slice();
  if (targetTime === 0) return state;
  const h = targetTime / steps;
  for (let i = 0; i < steps; i++) state = rk4BrowserStep(state, masses, dimension, h);
  return state;
}

function rk4BrowserStep(state, masses, dimension, h) {
  const k1 = browserDerivative(state, masses, dimension);
  const k2 = browserDerivative(addScaledState(state, k1, 0.5 * h), masses, dimension);
  const k3 = browserDerivative(addScaledState(state, k2, 0.5 * h), masses, dimension);
  const k4 = browserDerivative(addScaledState(state, k3, h), masses, dimension);
  const next = new Array(state.length);
  for (let i = 0; i < state.length; i++) {
    next[i] = state[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }
  return next;
}

function addScaledState(state, deriv, scale) {
  return state.map((value, i) => value + scale * deriv[i]);
}

function browserDerivative(state, masses, dimension) {
  const coordinateCount = 3 * dimension;
  const deriv = new Array(state.length).fill(0);
  for (let i = 0; i < coordinateCount; i++) deriv[i] = state[coordinateCount + i];
  const acc = Array.from({length: 3}, () => new Array(dimension).fill(0));
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      const diff = new Array(dimension);
      let r2 = 0;
      for (let d = 0; d < dimension; d++) {
        diff[d] = state[j * dimension + d] - state[i * dimension + d];
        r2 += diff[d] * diff[d];
      }
      if (r2 === 0) throw new Error("collision encountered during browser fallback");
      const invR3 = 1 / (Math.sqrt(r2) * r2);
      for (let d = 0; d < dimension; d++) {
        acc[i][d] += masses[j] * diff[d] * invR3;
        acc[j][d] -= masses[i] * diff[d] * invR3;
      }
    }
  }
  for (let i = 0; i < 3; i++) {
    for (let d = 0; d < dimension; d++) {
      deriv[coordinateCount + i * dimension + d] = acc[i][d];
    }
  }
  return deriv;
}

function browserEnergy(state, masses, dimension) {
  const coordinateCount = 3 * dimension;
  let kinetic = 0;
  let potential = 0;
  for (let i = 0; i < 3; i++) {
    let v2 = 0;
    for (let d = 0; d < dimension; d++) {
      const v = state[coordinateCount + i * dimension + d];
      v2 += v * v;
    }
    kinetic += 0.5 * masses[i] * v2;
  }
  for (let i = 0; i < 3; i++) {
    for (let j = i + 1; j < 3; j++) {
      let r2 = 0;
      for (let d = 0; d < dimension; d++) {
        const diff = state[i * dimension + d] - state[j * dimension + d];
        r2 += diff * diff;
      }
      potential -= masses[i] * masses[j] / Math.sqrt(r2);
    }
  }
  return kinetic + potential;
}

function browserCenterOfMass(state, masses, dimension) {
  const totalMass = masses.reduce((sum, value) => sum + value, 0);
  const center = new Array(dimension).fill(0);
  for (let i = 0; i < 3; i++) {
    for (let d = 0; d < dimension; d++) center[d] += masses[i] * state[i * dimension + d];
  }
  return center.map((value) => value / totalMass);
}

function browserLinearMomentum(state, masses, dimension) {
  const coordinateCount = 3 * dimension;
  const momentum = new Array(dimension).fill(0);
  for (let i = 0; i < 3; i++) {
    for (let d = 0; d < dimension; d++) {
      momentum[d] += masses[i] * state[coordinateCount + i * dimension + d];
    }
  }
  return momentum;
}

function compareBrowserStates(predictedState, simulatedState, dimension) {
  const coordinateCount = 3 * dimension;
  const positionDelta = predictedState.slice(0, coordinateCount).map((value, i) => value - simulatedState[i]);
  const velocityDelta = predictedState.slice(coordinateCount).map((value, i) => value - simulatedState[coordinateCount + i]);
  return {
    max_position_error: maxAbs(positionDelta),
    rms_position_error: rms(positionDelta),
    max_velocity_error: maxAbs(velocityDelta),
    rms_velocity_error: rms(velocityDelta),
    simulation_final_inside_prediction_interval: null
  };
}

function maxAbs(values) {
  return Math.max(...values.map((value) => Math.abs(value)));
}

function rms(values) {
  return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
}

function bounds() {
  const pts = [];
  if (result) {
    for (const sample of result.simulation.positions) {
      for (const body of sample) pts.push(body);
    }
    for (const body of result.prediction.positions) pts.push(body);
  }
  if (!pts.length) return {minX: -1, maxX: 1, minY: -1, maxY: 1};
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
  }
  const pad = Math.max(maxX - minX, maxY - minY, 1) * 0.12;
  return {minX: minX - pad, maxX: maxX + pad, minY: minY - pad, maxY: maxY + pad};
}

function draw() {
  const canvas = document.getElementById("scene");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#090c11";
  ctx.fillRect(0, 0, rect.width, rect.height);
  if (!result) return;
  const b = bounds();
  const sx = rect.width / (b.maxX - b.minX);
  const sy = rect.height / (b.maxY - b.minY);
  const s = Math.min(sx, sy);
  const ox = (rect.width - (b.maxX - b.minX) * s) / 2;
  const oy = (rect.height - (b.maxY - b.minY) * s) / 2;
  const xy = (p) => [ox + (p[0] - b.minX) * s, rect.height - (oy + (p[1] - b.minY) * s)];
  const maxFrame = result.simulation.positions.length - 1;
  const stop = Math.min(frame, maxFrame);

  ctx.lineWidth = 1.5;
  for (let body = 0; body < 3; body++) {
    ctx.strokeStyle = colors[body];
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    for (let i = 0; i <= stop; i++) {
      const [x, y] = xy(result.simulation.positions[i][body]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  for (let body = 0; body < 3; body++) {
    const [x, y] = xy(result.prediction.positions[body]);
    ctx.strokeStyle = colors[body];
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(x, y, 11, 0, Math.PI * 2);
    ctx.stroke();
  }

  const current = result.simulation.positions[stop];
  for (let body = 0; body < 3; body++) {
    const [x, y] = xy(current[body]);
    ctx.fillStyle = colors[body];
    ctx.beginPath();
    ctx.arc(x, y, 5 + Math.cbrt(result.input.masses[body]), 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#a8b3c4";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText(`t = ${result.simulation.times[stop].toFixed(4)}`, 14, 22);
}

function tick(ts) {
  if (playing && result) {
    if (!lastTick || ts - lastTick > 16) {
      frame += Math.max(1, Math.floor(result.simulation.positions.length / 360));
      if (frame >= result.simulation.positions.length - 1) {
        frame = result.simulation.positions.length - 1;
        playing = false;
      }
      draw();
      lastTick = ts;
    }
  }
  requestAnimationFrame(tick);
}

document.getElementById("solveBtn").addEventListener("click", solve);
document.getElementById("presetBtn").addEventListener("click", () => setPreset(defaultPreset));
document.getElementById("figureBtn").addEventListener("click", () => setPreset(figureEightPreset));
document.getElementById("playBtn").addEventListener("click", () => {
  if (!result) return;
  if (frame >= result.simulation.positions.length - 1) frame = 0;
  playing = !playing;
});
window.addEventListener("resize", draw);
setPreset(defaultPreset);
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def solve_payload(payload: dict[str, Any]) -> dict[str, Any]:
    masses, positions, velocities, target_time, samples, method, order = _parse_payload(
        payload,
    )
    start = time.perf_counter()
    prediction = _predict_endpoint(
        masses,
        positions,
        velocities,
        target_time,
        method=method,
        order=order,
    )
    prediction["elapsed_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
    simulation = _simulate_trajectory(
        masses,
        positions,
        velocities,
        target_time,
        samples=samples,
    )
    comparison = _compare_prediction_to_simulation(
        prediction,
        simulation,
        positions.shape[1],
    )
    return {
        "input": {
            "masses": masses.tolist(),
            "positions": positions.tolist(),
            "velocities": velocities.tolist(),
            "target_time": target_time,
            "samples": samples,
            "dimension": int(positions.shape[1]),
        },
        "prediction": prediction,
        "simulation": simulation,
        "comparison": comparison,
    }


def _parse_payload(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, str, int]:
    masses = np.asarray(payload.get("masses"), dtype=float)
    positions = np.asarray(payload.get("positions"), dtype=float)
    velocities = np.asarray(payload.get("velocities"), dtype=float)
    target_time = float(payload.get("target_time", 0.0))
    samples = int(payload.get("samples", 500))
    method = str(payload.get("prediction_method", "taylor"))
    order = int(payload.get("order", 24))
    if masses.shape != (3,) or np.any(masses <= 0.0):
        raise ValueError("masses must be three positive numbers")
    if positions.ndim != 2 or positions.shape[0] != 3 or positions.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (3, 2) or (3, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions shape")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("positions and velocities must be finite")
    if not np.isfinite(target_time):
        raise ValueError("target_time must be finite")
    if samples < 2 or samples > 2000:
        raise ValueError("samples must be between 2 and 2000")
    if method not in {"taylor", "validated_atlas", "compactified_sundman", "reference"}:
        raise ValueError("unknown prediction_method")
    if order < 2 or order > 64:
        raise ValueError("order must be between 2 and 64")
    for i in range(3):
        for j in range(i + 1, 3):
            if np.linalg.norm(positions[i] - positions[j]) <= 0.0:
                raise ValueError("initial positions must be noncollision")
    return masses, positions, velocities, target_time, samples, method, order


def _predict_endpoint(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    target_time: float,
    *,
    method: str,
    order: int,
) -> dict[str, Any]:
    coordinate_count = positions.size
    if method == "taylor":
        solution = construct_taylor_solution(
            positions,
            velocities,
            masses,
            order=order,
        )
        state = np.asarray(solution.state_at(target_time), dtype=float)
        return _prediction_from_state(
            method=method,
            state=state,
            coordinate_count=coordinate_count,
            status="computed",
            proof_certified=False,
            note="local Taylor polynomial predictor; use short target times",
        )
    if method == "reference":
        state = integrate_reference(positions, velocities, masses, target_time)
        return _prediction_from_state(
            method=method,
            state=state,
            coordinate_count=coordinate_count,
            status="computed",
            proof_certified=False,
            note="reference endpoint uses the same numerical integrator family",
        )

    evaluator_method = (
        "validated_atlas" if method == "validated_atlas" else "compactified_sundman"
    )
    result = evaluate_unrestricted_solution(
        masses,
        positions,
        velocities,
        target_time,
        method=evaluator_method,
        initial_radius=0.0,
        order=min(order, 24),
    )
    interval = np.asarray(result.target_state_interval, dtype=object).reshape(-1)
    midpoint, bounds, max_radius = _midpoint_and_bounds(interval)
    proof_certified = bool(getattr(result, "proof_certified", False))
    status = "certified" if proof_certified else "enclosed"
    prediction = _prediction_from_state(
        method=method,
        state=midpoint,
        coordinate_count=coordinate_count,
        status=status,
        proof_certified=proof_certified,
        note="midpoint of returned target-state interval",
    )
    prediction["state_interval"] = bounds
    prediction["max_interval_radius"] = max_radius
    prediction["missing_obligations"] = list(
        getattr(result, "missing_obligations", ()),
    )
    prediction["chart_count"] = getattr(result, "chart_count", None)
    prediction["route"] = _selector_route(result)
    return prediction


def _prediction_from_state(
    *,
    method: str,
    state: np.ndarray,
    coordinate_count: int,
    status: str,
    proof_certified: bool,
    note: str,
) -> dict[str, Any]:
    positions = np.asarray(state[:coordinate_count], dtype=float).reshape(3, -1)
    velocities = np.asarray(state[coordinate_count:], dtype=float).reshape(3, -1)
    return {
        "method": method,
        "status": status,
        "proof_certified": proof_certified,
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "state": np.asarray(state, dtype=float).tolist(),
        "state_interval": None,
        "max_interval_radius": 0.0,
        "missing_obligations": [],
        "chart_count": None,
        "route": None,
        "note": note,
    }


def _simulate_trajectory(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    target_time: float,
    *,
    samples: int,
) -> dict[str, Any]:
    initial_state = pack_state(positions, velocities)
    orbit = integrate(
        initial_state,
        target_time,
        masses=masses,
        samples=samples,
        rtol=1e-11,
        atol=1e-13,
    )
    dimension = positions.shape[1]
    coordinate_count = positions.size
    sample_positions = orbit.states[:, :coordinate_count].reshape(-1, 3, dimension)
    sample_velocities = orbit.states[:, coordinate_count:].reshape(-1, 3, dimension)
    initial_energy = energy(initial_state, masses)
    final_energy = energy(orbit.final_state, masses)
    return {
        "times": np.asarray(orbit.times, dtype=float).tolist(),
        "positions": sample_positions.tolist(),
        "velocities": sample_velocities.tolist(),
        "final_state": np.asarray(orbit.final_state, dtype=float).tolist(),
        "final_positions": sample_positions[-1].tolist(),
        "final_velocities": sample_velocities[-1].tolist(),
        "energy_initial": initial_energy,
        "energy_final": final_energy,
        "energy_drift": final_energy - initial_energy,
        "center_of_mass_initial": center_of_mass(initial_state, masses).tolist(),
        "center_of_mass_final": center_of_mass(orbit.final_state, masses).tolist(),
        "linear_momentum_initial": linear_momentum(initial_state, masses).tolist(),
        "linear_momentum_final": linear_momentum(orbit.final_state, masses).tolist(),
    }


def _compare_prediction_to_simulation(
    prediction: dict[str, Any],
    simulation: dict[str, Any],
    dimension: int,
) -> dict[str, Any]:
    predicted_state = np.asarray(prediction["state"], dtype=float)
    simulated_state = np.asarray(simulation["final_state"], dtype=float)
    coordinate_count = 3 * dimension
    position_delta = predicted_state[:coordinate_count] - simulated_state[:coordinate_count]
    velocity_delta = predicted_state[coordinate_count:] - simulated_state[coordinate_count:]
    interval = prediction.get("state_interval")
    within_interval = None
    if interval is not None:
        bounds = np.asarray(interval, dtype=float)
        within_interval = bool(
            np.all(simulated_state >= bounds[:, 0])
            and np.all(simulated_state <= bounds[:, 1])
        )
    return {
        "max_position_error": float(np.linalg.norm(position_delta, ord=np.inf)),
        "rms_position_error": float(np.sqrt(np.mean(position_delta**2))),
        "max_velocity_error": float(np.linalg.norm(velocity_delta, ord=np.inf)),
        "rms_velocity_error": float(np.sqrt(np.mean(velocity_delta**2))),
        "simulation_final_inside_prediction_interval": within_interval,
    }


def _midpoint_and_bounds(interval: np.ndarray) -> tuple[np.ndarray, list[list[float]], float]:
    mids = []
    bounds = []
    max_radius = 0.0
    for item in interval:
        lower, upper = _bounds_for_value(item)
        mids.append(0.5 * (lower + upper))
        bounds.append([lower, upper])
        max_radius = max(max_radius, 0.5 * abs(upper - lower))
    return np.asarray(mids, dtype=float), bounds, float(max_radius)


def _bounds_for_value(item: Any) -> tuple[float, float]:
    if hasattr(item, "lower") and hasattr(item, "upper"):
        return float(item.lower), float(item.upper)
    value = float(item)
    return value, value


def _selector_route(result: Any) -> str | None:
    trace = getattr(result, "selector_trace", None)
    if trace is None:
        return None
    for attempt in getattr(trace, "attempts", ()):
        if getattr(attempt, "selected", False):
            return str(getattr(attempt, "route_id", ""))
    return None


class VisualizerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            body = HTML.replace("__DEFAULT_PRESET__", json.dumps(DEFAULT_PRESET))
            self._send(200, body.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/default":
            self._send_json(200, DEFAULT_PRESET)
            return
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/solve":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = solve_payload(payload)
        except Exception as error:  # noqa: BLE001 - API boundary reports messages.
            self._send_json(400, {"error": str(error)})
            return
        self._send_json(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}", file=sys.stderr)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(
            status,
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving three-body visualizer at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
