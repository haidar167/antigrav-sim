"""AntiGravSim: Live Interactive Web Telemetry Dashboard.

Runs a lightweight, zero-dependency local web server (http://localhost:8080)
that executes the trained PPO policy in real time and streams flight telemetry
to an interactive aerospace HUD in your browser.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import os
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from stable_baselines3 import PPO

from antigrav_env import AntiGravEnv

# Global simulation state
sim_lock = threading.Lock()
env = None
model = None
obs = None
info = None
custom_target = 1.0
active_wind_impulse = 0.0

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AntiGravSim — Live Telemetry Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #090c10;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: #161b22;
      border-bottom: 1px solid #30363d;
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    header h1 {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 1px;
      color: #00e5ff;
    }
    .badge {
      background: #238636;
      color: #ffffff;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: bold;
    }
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 16px;
      padding: 16px;
      overflow: hidden;
    }
    .viewport-card {
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      display: flex;
      flex-direction: column;
      position: relative;
      overflow: hidden;
    }
    canvas {
      flex: 1;
      width: 100%;
      height: 100%;
      background: #0d1117;
    }
    .telemetry-panel {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 16px;
    }
    .card h2 {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #8b949e;
      margin-bottom: 12px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .metric {
      background: #0d1117;
      padding: 10px;
      border-radius: 6px;
      border-left: 3px solid #00e5ff;
    }
    .metric.green { border-left-color: #00ff88; }
    .metric.orange { border-left-color: #ff9100; }
    .metric.purple { border-left-color: #d500f9; }
    .metric-label {
      font-size: 11px;
      color: #8b949e;
    }
    .metric-val {
      font-size: 20px;
      font-weight: bold;
      margin-top: 4px;
      font-family: monospace;
    }
    .controls {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .slider-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 12px;
    }
    .slider-row span { color: #8b949e; }
    input[type=range] {
      width: 100%;
      accent-color: #00e5ff;
    }
    .btn-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    button {
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
      padding: 8px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      font-size: 12px;
      transition: all 0.2s;
    }
    button:hover {
      background: #30363d;
      border-color: #8b949e;
      color: #fff;
    }
    button.primary {
      background: #1f6feb;
      border-color: #388bfd;
      color: #fff;
    }
    button.primary:hover { background: #388bfd; }
    button.danger {
      background: #b62324;
      border-color: #e5534b;
      color: #fff;
    }
    button.danger:hover { background: #e5534b; }
    .hud-overlay {
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(22, 27, 34, 0.85);
      border: 1px solid #30363d;
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-family: monospace;
      line-height: 1.5;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <header>
    <h1>🛸 ANTIGRAVSIM — LIVE FLIGHT TELEMETRY</h1>
    <div class="badge" id="statusBadge">LIVE SIMULATION ACTIVE</div>
  </header>

  <main>
    <div class="viewport-card">
      <div class="hud-overlay" id="hudBox">
        STATUS: INITIALIZING...
      </div>
      <canvas id="simCanvas"></canvas>
    </div>

    <div class="telemetry-panel">
      <div class="card">
        <h2>Primary Instruments</h2>
        <div class="metric-grid">
          <div class="metric">
            <div class="metric-label">Altitude (z)</div>
            <div class="metric-val" id="valZ">0.00 m</div>
          </div>
          <div class="metric green">
            <div class="metric-label">Gravity Canceled</div>
            <div class="metric-val" id="valCancel">99.6%</div>
          </div>
          <div class="metric orange">
            <div class="metric-label">Thrust (u)</div>
            <div class="metric-val" id="valThrust">9.81 N</div>
          </div>
          <div class="metric purple">
            <div class="metric-label">Net Accel (az)</div>
            <div class="metric-val" id="valAz">0.00 m/s²</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Flight Controls</h2>
        <div class="controls">
          <div class="slider-row">
            <span>Target Altitude: <strong id="lblTarget">1.0</strong> m</span>
            <input type="range" id="rngTarget" min="0.3" max="2.5" step="0.1" value="1.0">
          </div>

          <div class="btn-row">
            <button onclick="injectWind(-4.0)">🌪️ Down Gust (-4N)</button>
            <button onclick="injectWind(4.0)">💨 Up Gust (+4N)</button>
          </div>
          <div class="btn-row">
            <button class="primary" onclick="resetSim(0.1)">🚀 Launch from Pad</button>
            <button class="danger" onclick="resetSim(2.0)">💥 Drop from 2.0m</button>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Physics Constants</h2>
        <p style="font-size: 11px; color: #8b949e; line-height: 1.6;">
          Mass: <strong>1.0 kg</strong><br>
          Earth Gravity (g): <strong>9.81 m/s²</strong><br>
          Equilibrium Thrust: <strong>mg = 9.81 N</strong><br>
          Control Policy: <strong>PPO (MlpPolicy, 50 Hz)</strong>
        </p>
      </div>
    </div>
  </main>

  <script>
    const canvas = document.getElementById('simCanvas');
    const ctx = canvas.getContext('2d');

    function resize() {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    let history = [];
    let state = { z: 1.0, vz: 0, thrust: 9.81, az: 0, target: 1.0, grav_cancel: 99.6, wind: 0, step: 0 };

    async function poll() {
      try {
        const res = await fetch('/api/step');
        if (res.ok) {
          state = await res.json();
          updateUI(state);
        }
      } catch (e) {}
      requestAnimationFrame(poll);
    }

    function updateUI(s) {
      document.getElementById('valZ').innerText = s.z.toFixed(2) + ' m';
      document.getElementById('valCancel').innerText = s.grav_cancel.toFixed(1) + '%';
      document.getElementById('valThrust').innerText = s.thrust.toFixed(2) + ' N';
      document.getElementById('valAz').innerText = (s.az >= 0 ? '+' : '') + s.az.toFixed(2) + ' m/s²';

      const isHover = Math.abs(s.z - s.target) < 0.08 && Math.abs(s.vz) < 0.15;
      const status = isHover ? 'ANTIGRAVITY HOVER' : (s.vz > 0.1 ? 'ASCENDING' : 'STABILIZING');

      document.getElementById('hudBox').innerHTML =
        `MODE:     ${status}<br>` +
        `ALTITUDE: ${s.z.toFixed(3)} m<br>` +
        `TARGET:   ${s.target.toFixed(2)} m<br>` +
        `NET ACC:  ${(s.az >= 0 ? '+' : '') + s.az.toFixed(3)} m/s²<br>` +
        `WIND:     ${s.wind.toFixed(2)} N`;

      history.push({ z: s.z, target: s.target });
      if (history.length > 200) history.shift();

      render();
    }

    function render() {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // Coordinate transforms: Ground at bottom (y = h - 60)
      const groundY = h - 70;
      const pixelsPerMeter = (groundY - 80) / 3.0;

      // Draw grid
      ctx.strokeStyle = '#1f242c';
      ctx.lineWidth = 1;
      for (let m = 0; m <= 3.0; m += 0.5) {
        const y = groundY - m * pixelsPerMeter;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();

        ctx.fillStyle = '#484f58';
        ctx.font = '10px monospace';
        ctx.fillText(m.toFixed(1) + 'm', 15, y - 4);
      }

      // Draw Ground
      ctx.fillStyle = '#161b22';
      ctx.fillRect(0, groundY, w, h - groundY);
      ctx.strokeStyle = '#30363d';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, groundY);
      ctx.lineTo(w, groundY);
      ctx.stroke();

      // Draw Target Altitude Line
      const targetY = groundY - state.target * pixelsPerMeter;
      ctx.strokeStyle = '#ffea00';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.beginPath();
      ctx.moveTo(0, targetY);
      ctx.lineTo(w, targetY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Target Label
      ctx.fillStyle = '#ffea00';
      ctx.font = 'bold 11px monospace';
      ctx.fillText('TARGET: ' + state.target.toFixed(1) + 'm', w - 120, targetY - 6);

      // Draw Levitating Object
      const objX = w / 2;
      const objY = groundY - state.z * pixelsPerMeter;
      const radius = 22;

      // Draw Thruster Plume
      const thrustRatio = Math.min(Math.max(state.thrust / 20.0, 0), 1);
      if (state.thrust > 0.5) {
        const plumeLen = 20 + thrustRatio * 70;
        const grad = ctx.createLinearGradient(objX, objY + radius, objX, objY + radius + plumeLen);
        grad.addColorStop(0, 'rgba(255, 255, 255, 0.9)');
        grad.addColorStop(0.2, 'rgba(255, 145, 0, 0.8)');
        grad.addColorStop(1, 'rgba(255, 0, 60, 0)');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(objX - 10, objY + radius);
        ctx.lineTo(objX + 10, objY + radius);
        ctx.lineTo(objX, objY + radius + plumeLen);
        ctx.closePath();
        ctx.fill();
      }

      // Draw Object Body (Sphere)
      const sphereGrad = ctx.createRadialGradient(objX - 6, objY - 6, 2, objX, objY, radius);
      sphereGrad.addColorStop(0, '#58a6ff');
      sphereGrad.addColorStop(1, '#0969da');

      ctx.fillStyle = sphereGrad;
      ctx.beginPath();
      ctx.arc(objX, objY, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#e6edf3';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Altitude Trail
      if (history.length > 1) {
        ctx.strokeStyle = 'rgba(0, 229, 255, 0.4)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < history.length; i++) {
          const tx = objX - (history.length - i) * 3;
          const ty = groundY - history[i].z * pixelsPerMeter;
          if (i === 0) ctx.moveTo(tx, ty);
          else ctx.lineTo(tx, ty);
        }
        ctx.stroke();
      }
    }

    document.getElementById('rngTarget').addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      document.getElementById('lblTarget').innerText = val.toFixed(1);
      fetch('/api/target', { method: 'POST', body: JSON.stringify({ target: val }) });
    });

    function injectWind(mag) {
      fetch('/api/wind', { method: 'POST', body: JSON.stringify({ wind: mag }) });
    }

    function resetSim(alt) {
      fetch('/api/reset', { method: 'POST', body: JSON.stringify({ z: alt }) });
    }

    poll();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global env, model, obs, info, active_wind_impulse

        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif self.path == "/api/step":
            with sim_lock:
                action, _ = model.predict(obs, deterministic=True)

                # Inject one-shot wind impulse if active
                if abs(active_wind_impulse) > 0.001:
                    env.wind_strength = abs(active_wind_impulse)
                    # Apply bias to wind
                    obs, reward, terminated, truncated, info = env.step(action)
                    active_wind_impulse *= 0.85
                    if abs(active_wind_impulse) < 0.1:
                        active_wind_impulse = 0.0
                        env.wind_strength = 0.0
                else:
                    obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    obs, info = env.reset(options={"initial_z": custom_target, "initial_vz": 0.0})

                mean_abs_az = abs(info["az"])
                grav_cancel = max(0.0, (1.0 - mean_abs_az / 9.81)) * 100.0

                payload = {
                    "z": float(info["z"]),
                    "vz": float(info["vz"]),
                    "thrust": float(info["thrust"]),
                    "az": float(info["az"]),
                    "wind": float(info.get("wind", 0.0)),
                    "target": float(env.z_target),
                    "grav_cancel": float(grav_cancel),
                    "step": int(info["step"]),
                }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        global custom_target, active_wind_impulse, obs, info
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8")
        data = json.loads(post_body) if post_body else {}

        if self.path == "/api/target":
            with sim_lock:
                custom_target = float(data.get("target", 1.0))
                env.z_target = custom_target

        elif self.path == "/api/wind":
            with sim_lock:
                active_wind_impulse = float(data.get("wind", 0.0))

        elif self.path == "/api/reset":
            with sim_lock:
                init_z = float(data.get("z", 1.0))
                obs, info = env.reset(options={"initial_z": init_z, "initial_vz": 0.0})

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, format, *args):
        # Silence default terminal request logging
        pass


def main():
    global env, model, obs, info

    model_path = "./logs/best_model.zip"
    if not os.path.exists(model_path):
        print("Error: logs/best_model.zip not found! Run train.py first.")
        sys.exit(1)

    print("Loading trained PPO model...")
    model = PPO.load(model_path)

    env = AntiGravEnv(z_target=1.0, random_start=False)
    obs, info = env.reset(options={"initial_z": 0.2, "initial_vz": 0.0})

    port = 8080
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://localhost:{port}"

    print("=" * 60)
    print(f"  AntiGravSim Live Dashboard Running at:")
    print(f"  >>> {url} <<<")
    print("=" * 60)
    print("  Controls available:")
    print("  - Adjust target altitude in real time")
    print("  - Trigger upward / downward wind disturbances")
    print("  - Launch from pad or drop from high altitude")
    print("  Press Ctrl+C to stop.")

    # Auto-open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live server...")
        server.server_close()


if __name__ == "__main__":
    main()
