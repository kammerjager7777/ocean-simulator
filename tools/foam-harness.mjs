// Developer fixture for the v31.1 foam checks. Not a look change.
//
// Page (either build that has ?fixtest=1, lay(), __foamRead, __oceanCamCtl):
//   http://127.0.0.1:4173/?fixtest=1&fixdt=0.05&nodraw=1&camdist=160&layers=wind,buoy
// Frozen v31 pair, local copy only:
//   http://127.0.0.1:4174/?fixtest=1&fixdt=0.05&nodraw=1&camdist=160&layers=wind,buoy
//
// Checks, viewport 640x400. Fetch this file and eval(install). Do not edit the shaders to rerun.
//
// (A) Zoom. settled() flat, then depositLine(30, -25, 1.6, 8, 8). Wait until
//     __foamReadResult.maxG > 0.4 and trailN is 9. Pause. aim(dist, 36, -25) and
//     crop at 48, 140, 400, then 48 again without unpausing. near and back must match.
// (B) Speed pair, two separate loads. Pause, aim(60, 38, -25), put the buoy at
//     (30, -25), then stroke([[30,-25],[46,-25]], 2.5) and the same line at 12.
//     untilAge(0.5) on each, park the buoy at (90, 90), crop. Compare normals, widths, peaks.
// (C) stroke([[30,-25],[42,-25],[42,-13]], 8). Pause, snapshot the points, step,
//     and confirm the count and the birth times did not change.
//
// stroke() dispatches PointerEvents one per frame. A busy-wait shares one timestamp
// and the speed stays 0. setPointerCapture is a no-op for these events: it throws
// on an untrusted pointer and the following pointercancel drops the drag.
// Stay unpaused during the stroke. A paused frame draws the sea, so the event gap
// no longer matches the requested world speed. The pointer leads the 120 ms spring
// until the buoy's along-track progress reaches the end.

export const FIXTURE = "fixtest=1&fixdt=0.05&nodraw=1&camdist=160&layers=wind,buoy";

export const install = `(() => {
  const sea = () => document.getElementById("sea");
  const api = () => window.__oceanBuoyApi;
  const P = () => window.__oceanUI23.proj;
  window.__foamHarness = {
    settled() {
      const s = window.__oceanState;
      if (!s || !s.fx) return null;
      const hs = (s.bands || []).map((b) => b.significantHeight);
      return {
        t: s.fx.t, speed: s.fx.speed, chop: Number(document.getElementById("chop").value),
        hs, flat: s.fx.speed === 0 && hs.every((v) => v < 0.05)
      };
    },
    depositLine(x0, z0, step, n, speed) {
      const a = api();
      a.trail.splice(0);
      a.pos[0] = x0; a.pos[1] = z0; a.tgt[0] = x0; a.tgt[1] = z0;
      a.lay(speed);
      for (let i = 1; i <= n; i++) { a.pos[0] = x0 + i * step; a.lay(speed); }
      const mid = a.trail[Math.floor(a.trail.length / 2)] || a.trail[0];
      window.__foamRead = mid ? { x: mid.x, z: mid.z } : null;
      window.__foamReadResult = null;
      a.pos[0] = 90; a.pos[1] = 90; a.tgt[0] = 90; a.tgt[1] = 90;
      return { n: a.trail.length, t: window.__oceanState.fx.t };
    },
    aim(dist, x = 36, z = -25) {
      const ctl = window.__oceanCamCtl;
      ctl.camTarget[0] = x; ctl.camTarget[1] = 0; ctl.camTarget[2] = z;
      ctl.camera.distance = dist;
    },
    pause(on) {
      const p = document.getElementById("pause");
      if (p.checked !== on) { p.checked = on; p.dispatchEvent(new Event("change", { bubbles: true })); }
      return p.checked;
    },
    // Straight or bent stroke. pts is [[x,z], ...]. Speed is world m/s from event time.
    // Resolves when the buoy is within 0.4 m of the last point, or 6 m past it.
    stroke(pts, speed) {
      const sea = document.getElementById("sea");
      const api = window.__oceanBuoyApi;
      const proj = window.__oceanUI23.proj;
      const segs = [];
      let total = 0;
      for (let i = 1; i < pts.length; i++) {
        const len = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
        segs.push({ a: pts[i - 1], b: pts[i], len, at: total });
        total += len;
      }
      const at = (s) => {
        const d = Math.max(0, s);
        for (const g of segs) {
          if (d <= g.at + g.len || g === segs[segs.length - 1]) {
            const u = g.len ? Math.min(Math.max(0, d - g.at), g.len + 20) / g.len : 0;
            return [g.a[0] + (g.b[0] - g.a[0]) * u, g.a[1] + (g.b[1] - g.a[1]) * u];
          }
        }
        return pts[pts.length - 1];
      };
      api.trail.splice(0);
      api.pos[0] = pts[0][0]; api.pos[1] = pts[0][1];
      api.tgt[0] = pts[0][0]; api.tgt[1] = pts[0][1];
      const speeds = [];
      const frame = () => new Promise((r) => requestAnimationFrame(r));
      // Along-track metres of a world point on the polyline, so an L keeps its corner.
      const progress = (x, z) => {
        let best = 0;
        for (const g of segs) {
          const dx = g.b[0] - g.a[0], dz = g.b[1] - g.a[1], l2 = g.len * g.len || 1;
          let u = ((x - g.a[0]) * dx + (z - g.a[1]) * dz) / l2;
          u = Math.max(0, Math.min(1, u));
          const px = g.a[0] + dx * u, pz = g.a[1] + dz * u;
          if (Math.hypot(x - px, z - pz) < 1.2) best = Math.max(best, g.at + g.len * u);
        }
        return best;
      };
      // One move per frame, then a frame while the button is still down, so buoy.draw
      // lays the trail before pointerup. Stay unpaused: a paused frame draws the sea
      // and the event gap no longer matches the requested world speed.
      return (async () => {
        this.pause(false);
        await frame();
        // Untrusted events make setPointerCapture throw, and the browser then
        // sends pointercancel, which drops the drag. The hit test already ran.
        sea.setPointerCapture = () => {};
        const rect = sea.getBoundingClientRect();
        const fire = (type, x, z) => {
          const p = proj(x, 0, z); if (!p) return false;
          sea.dispatchEvent(new PointerEvent(type, {
            bubbles: true, cancelable: true, pointerId: 1, pointerType: "mouse", isPrimary: true,
            clientX: rect.left + p[0], clientY: rect.top + p[1], button: 0, buttons: type === "pointerup" ? 0 : 1
          }));
          return true;
        };
        const tBirth = window.__oceanState.fx.t;
        fire("pointerdown", pts[0][0], pts[0][1]);
        const t0 = performance.now();
        let caught = false, guard = 0;
        for (; guard < 900; guard++) {
          await frame();
          const elapsed = (performance.now() - t0) / 1000;
          const dist = speed * elapsed;
          const [x, z] = at(dist);
          if (Math.hypot(api.tgt[0] - x, api.tgt[1] - z) > 3) fire("pointerdown", x, z);
          fire("pointermove", x, z);
          const ds = window.__oceanUI23.debug && window.__oceanUI23.debug.buoy && window.__oceanUI23.debug.buoy.dragSpeed;
          if (dist > total * 0.2 && ds > 0) speeds.push(ds);
          const along = progress(api.pos[0], api.pos[1]);
          if (dist >= total && along >= total - 0.45) { caught = true; await frame(); break; }
          if (elapsed > total / speed + 4) break;
        }
        const end = at(Math.max(0, speed * (performance.now() - t0) / 1000));
        fire("pointerup", end[0], end[1]);
        const sorted = speeds.slice().sort((p, q) => p - q);
        const tr = api.trail;
        return {
          n: tr.length, t0: tBirth, t: window.__oceanState.fx.t, paused: document.getElementById("pause").checked,
          pos: [+api.pos[0].toFixed(2), +api.pos[1].toFixed(2)],
          tgt: [+api.tgt[0].toFixed(2), +api.tgt[1].toFixed(2)],
          dragMed: sorted.length ? +sorted[sorted.length >> 1].toFixed(2) : null,
          caught, frames: guard,
          pts: tr.map((p) => [+p.x.toFixed(2), +p.z.toFixed(2), +Math.hypot(p.nx, p.nz).toFixed(3), +p.w.toFixed(2)])
        };
      })();
    },
    trailStats() {
      const tr = api().trail;
      const gaps = [];
      for (let i = 1; i < tr.length; i++) gaps.push(Math.hypot(tr[i].x - tr[i - 1].x, tr[i].z - tr[i - 1].z));
      const distb = tr.map((p) => Math.hypot(p.nx, p.nz));
      const t = window.__oceanState.fx.t;
      return {
        n: tr.length,
        age: tr.length ? t - tr[0].t : null,
        t,
        gaps,
        disturb: distb,
        field: window.__foamReadResult || null,
        xs: tr.map((p) => +p.x.toFixed(2)),
        zs: tr.map((p) => +p.z.toFixed(2)),
        w: tr.map((p) => +p.w.toFixed(2))
      };
    },
    // Unpause until the newest trail point reaches the given age in seconds, then pause.
    // fixdt makes the landing exact to one physics step.
    async untilAge(age) {
      const tr = api().trail;
      if (!tr.length) return null;
      const born = Math.max(...tr.map((p) => p.t));
      this.pause(false);
      for (let i = 0; i < 40; i++) {
        if (window.__oceanState.fx.t - born >= age) break;
        await new Promise((r) => requestAnimationFrame(r));
      }
      this.pause(true);
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      return { t: window.__oceanState.fx.t, age: window.__oceanState.fx.t - born, oldest: window.__oceanState.fx.t - Math.min(...tr.map((p) => p.t)) };
    },
    async crop(name) {
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const tr = api().trail;
      const pts = tr.length
        ? tr.map((q) => P()(q.x, 0, q.z))
        : [0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => P()(30 + i * 1.6, 0, -25));
      const xs = pts.filter(Boolean).map((q) => q[0]), ys = pts.filter(Boolean).map((q) => q[1]);
      const x0 = Math.max(0, Math.floor(Math.min(...xs) - 10));
      const x1 = Math.min(639, Math.ceil(Math.max(...xs) + 10));
      const y0 = Math.max(0, Math.floor(Math.min(...ys) - 10));
      const y1 = Math.min(399, Math.ceil(Math.max(...ys) + 10));
      const gl = sea().getContext("webgl2");
      const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
      let peak = 0, bright = 0, area = 0;
      const on = x1 >= x0 && y1 >= y0;
      if (on) for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const i = ((h - 1 - y) * w + x) * 4;
        const r = px[i], g = px[i + 1], b = px[i + 2];
        const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        area++;
        if (luma > peak) peak = luma;
        if (luma > 150 && r > 140 && g > 145) bright++;
      }
      const c = document.createElement("canvas");
      c.width = Math.max(1, x1 - x0 + 1); c.height = Math.max(1, y1 - y0 + 1);
      const ctx = c.getContext("2d"), img = ctx.createImageData(c.width, c.height);
      if (on) for (let y = y0; y <= y1; y++) img.data.set(px.subarray(((h - 1 - y) * w + x0) * 4, ((h - 1 - y) * w + x1 + 1) * 4), (y - y0) * c.width * 4);
      ctx.putImageData(img, 0, 0);
      await fetch("http://127.0.0.1:8765/" + name, { method: "POST", body: c.toDataURL("image/png").split(",")[1] });
      return { name, box: [x0, y0, x1, y1], bright, peak: Math.round(peak), fill: area ? bright / area : 0, n: tr.length, age: tr.length ? window.__oceanState.fx.t - tr[0].t : null, t: window.__oceanState.fx.t, field: window.__foamReadResult };
    }
  };
})();`;
