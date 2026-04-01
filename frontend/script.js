/* ═══════════════════════════════════════════════════════════════
   ExoHabitAI — Frontend Script
   Features: Starfield · API calls · Predictions · Rankings
             Random fill · Scan animation · Feature bars
═══════════════════════════════════════════════════════════════ */

// Auto-detect: use relative /api in production, localhost in dev
const API = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:5000'
  : '/api';

/* ── Starfield ────────────────────────────────────────────────── */
(function () {
  const cv  = document.getElementById('starfield');
  const ctx = cv.getContext('2d');
  let stars = [], W, H;

  function resize() {
    W = cv.width  = window.innerWidth;
    H = cv.height = window.innerHeight;
  }
  function make() {
    stars = Array.from({ length: 300 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.5 + 0.2,
      a: Math.random() * 0.8 + 0.2,
      t: Math.random() * Math.PI * 2,
      s: Math.random() * 0.004 + 0.001,
    }));
  }
  function draw() {
    ctx.fillStyle = '#08080f';
    ctx.fillRect(0, 0, W, H);
    stars.forEach(s => {
      s.t += s.s;
      const a = s.a * (0.45 + 0.55 * Math.sin(s.t));
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(180, 215, 255, ${a})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize', () => { resize(); make(); });
  resize(); make(); draw();
})();

/* ── Tab Navigation ───────────────────────────────────────────── */
function showTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  const tab = document.getElementById('tab-' + id);
  const btn = document.querySelector(`.nav-btn[onclick*="'${id}'"]`);
  if (tab) tab.classList.add('active');
  if (btn) btn.classList.add('active');
  if (id === 'rankings') loadRankings();
  if (id === 'model')    renderModelFeats();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ── API Health ───────────────────────────────────────────────── */
async function checkHealth() {
  const dot = document.getElementById('apiDot');
  const txt = document.getElementById('apiTxt');
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    dot.className = d.model_loaded ? 'dot online' : 'dot offline';
    txt.textContent = d.model_loaded ? 'API Online' : 'Model Error';
  } catch {
    dot.className = 'dot offline';
    txt.textContent = 'API Offline';
  }
}

async function loadStats() {
  try {
    const r = await fetch(`${API}/stats`);
    const d = await r.json();
    if (d.status === 'success') {
      document.getElementById('sTot').textContent = d.total_planets.toLocaleString();
      document.getElementById('sHab').textContent = d.habitable_planets;
    }
  } catch { /* silent */ }
}

checkHealth();
loadStats();
setInterval(checkHealth, 30000);

/* ── Presets ──────────────────────────────────────────────────── */
const PRESETS = {
  earth: {
    planet_name:'Earth (Reference)', radius_earth:1.0, mass_earth:1.0,
    orbital_period:365.25, semimajor_axis:1.0, eq_temp_k:255, density:5.51,
    star_temp_k:5778, star_luminosity:0.0, star_metallicity:0.0, star_spectype:'G2 V',
    habitability_score:0.87, stellar_compatibility:1.0, orbital_stability:6.5,
  },
  kepler452b: {
    planet_name:'Kepler-452b', radius_earth:1.63, mass_earth:5.0,
    orbital_period:384.84, semimajor_axis:1.046, eq_temp_k:265, density:5.5,
    star_temp_k:5757, star_luminosity:0.079, star_metallicity:0.21, star_spectype:'G2 V',
    habitability_score:0.61, stellar_compatibility:0.9, orbital_stability:5.2,
  },
  proxima: {
    planet_name:'Proxima Centauri b', radius_earth:1.1, mass_earth:1.27,
    orbital_period:11.2, semimajor_axis:0.0485, eq_temp_k:234, density:5.8,
    star_temp_k:3042, star_luminosity:-1.37, star_metallicity:0.21, star_spectype:'M3 V',
    habitability_score:0.55, stellar_compatibility:0.7, orbital_stability:4.8,
  },
  hot_jupiter: {
    planet_name:'Hot Jupiter (Example)', radius_earth:11.2, mass_earth:317.8,
    orbital_period:3.5, semimajor_axis:0.048, eq_temp_k:1500, density:1.2,
    star_temp_k:5900, star_luminosity:0.1, star_metallicity:0.15, star_spectype:'G2 V',
    habitability_score:0.05, stellar_compatibility:0.3, orbital_stability:8.0,
  },
};

function fillPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  Object.entries(p).forEach(([k, v]) => {
    const el = document.getElementById(k);
    if (el) { el.value = v; el.classList.remove('invalid', 'valid'); }
  });
  hideResult();
}

/* ── Random Fill ─────────────────────────────────────────────── */
const SPECTRAL_TYPES = ['G2 V', 'K5 V', 'M3 V', 'F8 V', 'K0 III', 'G8 III', 'M0 V', 'A5 V'];

function rnd(lo, hi, dp = 2) {
  return parseFloat((Math.random() * (hi - lo) + lo).toFixed(dp));
}

function fillRandom() {
  const values = {
    planet_name:           `Planet-${Math.random().toString(36).slice(2,7).toUpperCase()}`,
    radius_earth:          rnd(0.4, 4.0),
    mass_earth:            rnd(0.1, 20.0),
    orbital_period:        rnd(10, 1000, 2),
    semimajor_axis:        rnd(0.05, 3.0, 3),
    eq_temp_k:             rnd(100, 600, 0),
    density:               rnd(0.5, 12.0),
    star_temp_k:           rnd(3000, 8000, 0),
    star_luminosity:       rnd(-2.5, 1.5),
    star_metallicity:      rnd(-1.5, 0.8),
    star_spectype:         SPECTRAL_TYPES[Math.floor(Math.random() * SPECTRAL_TYPES.length)],
    habitability_score:    rnd(0.0, 1.0),
    stellar_compatibility: rnd(0.0, 2.0),
    orbital_stability:     rnd(1.0, 15.0),
  };
  Object.entries(values).forEach(([k, v]) => {
    const el = document.getElementById(k);
    if (el) { el.value = v; el.classList.remove('invalid', 'valid'); }
  });
  hideResult();
  // Flash hint
  const btn = document.querySelector('.qf-random');
  if (btn) {
    btn.textContent = '✓ Filled!';
    setTimeout(() => { btn.textContent = '🎲 Random'; }, 1200);
  }
}

function hideResult() {
  document.getElementById('resultCol').style.display = 'none';
  document.getElementById('formErr').style.display   = 'none';
}

/* ── Form Validation ─────────────────────────────────────────── */
const NUM_FIELDS = [
  'radius_earth','mass_earth','orbital_period','semimajor_axis',
  'eq_temp_k','density','star_temp_k','star_luminosity',
  'star_metallicity','habitability_score','stellar_compatibility','orbital_stability',
];

function getFormData() {
  const data = {}, errors = [];

  const nameEl = document.getElementById('planet_name');
  data.planet_name = nameEl?.value.trim() || 'Unknown Planet';

  NUM_FIELDS.forEach(f => {
    const el  = document.getElementById(f);
    const val = el?.value.trim();
    if (!val) {
      errors.push(`${f.replace(/_/g,' ')} is required`);
      el?.classList.add('invalid'); el?.classList.remove('valid');
    } else {
      const n = parseFloat(val);
      if (isNaN(n)) {
        errors.push(`${f.replace(/_/g,' ')} must be a number`);
        el?.classList.add('invalid'); el?.classList.remove('valid');
      } else {
        data[f] = n;
        el?.classList.remove('invalid'); el?.classList.add('valid');
      }
    }
  });

  const stEl = document.getElementById('star_spectype');
  const stVal = stEl?.value;
  if (!stVal) { errors.push('Spectral type is required'); stEl?.classList.add('invalid'); }
  else { data.star_spectype = stVal; stEl?.classList.remove('invalid'); stEl?.classList.add('valid'); }

  return { data, errors };
}

/* ── Scan Animation ──────────────────────────────────────────── */
const STEPS = [
  'Initializing XGBoost pipeline…',
  'Applying ColumnTransformer…',
  'Running median imputation…',
  'StandardScaler normalizing…',
  'Running gradient boosted trees…',
  'Computing probability estimates…',
  'Evaluating habitability class…',
];
let scanTick;

function startScan() {
  const ov   = document.getElementById('scanOverlay');
  const step = document.getElementById('scanStep');
  ov.classList.add('active');
  let i = 0;
  step.textContent = STEPS[0];
  scanTick = setInterval(() => { i = (i + 1) % STEPS.length; step.textContent = STEPS[i]; }, 550);
}
function stopScan() {
  clearInterval(scanTick);
  document.getElementById('scanOverlay').classList.remove('active');
}

/* ── Gauge ───────────────────────────────────────────────────── */
function renderGauge(score, color) {
  const arc   = document.getElementById('gaugeArc');
  const num   = document.getElementById('gScore');
  const total = 267;
  arc.style.strokeDashoffset = (total - total * score / 100).toString();
  num.style.fill = color;

  let cur = 0;
  const step = score / 55;
  const t = setInterval(() => {
    cur = Math.min(cur + step, score);
    num.textContent = cur.toFixed(1) + '%';
    if (cur >= score) clearInterval(t);
  }, 16);
}

function scoreColor(s) {
  if (s >= 80) return '#22e87a';
  if (s >= 50) return '#f5c842';
  if (s >= 20) return '#ff9a3c';
  return '#ff5555';
}
function scoreClass(s) {
  if (s >= 80) return 'score-hi';
  if (s >= 50) return 'score-md';
  if (s >= 20) return 'score-lo';
  return 'score-un';
}

/* ── Display Result ──────────────────────────────────────────── */
const FEATURES = [
  { name:'Habitability Score',      pct:49.3 },
  { name:'Equilibrium Temperature', pct:21.5 },
  { name:'Eq. Temp (Scaled)',       pct:19.7 },
  { name:'Planet Radius (Scaled)',  pct:2.7  },
  { name:'Jupiter Radius',          pct:2.5  },
  { name:'Star Temperature',        pct:1.7  },
  { name:'Planet Radius',           pct:1.6  },
];

function displayResult(d) {
  const col = document.getElementById('resultCol');
  col.style.display = 'flex';

  const score = d.habitability_score;
  const color = d.class_color || scoreColor(score);
  const pHab  = d.probabilities.habitable;
  const pNo   = d.probabilities.non_habitable;

  // Gauge
  renderGauge(score, color);

  // Verdict
  document.getElementById('vIcon').textContent  = d.class_icon || '—';
  const vLbl = document.getElementById('vLabel');
  vLbl.textContent  = d.label || '—';
  vLbl.style.color  = color;
  vLbl.style.textShadow = `0 0 14px ${color}`;
  document.getElementById('vClass').textContent = `Habitability Class: ${d.habitability_class}`;

  // Planet name
  document.getElementById('resName').textContent = d.planet_name || '—';

  // Prob bars
  setTimeout(() => {
    document.getElementById('barHab').style.width = pHab + '%';
    document.getElementById('barNo').style.width  = pNo  + '%';
  }, 120);
  document.getElementById('valHab').textContent = pHab.toFixed(1) + '%';
  document.getElementById('valNo').textContent  = pNo.toFixed(1)  + '%';

  // Meta
  const m = d.input_summary || {};
  document.getElementById('metaGrid').innerHTML = `
    <div class="meta-item">Radius: <span>${m.radius_earth ?? '—'} R⊕</span></div>
    <div class="meta-item">Eq. Temp: <span>${m.eq_temp_k ?? '—'} K</span></div>
    <div class="meta-item">Star Temp: <span>${m.star_temp_k ?? '—'} K</span></div>
    <div class="meta-item">Period: <span>${m.orbital_period ?? '—'} d</span></div>
  `;

  // HZ badge
  const eqT = m.eq_temp_k;
  const hzEl = document.getElementById('hzTag');
  if (eqT != null) {
    hzEl.style.display = 'block';
    const inHz = eqT >= 180 && eqT <= 310;
    hzEl.className = `hz-tag ${inHz ? 'hz-in' : 'hz-out'}`;
    hzEl.textContent = inHz
      ? '✓ Within Habitable Zone Temperature Range (180–310 K)'
      : `✗ Outside Habitable Zone — Eq. Temp: ${eqT} K`;
  } else { hzEl.style.display = 'none'; }

  // Feature bars
  const featEl = document.getElementById('featRows');
  featEl.innerHTML = FEATURES.map(f => `
    <div class="feat-row">
      <span class="feat-name">${f.name}</span>
      <div class="feat-track"><div class="feat-fill" style="width:0" data-w="${f.pct}%"></div></div>
      <span class="feat-pct">${f.pct}%</span>
    </div>
  `).join('');
  setTimeout(() => {
    featEl.querySelectorAll('.feat-fill').forEach(el => { el.style.width = el.dataset.w; });
  }, 200);

  // scroll result into view on mobile
  if (window.innerWidth < 1100) col.scrollIntoView({ behavior:'smooth', block:'start' });
}

/* ── Form Submit ─────────────────────────────────────────────── */
document.getElementById('predictForm').addEventListener('submit', async e => {
  e.preventDefault();
  const errEl = document.getElementById('formErr');
  errEl.style.display = 'none';

  const { data, errors } = getFormData();
  if (errors.length) { errEl.textContent = '⚠ ' + errors[0]; errEl.style.display = 'block'; return; }

  const btn = document.getElementById('submitBtn');
  btn.classList.add('loading'); btn.disabled = true;
  startScan();

  try {
    const res  = await fetch(`${API}/predict`, {
      method:'POST', headers:{ 'Content-Type':'application/json' }, body:JSON.stringify(data),
    });
    const result = await res.json();
    if (!res.ok || result.status === 'error') throw new Error(result.message || 'Prediction failed');
    stopScan();
    displayResult(result);
  } catch (err) {
    stopScan();
    errEl.textContent = err.message.includes('fetch')
      ? '⚠ Cannot connect to API. Make sure Flask is running on port 5000.'
      : `⚠ ${err.message}`;
    errEl.style.display = 'block';
    document.getElementById('resultCol').style.display = 'none';
  } finally {
    btn.classList.remove('loading'); btn.disabled = false;
  }
});

/* ── Rankings ─────────────────────────────────────────────────── */
async function loadRankings() {
  const limit   = document.getElementById('rankLimit').value;
  const habOnly = document.getElementById('rankFilter').value;
  const loading = document.getElementById('rankLoading');
  const table   = document.getElementById('rankTable');
  const tbody   = document.getElementById('rankBody');
  const top3w   = document.getElementById('top3Wrap');

  loading.style.display = 'flex';
  table.style.display   = 'none';
  top3w.style.display   = 'none';
  tbody.innerHTML       = '';

  try {
    const res  = await fetch(`${API}/rank?limit=${limit}&habitable_only=${habOnly}`);
    const data = await res.json();

    if (data.status !== 'success' || !data.planets?.length) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:2.5rem;color:var(--txt-muted);font-family:var(--f-mono)">No data available — ensure the Flask API is running and habitability_ranked.csv exists.</td></tr>`;
      loading.style.display = 'none'; table.style.display = 'table'; return;
    }

    // Top 3 podium
    ['top3-1','top3-2','top3-3'].forEach((id, i) => {
      const p = data.planets[i];
      if (!p) return;
      document.querySelector(`#${id} .t3-name`).textContent  = p.planet_name;
      document.querySelector(`#${id} .t3-score`).textContent = p.habitability_score.toFixed(1) + '% habitable';
    });
    top3w.style.display = 'grid';

    tbody.innerHTML = data.planets.map((p, i) => {
      const sc  = p.habitability_score;
      const cls = scoreClass(sc);
      const clr = scoreColor(sc);
      const rk  = i < 3 ? 'rn-top' : 'rn';
      const bg  = clr + '22';
      const badge = `<span class="cbadge" style="background:${bg};color:${clr};border:1px solid ${clr}55">${esc(p.habitability_class)}</span>`;
      return `
        <tr>
          <td><span class="${rk}">${p.rank || i+1}</span></td>
          <td><span class="planet-cell">${esc(p.planet_name)}</span></td>
          <td style="color:var(--txt-secondary)">${esc(p.host_star_name)}</td>
          <td><strong class="${cls}">${sc.toFixed(1)}%</strong></td>
          <td>${badge}</td>
          <td>${p.radius_earth    ?? '—'}</td>
          <td>${p.eq_temp_k       ?? '—'}</td>
          <td>${p.orbital_period  ?? '—'}</td>
          <td>${p.star_temp_k     ?? '—'}</td>
        </tr>`;
    }).join('');

    loading.style.display = 'none'; table.style.display = 'table';

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:2rem;color:var(--red);font-family:var(--f-mono)">⚠ ${err.message.includes('fetch') ? 'API unavailable — please try again shortly' : esc(err.message)}</td></tr>`;
    loading.style.display = 'none'; table.style.display = 'table';
  }
}

/* ── Model Feature Bars (static) ─────────────────────────────── */
function renderModelFeats() {
  const el = document.getElementById('modelFeatBars');
  if (!el || el.dataset.rendered) return;
  el.innerHTML = FEATURES.map(f => `
    <div class="fi-row">
      <span class="fi-name">${f.name}</span>
      <div class="fi-track"><div class="fi-fill" style="width:0" data-w="${f.pct}%"></div></div>
      <span class="fi-val">${f.pct}%</span>
    </div>
  `).join('');
  el.dataset.rendered = '1';
  setTimeout(() => {
    el.querySelectorAll('.fi-fill').forEach(b => { b.style.width = b.dataset.w; });
  }, 150);
}

/* ── Helpers ─────────────────────────────────────────────────── */
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
