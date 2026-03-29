// ════════════════════════════════════════════════════
//  ExoHabitAI — script.js  (complete rewrite)
// ════════════════════════════════════════════════════

const API_BASE_URL = 'http://localhost:5000';

// ── Sample datasets ──────────────────────────────────────────
const SAMPLE_DATA = {
  habitable: [
    { pl_name:'Kepler-442b-like', pl_rade:1.20, pl_bmasse:1.00, pl_orbper:365.25, pl_orbsmax:1.00,  pl_eqt:288,  st_teff:5772, st_rad:1.00, st_mass:1.00, st_met: 0.00, st_lum:1.00,  temp_score:0.95, dist_score:0.98, lum_score:0.92, stellar_temp_score:0.90, stellar_mass_score:0.88, stellar_compatibility_index:0.92, orbital_stability:0.95, spec_F:0, spec_G:1, spec_K:0, spec_M:0, spec_m:0 },
    { pl_name:'ExoEarth-Alpha',   pl_rade:1.50, pl_bmasse:1.20, pl_orbper:450.00, pl_orbsmax:1.10,  pl_eqt:275,  st_teff:5900, st_rad:1.05, st_mass:1.02, st_met: 0.05, st_lum:1.05, temp_score:0.92, dist_score:0.95, lum_score:0.90, stellar_temp_score:0.88, stellar_mass_score:0.85, stellar_compatibility_index:0.90, orbital_stability:0.93, spec_F:0, spec_G:1, spec_K:0, spec_M:0, spec_m:0 },
    { pl_name:'GreenWorld-7',     pl_rade:1.10, pl_bmasse:0.90, pl_orbper:330.00, pl_orbsmax:0.95,  pl_eqt:295,  st_teff:5600, st_rad:0.98, st_mass:0.98, st_met:-0.10, st_lum:0.95, temp_score:0.94, dist_score:0.97, lum_score:0.91, stellar_temp_score:0.91, stellar_mass_score:0.87, stellar_compatibility_index:0.91, orbital_stability:0.94, spec_F:0, spec_G:1, spec_K:0, spec_M:0, spec_m:0 },
    { pl_name:'TRAPPIST-1e-like', pl_rade:0.92, pl_bmasse:0.77, pl_orbper:280.00, pl_orbsmax:0.88,  pl_eqt:292,  st_teff:5400, st_rad:0.95, st_mass:0.95, st_met:-0.05, st_lum:0.90, temp_score:0.90, dist_score:0.93, lum_score:0.88, stellar_temp_score:0.86, stellar_mass_score:0.85, stellar_compatibility_index:0.89, orbital_stability:0.91, spec_F:0, spec_G:0, spec_K:1, spec_M:0, spec_m:0 },
    { pl_name:'Kepler-186f-like', pl_rade:1.17, pl_bmasse:1.08, pl_orbper:395.00, pl_orbsmax:1.03,  pl_eqt:269,  st_teff:5650, st_rad:1.02, st_mass:1.01, st_met: 0.02, st_lum:1.02, temp_score:0.88, dist_score:0.91, lum_score:0.86, stellar_temp_score:0.84, stellar_mass_score:0.83, stellar_compatibility_index:0.87, orbital_stability:0.89, spec_F:0, spec_G:1, spec_K:0, spec_M:0, spec_m:0 },
  ],
  nonHabitable: [
    // ❄️ FROZEN — eqt:85K, all scores very low, M-type star
    { pl_name:'Frozen-Cryo',  pl_rade:1.30, pl_bmasse:1.10, pl_orbper:1820.0, pl_orbsmax:3.50, pl_eqt:85,   st_teff:3900, st_rad:0.55, st_mass:0.52, st_met:-0.30, st_lum:0.08, temp_score:0.04, dist_score:0.02, lum_score:0.03, stellar_temp_score:0.06, stellar_mass_score:0.05, stellar_compatibility_index:0.04, orbital_stability:0.05, spec_F:0, spec_G:0, spec_K:0, spec_M:1, spec_m:0 },
    { pl_name:'HotJupiter-X1',pl_rade:4.50, pl_bmasse: 8.20, pl_orbper: 45.30, pl_orbsmax:0.20, pl_eqt:1200, st_teff:3500, st_rad:0.50, st_mass:0.40, st_met:-0.50, st_lum:0.05, temp_score:0.04, dist_score:0.03, lum_score:0.02, stellar_temp_score:0.05, stellar_mass_score:0.04, stellar_compatibility_index:0.03, orbital_stability:0.04, spec_F:0, spec_G:0, spec_K:0, spec_M:0, spec_m:1 },
    { pl_name:'FrozenGiant-Beta', pl_rade:3.80, pl_bmasse:12.50, pl_orbper:28.50, pl_orbsmax:0.15, pl_eqt:1400, st_teff:3200, st_rad:0.45, st_mass:0.35, st_met:-0.70, st_lum:0.03, temp_score:0.03, dist_score:0.02, lum_score:0.02, stellar_temp_score:0.04, stellar_mass_score:0.03, stellar_compatibility_index:0.03, orbital_stability:0.03, spec_F:0, spec_G:0, spec_K:0, spec_M:0, spec_m:1 },
    { pl_name:'ScorchWorld-Zeta', pl_rade:5.20, pl_bmasse:15.00, pl_orbper:100.0, pl_orbsmax:0.50, pl_eqt:800,  st_teff:8000, st_rad:2.00, st_mass:1.80, st_met: 0.30, st_lum:3.00, temp_score:0.05, dist_score:0.04, lum_score:0.03, stellar_temp_score:0.05, stellar_mass_score:0.04, stellar_compatibility_index:0.03, orbital_stability:0.04, spec_F:1, spec_G:0, spec_K:0, spec_M:0, spec_m:0 },
    { pl_name:'DeadRock-Omega',   pl_rade:0.60, pl_bmasse: 0.30, pl_orbper: 18.20, pl_orbsmax:0.10, pl_eqt:950,  st_teff:4200, st_rad:0.55, st_mass:0.52, st_met:-0.30, st_lum:0.15, temp_score:0.03, dist_score:0.02, lum_score:0.02, stellar_temp_score:0.04, stellar_mass_score:0.03, stellar_compatibility_index:0.03, orbital_stability:0.04, spec_F:0, spec_G:0, spec_K:0, spec_M:1, spec_m:0 },
    { pl_name:'IceGiant-Delta',   pl_rade:2.90, pl_bmasse: 5.50, pl_orbper: 15.00, pl_orbsmax:0.08, pl_eqt:1600, st_teff:3100, st_rad:0.40, st_mass:0.38, st_met:-0.65, st_lum:0.02, temp_score:0.02, dist_score:0.02, lum_score:0.02, stellar_temp_score:0.03, stellar_mass_score:0.02, stellar_compatibility_index:0.02, orbital_stability:0.02, spec_F:0, spec_G:0, spec_K:0, spec_M:0, spec_m:1 },
  ]
};
// ════════════════════════════════════════════════════
//  RANDOM FOREST SIMULATOR  
//  Best Model: Tuned Random Forest
//  Accuracy 98.89% | Precision 100% | Recall 97.5%
// ════════════════════════════════════════════════════

const SCALER = {
  mean: {
    pl_rade:1.85, pl_bmasse:5.2, pl_orbper:380, pl_orbsmax:0.95,
    pl_eqt:550, st_teff:5100, st_rad:0.92, st_mass:0.91, st_met:-0.05,
    st_lum:0.72, spec_F:0.08, spec_G:0.28, spec_K:0.32, spec_M:0.18, spec_m:0.14,
    temp_score:0.45, dist_score:0.48, lum_score:0.44,
    stellar_temp_score:0.52, stellar_mass_score:0.50,
    stellar_compatibility_index:0.46, orbital_stability:0.47
  },
  std: {
    pl_rade:1.60, pl_bmasse:8.5, pl_orbper:550, pl_orbsmax:0.80,
    pl_eqt:380, st_teff:980, st_rad:0.55, st_mass:0.42, st_met:0.22,
    st_lum:0.65, spec_F:0.27, spec_G:0.45, spec_K:0.47, spec_M:0.38, spec_m:0.35,
    temp_score:0.30, dist_score:0.30, lum_score:0.29,
    stellar_temp_score:0.26, stellar_mass_score:0.26,
    stellar_compatibility_index:0.27, orbital_stability:0.28
  }
};

// Calibrated RF feature weights — derived from NASA training data
// Score features dominate (temp_score, dist_score etc.) as they encode physics
// Physical features act as secondary guards
const RF_WEIGHTS = {  // Random Forest feature weights
  // Physical planet params — strong negative guards for extreme values
  pl_rade:            -1.20,   // larger radius → less habitable
  pl_bmasse:          -0.65,   // heavier → less habitable
  pl_orbper:           0.08,
  pl_orbsmax:          0.20,
  pl_eqt:             -2.20,   // HIGH weight — temperature is critical
  // Stellar params
  st_teff:            -1.50,   // too hot/cold star → bad (penalty for extremes)
  st_rad:             -0.15,
  st_mass:             0.08,
  st_met:              0.05,
  st_lum:             -0.25,   // extreme luminosity → bad
  // Spectral type — G/K are best for habitability
  spec_F:             -0.30,
  spec_G:              0.60,
  spec_K:              0.55,
  spec_M:             -0.35,
  spec_m:             -0.70,
  // Habitability score features — DOMINANT signals (same as model training)
  temp_score:          3.80,
  dist_score:          3.50,
  lum_score:           2.20,
  stellar_temp_score:  2.00,
  stellar_mass_score:  1.60,
  stellar_compatibility_index: 2.10,
  orbital_stability:   1.80
};
const RF_INTERCEPT = -4.5;  // RF bias term

const FEATURE_ORDER = [
  'pl_rade','pl_bmasse','pl_orbper','pl_orbsmax','pl_eqt',
  'st_teff','st_rad','st_mass','st_met','st_lum',
  'spec_F','spec_G','spec_K','spec_M','spec_m',
  'temp_score','dist_score','lum_score',
  'stellar_temp_score','stellar_mass_score',
  'stellar_compatibility_index','orbital_stability'
];

function predictLocally(d) {
  // RF model actual feature weights — planet radius & mass dominate (43.5% + 24.4%)
  const rade   = parseFloat(d.pl_rade)   || 1;
  const bmasse = parseFloat(d.pl_bmasse) || 1;
  const eqt    = parseFloat(d.pl_eqt)   || 288;
  const strad  = parseFloat(d.st_rad)   || 1;
  const stmet  = parseFloat(d.st_met)   || 0;
  const stmass = parseFloat(d.st_mass)  || 1;
  const stlum  = parseFloat(d.st_lum)   || 1;
  const steff  = parseFloat(d.st_teff)  || 5772;
  const orbper = parseFloat(d.pl_orbper)|| 365;
  const stellarTempScore = parseFloat(d.stellar_temp_score) || 0;

  // RF-based probability estimate using actual feature importances
  // pl_rade: 43.46%  pl_bmasse: 24.45%  st_rad: 5.04%  st_met: 4.53%  st_mass: 4.39%  st_lum: 4.20%
  // Score starts at 1.0 (optimistic), then penalize for non-habitable features

  // Core RF decision: pl_rade and pl_bmasse are most important
  // Habitable zone: radius 0.5-1.8 R⊕, mass 0.1-5 M⊕
  const radePenalty  = rade  < 0.5 ? 0.6 : rade  > 2.5 ? Math.max(0, 1 - (rade-2.5)*0.5) : 1.0;
  const massePenalty = bmasse < 0.1 ? 0.5 : bmasse > 5.0 ? Math.max(0, 1 - (bmasse-5)*0.15) : 1.0;

  // Temperature: 220-340K habitable zone
  const tempScore = eqt >= 220 && eqt <= 340 ? 1.0
    : eqt > 340 && eqt <= 500 ? 1 - (eqt-340)/320
    : eqt < 220 && eqt >= 150 ? 1 - (220-eqt)/140
    : eqt > 500 ? Math.max(0, 1 - (eqt-500)/800)
    : Math.max(0, 1 - (150-eqt)/150);

  // Stellar params
  const radScore  = strad  >= 0.7 && strad  <= 1.5 ? 1.0 : Math.max(0, 1 - Math.abs(strad -1.0)*0.6);
  const metScore  = stmet  >= -0.5 && stmet <= 0.5  ? 1.0 : Math.max(0, 1 - Math.abs(stmet)*0.5);
  const massScore = stmass >= 0.7 && stmass <= 1.4  ? 1.0 : Math.max(0, 1 - Math.abs(stmass-1.0)*0.8);
  const lumScore2 = stlum  >= 0.5 && stlum  <= 1.8  ? 1.0 : Math.max(0, 1 - Math.abs(Math.log(stlum+0.01))*0.5);
  const steffScore= steff  >= 4500 && steff <= 7000 ? 1.0 : Math.max(0, 1 - Math.abs(steff-5778)/4000);

  // Derived scores from input
  const dScore = parseFloat(d.dist_score)||0;
  const lScore = parseFloat(d.lum_score)||0;
  const orbStab= parseFloat(d.orbital_stability)||0;
  const distScore = dScore > 0 ? dScore : (stellarTempScore > 0 ? stellarTempScore : 0.5);

  // Weighted sum matching RF feature importances
  const probHab = Math.min(0.99, Math.max(0.01,
    radePenalty  * 0.4346 +
    massePenalty * 0.2445 +
    radScore     * 0.0504 +
    metScore     * 0.0453 +
    massScore    * 0.0439 +
    lumScore2    * 0.0420 +
    tempScore    * 0.0205 +
    (orbper>200&&orbper<700?1:0.4) * 0.0190 +
    steffScore   * 0.0189 +
    stellarTempScore * 0.0170 +
    distScore    * 0.0157 +
    orbStab      * 0.0154
  ));

  // app.py threshold: 0.9 — only predict habitable if prob >= 0.9
  const isHabitable = probHab >= 0.9;
  const confidence  = isHabitable ? probHab : (1 - probHab);

  return {
    prediction_result: isHabitable ? 1 : 0,
    confidence_score:  parseFloat(confidence.toFixed(4)),
    prob_habitable:    parseFloat(probHab.toFixed(4)),
    local: true
  };
}

function _nonHab(d, probHab) {
  return {
    prediction_result: 0,
    confidence_score:  parseFloat((1 - probHab).toFixed(4)),   // high confidence it's NOT habitable
    prob_habitable:    parseFloat(probHab.toFixed(4)),
    local: true
  };
}

// ── Page navigation (single page — smooth scroll) ──────────────
function showPage(pageId) {
  const el = document.getElementById(pageId);
  if (el) el.scrollIntoView({ behavior: 'smooth' });
  if (pageId === 'ranking')   loadRankingData();
  if (pageId === 'analytics') anlRefresh();
}

document.addEventListener('DOMContentLoaded', () => {
  // Load ranking on startup
  loadRankingData();
  const form = document.getElementById('predictionForm');
  if (form) {
    form.addEventListener('submit', handleSinglePrediction);
    ['pl_rade','pl_eqt','pl_orbper','st_teff','pl_bmasse'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', updateVisualIndicators);
    });
  }
  // Load analytics history
  try { _anlHistory = JSON.parse(localStorage.getItem('exo_anl_v2') || localStorage.getItem('exo_anl') || '[]'); } catch(e){}
  const badge = document.getElementById('anlNavBadge');
  if (badge && _anlHistory.length) { badge.textContent = _anlHistory.length; badge.style.display = 'inline-block'; }
  // Init analytics charts
  anlRefresh();
});

// ════════════════════════════════════════════════════
//  SINGLE PREDICTION
// ════════════════════════════════════════════════════
async function handleSinglePrediction(e) {
  e.preventDefault();
  const btn = document.getElementById('predictBtn');
  setLoadingState(btn, true);

  try {
    const data = collectFormData();
    const valid = validateFormData(data);
    if (!valid.isValid) { showFormError(valid.message); return; }

    let result;
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(data), signal: AbortSignal.timeout(4000)
      });
      result = await res.json();
      if (!res.ok) throw new Error(result.message);
    } catch {
      // API unavailable — use local prediction
      result = predictLocally(data);
    }

    displaySingleResults(result);
  } finally {
    setLoadingState(btn, false);
  }
}

function collectFormData() {
  const data = {};
  // Sirf wahi 22 features jo utils.py ka FEATURES list mein hain
  const numericFields = [
    'pl_rade','pl_bmasse','pl_orbper','pl_orbsmax','pl_eqt',
    'st_teff','st_rad','st_mass','st_met','st_lum',
    'temp_score','dist_score','lum_score','stellar_temp_score',
    'stellar_mass_score','stellar_compatibility_index','orbital_stability'
  ];
  numericFields.forEach(f => {
    const el = document.getElementById(f);
    if (el && el.value !== '') data[f] = parseFloat(el.value);
  });
  ['spec_F','spec_G','spec_K','spec_M','spec_m'].forEach(f => {
    const el = document.getElementById(f);
    data[f] = el && el.checked ? 1 : 0;
  });
  return data;
}

// ════════════════════════════════════════════════════
//  BATCH — JSON tab
// ════════════════════════════════════════════════════
async function runJSONBatch() {
  const textarea = document.getElementById('jsonInput');
  const raw = (textarea?.value || '').trim();
  if (!raw) { showBatchError('JSON textarea mein data paste karo ya sample load karo.'); return; }

  let planets;
  try {
    const parsed = JSON.parse(raw);
    planets = Array.isArray(parsed) ? parsed : (parsed.planets || []);
    if (!planets.length) throw new Error('empty');
  } catch {
    showBatchError('Invalid JSON format. Expected: {"planets":[{...}]} ya array of objects.');
    return;
  }

  await runBatchOnData(planets, 'json');
}

// ════════════════════════════════════════════════════
//  BATCH — CSV tab
// ════════════════════════════════════════════════════
async function handleBatchFromUI() {
  const fileInput = document.getElementById('batchFile');
  if (!fileInput?.files[0]) { showBatchError('Pehle CSV file select karo.'); return; }

  const text = await fileInput.files[0].text();
  const rows = text.split('\n').filter(r => r.trim());
  if (rows.length < 2) { showBatchError('CSV mein header + at least 1 data row hona chahiye.'); return; }

  const headers = rows[0].split(',').map(h => h.trim());
  const planets = rows.slice(1).map(row => {
    const vals = row.split(',').map(c => c.trim());
    const obj = {};
    headers.forEach((h, i) => {
      if (vals[i] !== undefined && vals[i] !== '') {
        obj[h] = isNaN(vals[i]) ? vals[i] : parseFloat(vals[i]);
      }
    });
    return obj;
  }).filter(o => Object.keys(o).length > 0);

  await runBatchOnData(planets, 'csv');
}

// ── Core batch runner ─────────────────────────────────────────
async function runBatchOnData(planets, source) {
  const btn = source === 'json'
    ? document.getElementById('jsonBatchBtn')
    : document.getElementById('batchBtn');

  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...'; }

  // Show progress
  showBatchProgress(0, planets.length);

  const results = [];
  let successCount = 0;

  for (let i = 0; i < planets.length; i++) {
    const planet = planets[i];
    showBatchProgress(i + 1, planets.length);

    let result;
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(planet), signal: AbortSignal.timeout(3000)
      });
      result = await res.json();
      if (!res.ok) throw new Error(result.message);
      results.push({
        row: i + 1,
        name: planet.pl_name || `Planet ${i+1}`,
        habitability: result.prediction_result === 1 ? 'Habitable' : 'Non-Habitable',
        confidence: result.confidence_score != null ? (result.confidence_score * 100).toFixed(1) + '%' : 'N/A',
        status: 'API ✓',
        isHabitable: result.prediction_result === 1
      });
      successCount++;
    } catch {
      // Flask offline — use RF simulator fallback
      const local = predictLocally(planet);
      results.push({
        row: i + 1,
        name: planet.pl_name || `Planet ${i+1}`,
        habitability: local.prediction_result === 1 ? 'Habitable' : 'Non-Habitable',
        confidence:   (local.confidence_score * 100).toFixed(1) + '%',
        status:       'RF Sim',   // RF Simulator — not the real trained model
        isHabitable:  local.prediction_result === 1
      });
      successCount++;
    }
  }

  displayBatchResults(results, successCount);
  if (btn) { btn.disabled = false; btn.innerHTML = source === 'json' ? '<i class="fas fa-rocket"></i> Run Batch Analysis' : '<i class="fas fa-layer-group"></i> Process Batch'; }
}

// ── Batch progress bar ────────────────────────────────────────
function showBatchProgress(done, total) {
  const batchResults = document.getElementById('batchResults');
  if (batchResults) batchResults.style.display = 'block';
  let prog = document.getElementById('batchProgress');
  if (!prog) {
    prog = document.createElement('div');
    prog.id = 'batchProgress';
    prog.innerHTML = `
      <div style="background:rgba(0,212,170,.08);border:1px solid rgba(0,212,170,.2);border-radius:8px;padding:1rem 1.5rem;margin-bottom:1rem">
        <div style="display:flex;justify-content:space-between;margin-bottom:.5rem">
          <span style="font-size:.82rem;color:#8ec4dc">Processing planets...</span>
          <span id="progText" style="font-size:.82rem;color:var(--teal);font-family:var(--fh)">0 / 0</span>
        </div>
        <div style="background:rgba(0,0,0,.4);border-radius:99px;height:6px;overflow:hidden">
          <div id="progBar" style="height:100%;background:linear-gradient(90deg,var(--teal),var(--blue));border-radius:99px;transition:width .3s;width:0%"></div>
        </div>
      </div>`;
    if (batchResults) batchResults.insertBefore(prog, batchResults.firstChild);
  }
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const bar = document.getElementById('progBar');
  const txt = document.getElementById('progText');
  if (bar) bar.style.width = pct + '%';
  if (txt) txt.textContent = `${done} / ${total}`;
  if (done >= total) setTimeout(() => { if(prog.parentNode) prog.remove(); }, 1200);
}

function showBatchError(msg) {
  const batchResults = document.getElementById('batchResults');
  if (batchResults) batchResults.style.display = 'block';
  const existing = document.getElementById('batchErrBanner');
  if (existing) existing.remove();
  const div = document.createElement('div');
  div.id = 'batchErrBanner';
  div.style.cssText = 'background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.3);color:#f43f5e;border-radius:6px;padding:1rem 1.5rem;margin-bottom:1rem;font-size:.88rem';
  div.innerHTML = `<i class="fas fa-exclamation-triangle"></i>  ${msg}`;
  if (batchResults) batchResults.insertBefore(div, batchResults.firstChild);
  if (batchResults) batchResults.scrollIntoView({ behavior:'smooth' });
}

// ── Display batch results ─────────────────────────────────────
function displayBatchResults(results, _unused) {
  const batchResults = document.getElementById('batchResults');
  const batchBody    = document.getElementById('batchBody');
  const batchSummary = document.getElementById('batchSummary');
  const totalBadge   = document.getElementById('batchTotalBadge');

  if (!batchResults || !batchBody || !batchSummary) return;

  const habitable    = results.filter(r => r.isHabitable === true).length;
  const nonHabitable = results.filter(r => r.isHabitable === false).length;
  const total        = results.length;

  batchSummary.innerHTML = `
    <div class="sum-b"><h6>Total</h6><span style="color:var(--white)">${total}</span></div>
    <div class="sum-b"><h6>Habitable</h6><span style="color:var(--green)">${habitable}</span></div>
    <div class="sum-b"><h6>Non-Habitable</h6><span style="color:var(--red)">${nonHabitable}</span></div>
    <div class="sum-b"><h6>Analyzed</h6><span style="color:var(--teal)">${total}/${total}</span></div>`;

  if (totalBadge) totalBadge.textContent = `Showing top ${Math.min(4, total)} of ${total}`;

  // Show only top 4 results
  batchBody.innerHTML = results.slice(0, 4).map(r => `
    <tr>
      <td style="color:var(--amber);font-weight:700;font-family:var(--fh)">#${r.row}</td>
      <td style="color:#c8e4f4;font-size:.86rem">${r.name}</td>
      <td><span class="${r.isHabitable ? 'bg' : 'br'}">${r.habitability}</span></td>
      <td style="color:var(--teal);font-weight:700;font-family:var(--fh)">${r.confidence}</td>
    </tr>`).join('');

  batchResults.style.display = 'block';
  batchResults.scrollIntoView({ behavior: 'smooth' });
}

// ════════════════════════════════════════════════════
//  LOAD SAMPLE DATA — JSON tab
// ════════════════════════════════════════════════════
function loadSampleJSON(type) {
  const data = type === 'habitable' ? SAMPLE_DATA.habitable : SAMPLE_DATA.nonHabitable;
  document.getElementById('jsonInput').value = JSON.stringify({ planets: data }, null, 2);
  // Animate textarea
  const ta = document.getElementById('jsonInput');
  ta.style.borderColor = 'var(--teal)';
  ta.style.boxShadow   = '0 0 20px rgba(0,212,170,.2)';
  setTimeout(() => { ta.style.borderColor=''; ta.style.boxShadow=''; }, 1500);
}

// ════════════════════════════════════════════════════
//  LOAD SAMPLE DATA — CSV tab (load + auto-process)
// ════════════════════════════════════════════════════
function loadSampleCSVData(type) {
  const rows  = type === 'habitable' ? SAMPLE_DATA.habitable : SAMPLE_DATA.nonHabitable;
  const headers = ['pl_rade','pl_bmasse','pl_orbper','pl_orbsmax','pl_eqt','st_teff','st_rad','st_mass','st_met','st_lum','temp_score','dist_score','lum_score','stellar_temp_score','stellar_mass_score','stellar_compatibility_index','orbital_stability','spec_F','spec_G','spec_K','spec_M','spec_m','pl_name'];
  let csv = headers.join(',') + '\n';
  rows.forEach(r => { csv += headers.map(h => r[h] !== undefined ? r[h] : '').join(',') + '\n'; });

  try {
    const blob = new Blob([csv], { type:'text/csv' });
    const file = new File([blob], `sample_${type}.csv`, { type:'text/csv' });
    const dt = new DataTransfer();
    dt.items.add(file);
    document.getElementById('batchFile').files = dt.files;

    // Update drop zone label with animated feedback
    const lbl = document.getElementById('dropLabel');
    if (lbl) {
      lbl.textContent = `✓  sample_${type}.csv  (${rows.length} planets loaded)`;
      lbl.style.color = 'var(--teal)';
    }
    const dz = document.getElementById('dropZone');
    if (dz) {
      dz.style.borderColor = 'var(--teal)';
      dz.style.background  = 'rgba(0,212,170,.06)';
      setTimeout(() => { dz.style.borderColor=''; dz.style.background=''; }, 1500);
    }

    // Auto-run batch after short delay
    setTimeout(() => handleBatchFromUI(), 300);
  } catch(err) {
    // Fallback: run directly from SAMPLE_DATA array
    runBatchOnData(rows, 'csv');
  }
}

// ════════════════════════════════════════════════════
//  RANKING
// ════════════════════════════════════════════════════
async function loadRankingData() {
  const limitVal  = parseInt(document.getElementById('rankLimit')?.value) || 22;
  const minScore  = parseFloat(document.getElementById('minScore')?.value) || 0;
  const statusVal = document.getElementById('rankStatus')?.value || 'all';
  const loadEl    = document.getElementById('rankingLoading');
  const tableWr   = document.getElementById('rankingTableWrap');

  if (loadEl) loadEl.style.display = 'block';
  if (tableWr) tableWr.style.display = 'none';

  try {
    const res = await fetch(`${API_BASE_URL}/rank?limit=${limitVal}&min_score=${minScore}`,
      { signal: AbortSignal.timeout(4000) });
    const data = await res.json();
    if (res.ok) { displayRankingData(data.data, statusVal, limitVal, minScore); return; }
    throw new Error(data.message);
  } catch {
    // ── Full 25-planet local pool (habitable + non-habitable) ──
    const pool = [
      // Habitable candidates
      { pl_name:'Kepler-442b',      pl_rade:1.20, pl_bmasse:1.00, pl_orbper:365.3, pl_orbsmax:1.00, pl_eqt:288,  st_teff:5772, temp_score:0.95,dist_score:0.98,lum_score:0.92,stellar_temp_score:0.90,stellar_mass_score:0.88,stellar_compatibility_index:0.92,orbital_stability:0.95 },
      { pl_name:'ExoEarth-Alpha',   pl_rade:1.50, pl_bmasse:1.20, pl_orbper:450.0, pl_orbsmax:1.10, pl_eqt:275,  st_teff:5900, temp_score:0.92,dist_score:0.95,lum_score:0.90,stellar_temp_score:0.88,stellar_mass_score:0.85,stellar_compatibility_index:0.90,orbital_stability:0.93 },
      { pl_name:'GreenWorld-7',     pl_rade:1.10, pl_bmasse:0.90, pl_orbper:330.0, pl_orbsmax:0.95, pl_eqt:295,  st_teff:5600, temp_score:0.94,dist_score:0.97,lum_score:0.91,stellar_temp_score:0.91,stellar_mass_score:0.87,stellar_compatibility_index:0.91,orbital_stability:0.94 },
      { pl_name:'Proxima-Cen-d',    pl_rade:1.35, pl_bmasse:1.10, pl_orbper:390.0, pl_orbsmax:1.05, pl_eqt:280,  st_teff:5800, temp_score:0.91,dist_score:0.93,lum_score:0.89,stellar_temp_score:0.87,stellar_mass_score:0.86,stellar_compatibility_index:0.89,orbital_stability:0.92 },
      { pl_name:'Kepler-62f',       pl_rade:0.95, pl_bmasse:0.85, pl_orbper:312.0, pl_orbsmax:0.92, pl_eqt:302,  st_teff:5500, temp_score:0.89,dist_score:0.91,lum_score:0.87,stellar_temp_score:0.85,stellar_mass_score:0.84,stellar_compatibility_index:0.88,orbital_stability:0.90 },
      { pl_name:'TRAPPIST-1e',      pl_rade:1.60, pl_bmasse:1.30, pl_orbper:480.0, pl_orbsmax:1.15, pl_eqt:268,  st_teff:6000, temp_score:0.87,dist_score:0.90,lum_score:0.85,stellar_temp_score:0.83,stellar_mass_score:0.82,stellar_compatibility_index:0.86,orbital_stability:0.88 },
      { pl_name:'K2-18b',           pl_rade:1.05, pl_bmasse:0.95, pl_orbper:345.0, pl_orbsmax:0.98, pl_eqt:291,  st_teff:5700, temp_score:0.86,dist_score:0.88,lum_score:0.84,stellar_temp_score:0.82,stellar_mass_score:0.81,stellar_compatibility_index:0.85,orbital_stability:0.87 },
      { pl_name:'LHS-1140b',        pl_rade:1.45, pl_bmasse:1.15, pl_orbper:420.0, pl_orbsmax:1.08, pl_eqt:278,  st_teff:5850, temp_score:0.85,dist_score:0.87,lum_score:0.83,stellar_temp_score:0.81,stellar_mass_score:0.80,stellar_compatibility_index:0.84,orbital_stability:0.86 },
      { pl_name:'Wolf-1061c',       pl_rade:1.25, pl_bmasse:1.05, pl_orbper:375.0, pl_orbsmax:1.02, pl_eqt:285,  st_teff:5750, temp_score:0.83,dist_score:0.86,lum_score:0.81,stellar_temp_score:0.79,stellar_mass_score:0.78,stellar_compatibility_index:0.82,orbital_stability:0.84 },
      { pl_name:'Kepler-186f',      pl_rade:0.88, pl_bmasse:0.80, pl_orbper:298.0, pl_orbsmax:0.89, pl_eqt:308,  st_teff:5450, temp_score:0.82,dist_score:0.84,lum_score:0.80,stellar_temp_score:0.78,stellar_mass_score:0.77,stellar_compatibility_index:0.81,orbital_stability:0.83 },
      { pl_name:'Gliese-667Cc',     pl_rade:1.70, pl_bmasse:1.40, pl_orbper:510.0, pl_orbsmax:1.20, pl_eqt:262,  st_teff:6100, temp_score:0.80,dist_score:0.83,lum_score:0.78,stellar_temp_score:0.76,stellar_mass_score:0.75,stellar_compatibility_index:0.79,orbital_stability:0.81 },
      { pl_name:'HD-40307g',        pl_rade:1.15, pl_bmasse:0.92, pl_orbper:355.0, pl_orbsmax:0.97, pl_eqt:293,  st_teff:5650, temp_score:0.79,dist_score:0.81,lum_score:0.77,stellar_temp_score:0.75,stellar_mass_score:0.74,stellar_compatibility_index:0.78,orbital_stability:0.80 },
      { pl_name:'Tau-Ceti-e',       pl_rade:1.40, pl_bmasse:1.12, pl_orbper:405.0, pl_orbsmax:1.06, pl_eqt:276,  st_teff:5820, temp_score:0.55,dist_score:0.60,lum_score:0.58,stellar_temp_score:0.52,stellar_mass_score:0.53,stellar_compatibility_index:0.55,orbital_stability:0.57 },
      // Non-habitable planets
      { pl_name:'HotJupiter-X1',    pl_rade:4.50, pl_bmasse:8.20,  pl_orbper:45.3, pl_orbsmax:0.20, pl_eqt:1200, st_teff:3500, temp_score:0.15,dist_score:0.10,lum_score:0.08,stellar_temp_score:0.20,stellar_mass_score:0.18,stellar_compatibility_index:0.12,orbital_stability:0.10 },
      { pl_name:'FrozenGiant-Beta', pl_rade:3.80, pl_bmasse:12.5,  pl_orbper:28.5, pl_orbsmax:0.15, pl_eqt:1400, st_teff:3200, temp_score:0.12,dist_score:0.08,lum_score:0.05,stellar_temp_score:0.18,stellar_mass_score:0.15,stellar_compatibility_index:0.10,orbital_stability:0.08 },
      { pl_name:'ScorchWorld-Zeta', pl_rade:5.20, pl_bmasse:15.0,  pl_orbper:100,  pl_orbsmax:0.50, pl_eqt:800,  st_teff:8000, temp_score:0.20,dist_score:0.15,lum_score:0.12,stellar_temp_score:0.25,stellar_mass_score:0.22,stellar_compatibility_index:0.15,orbital_stability:0.18 },
      { pl_name:'DeadRock-9',       pl_rade:0.60, pl_bmasse:0.30,  pl_orbper:18.2, pl_orbsmax:0.10, pl_eqt:950,  st_teff:4200, temp_score:0.10,dist_score:0.08,lum_score:0.06,stellar_temp_score:0.22,stellar_mass_score:0.14,stellar_compatibility_index:0.09,orbital_stability:0.12 },
      { pl_name:'GasGiant-Omicron', pl_rade:6.10, pl_bmasse:20.0,  pl_orbper:220,  pl_orbsmax:0.80, pl_eqt:650,  st_teff:7200, temp_score:0.18,dist_score:0.12,lum_score:0.10,stellar_temp_score:0.16,stellar_mass_score:0.20,stellar_compatibility_index:0.13,orbital_stability:0.15 },
      { pl_name:'IceGiant-Delta',   pl_rade:2.90, pl_bmasse:5.50,  pl_orbper:15.0, pl_orbsmax:0.08, pl_eqt:1600, st_teff:3100, temp_score:0.08,dist_score:0.06,lum_score:0.04,stellar_temp_score:0.12,stellar_mass_score:0.10,stellar_compatibility_index:0.07,orbital_stability:0.06 },
      { pl_name:'PulsarWorld-Sigma',pl_rade:1.80, pl_bmasse:2.20,  pl_orbper:8.50, pl_orbsmax:0.06, pl_eqt:2200, st_teff:9500, temp_score:0.05,dist_score:0.04,lum_score:0.03,stellar_temp_score:0.08,stellar_mass_score:0.06,stellar_compatibility_index:0.04,orbital_stability:0.05 },
      { pl_name:'TidalLock-Rho',    pl_rade:1.30, pl_bmasse:1.40,  pl_orbper:12.8, pl_orbsmax:0.09, pl_eqt:750,  st_teff:3800, temp_score:0.22,dist_score:0.18,lum_score:0.14,stellar_temp_score:0.28,stellar_mass_score:0.24,stellar_compatibility_index:0.19,orbital_stability:0.20 },
      { pl_name:'VolcanicHell-Pi',  pl_rade:0.95, pl_bmasse:0.75,  pl_orbper:25.0, pl_orbsmax:0.12, pl_eqt:900,  st_teff:4800, temp_score:0.16,dist_score:0.14,lum_score:0.11,stellar_temp_score:0.30,stellar_mass_score:0.26,stellar_compatibility_index:0.17,orbital_stability:0.22 },
      { pl_name:'MiniNeptune-Lam',  pl_rade:3.20, pl_bmasse:6.80,  pl_orbper:55.0, pl_orbsmax:0.30, pl_eqt:480,  st_teff:5100, temp_score:0.30,dist_score:0.25,lum_score:0.22,stellar_temp_score:0.35,stellar_mass_score:0.28,stellar_compatibility_index:0.27,orbital_stability:0.32 },
    ];

    // Compute scores and sort
    const scored = pool.map((p, i) => {
      const score = p.temp_score*0.22 + p.dist_score*0.20 + p.lum_score*0.14 +
                    p.stellar_temp_score*0.12 + p.stellar_mass_score*0.10 +
                    p.stellar_compatibility_index*0.12 + p.orbital_stability*0.10;
      return { ...p, Habitability_Score: score };
    }).sort((a, b) => b.Habitability_Score - a.Habitability_Score)
      .map((p, i) => ({ ...p, Rank: i + 1 }));

    // Filter by minScore and status
    let filtered = scored.filter(p => p.Habitability_Score >= minScore);
    if (statusVal === 'habitable')    filtered = filtered.filter(p => p.Habitability_Score >= 0.50);
    if (statusVal === 'nonhabitable') filtered = filtered.filter(p => p.Habitability_Score <  0.50);

    // Apply limit
    const finalList = limitVal < 100 ? filtered.slice(0, limitVal) : filtered;
    displayRankingData(finalList, statusVal, limitVal, minScore);
  }
}

function displayRankingData(data) {
  const loadEl  = document.getElementById('rankingLoading');
  const tableWr = document.getElementById('rankingTableWrap');
  const tbody   = document.getElementById('rankingBody');
  const countEl = document.getElementById('rankResultCount');

  if (loadEl)  loadEl.style.display  = 'none';
  if (tableWr) tableWr.style.display = 'block';

  if (!data?.length) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:#8ec4dc;padding:2rem;font-size:.9rem">No planets found matching the filter criteria</td></tr>`;
    if (countEl) countEl.textContent = '0 planets found';
    return;
  }

  if (countEl) {
    const habCount = data.filter(p => (p.Habitability_Score || 0) >= 0.50).length;
    countEl.innerHTML = `Showing <strong style="color:#c8e8f5">${data.length}</strong> planets &nbsp;·&nbsp; <span style="color:#10b981">✓ ${habCount} Habitable</span> &nbsp;·&nbsp; <span style="color:#f43f5e">✗ ${data.length - habCount} Non-Hab</span>`;
  }

  tbody.innerHTML = data.map((p, i) => {
    const score    = p.Habitability_Score || 0;
    const isH      = score >= 0.50;
    const rankN    = p.Rank || i + 1;
    const medal    = rankN === 1 ? '🥇' : rankN === 2 ? '🥈' : rankN === 3 ? '🥉' : '';
    const scoreCol = score >= 0.80 ? '#10b981' : score >= 0.60 ? '#00d4aa' : score >= 0.40 ? '#f59e0b' : '#f43f5e';
    const name     = p.pl_name || `Planet-${rankN}`;
    return `<tr>
      <td style="font-family:var(--fh);font-weight:700;color:var(--amber);white-space:nowrap">${medal} #${rankN}</td>
      <td style="color:#c8e8f5;font-weight:600;max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${name}">${name}</td>
      <td style="color:#c8e8f5">${p.pl_rade    != null ? p.pl_rade.toFixed(2)   : 'N/A'} R⊕</td>
      <td style="color:#c8e8f5">${p.pl_bmasse  != null ? p.pl_bmasse.toFixed(2)  : 'N/A'} M⊕</td>
      <td style="color:#c8e8f5">${p.pl_orbper  != null ? p.pl_orbper.toFixed(1)  : 'N/A'} d</td>
      <td style="color:#c8e8f5">${p.pl_eqt     != null && p.pl_eqt !== 0 ? p.pl_eqt.toFixed(0) + ' K' : (p.pl_eqt === 0 ? '0 K' : 'N/A')}</td>
      <td style="color:#c8e8f5">${p.st_teff    != null ? p.st_teff.toFixed(0) + ' K' : 'N/A'}</td>
      <td style="color:${scoreCol};font-weight:700;font-family:var(--fh)">${(score*100).toFixed(1)}%</td>
      <td><span class="${isH ? 'bg' : 'br'}" style="white-space:nowrap">${isH ? '✓ Habitable' : '✗ Non-Hab'}</span></td>
    </tr>`;
  }).join('');
}

// ════════════════════════════════════════════════════
//  SINGLE PREDICTION DISPLAY  (full 5-feature upgrade)
// ════════════════════════════════════════════════════
function displaySingleResults(result) {
  const div   = document.getElementById('results');
  const alert = document.getElementById('resultAlert');
  const hScore= document.getElementById('habitableScore');
  const nScore= document.getElementById('nonHabitableScore');

  div.classList.remove('d-none');
  const isH = result.prediction_result === 1;

  // prob_habitable = P(habitable) — ALWAYS 0-1, used for Hab Score / Non-Hab Score
  // confidence_score = certainty of prediction = P(habitable) if hab, (1-P) if non-hab
  const pHab       = result.prob_habitable    ?? result.confidence_score ?? (isH ? 0.93 : 0.07);
  const confidence = result.confidence_score  ?? (isH ? pHab : 1 - pHab);

  // Hab Score = P(habitable), Non-Hab Score = P(non-habitable)
  const hPct = (pHab * 100).toFixed(1);
  const nPct = ((1 - pHab) * 100).toFixed(1);

  hScore.textContent = hPct + '%';
  nScore.textContent = nPct + '%';

  const sourceTag = result.local
    ? ' &nbsp;<span style="opacity:.55;font-size:.8em">(RF Simulator — Flask band hai, python app.py chalao)</span>'
    : '';
  alert.className = `res-banner ${isH ? 'res-ok' : 'res-err'}`;
  alert.innerHTML = `<i class="fas fa-${isH ? 'earth-americas' : 'times-circle'}"></i>&nbsp; This exoplanet is predicted to be <strong>${isH ? 'HABITABLE 🌍' : 'NOT HABITABLE ✗'}</strong>${sourceTag}`;

  // ── Save to Analytics ──
  const _eqt   = parseFloat(document.getElementById('pl_eqt')?.value)    || 0;
  const _rade  = parseFloat(document.getElementById('pl_rade')?.value)    || 1;
  const _steff = parseFloat(document.getElementById('st_teff')?.value)    || 5772;
  const _bmasse= parseFloat(document.getElementById('pl_bmasse')?.value)  || 1;
  const _orbper= parseFloat(document.getElementById('pl_orbper')?.value)  || 365;
  const _scores = {
    'Temp Score':    parseFloat(document.getElementById('temp_score')?.value)                   || 0,
    'Dist Score':    parseFloat(document.getElementById('dist_score')?.value)                   || 0,
    'Lum Score':     parseFloat(document.getElementById('lum_score')?.value)                    || 0,
    'Stellar Temp':  parseFloat(document.getElementById('stellar_temp_score')?.value)           || 0,
    'Stellar Mass':  parseFloat(document.getElementById('stellar_mass_score')?.value)           || 0,
    'Compatibility': parseFloat(document.getElementById('stellar_compatibility_index')?.value)  || 0,
    'Orbital Stab':  parseFloat(document.getElementById('orbital_stability')?.value)            || 0,
  };

  // habScore = P(habitable) always (0–1 raw probability)
  // conf     = certainty of the prediction (what the confidence meter shows)
  const habScore = pHab;
  const _orbsmax = parseFloat(document.getElementById('pl_orbsmax')?.value) || 1;
  const _specF = document.getElementById('spec_F')?.checked ? 1 : 0;
  const _specG = document.getElementById('spec_G')?.checked ? 1 : 0;
  const _specK = document.getElementById('spec_K')?.checked ? 1 : 0;
  const _specM = document.getElementById('spec_M')?.checked ? 1 : 0;
  const _specm = document.getElementById('spec_m')?.checked ? 1 : 0;
  anlSaveEntry({ isH, conf: confidence, rade: _rade, eqt: _eqt, steff: _steff,
    bmasse: _bmasse, orbper: _orbper, orbsmax: _orbsmax, mass: _bmasse,
    specF: _specF, specG: _specG, specK: _specK, specM: _specM, specm: _specm,
    scores: _scores, habScore });

  // ── Gather raw inputs for visuals ──
  const eqt   = _eqt;
  const rade  = _rade;
  const steff = _steff;
  const scores = _scores;

  // 1. 3D Planet
  draw3DPlanet(isH, rade, eqt, steff);

  // 2. Temperature Zone
  updateTempZone(eqt);

  // 3. Confidence Meter — pass confidence (certainty of prediction), not P(habitable)
  updateConfidence(confidence, isH);

  // 4. Score Breakdown bars
  renderScoreBars(scores);

  div.scrollIntoView({ behavior: 'smooth' });
}

// ── 3D Planet Visualization ──────────────────────────
let p3anim = null;
function draw3DPlanet(isHabitable, radius, eqt, steff) {
  const canvas = document.getElementById('planetVis3D');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (p3anim) cancelAnimationFrame(p3anim);

  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const pr = Math.min(W, H) * 0.28 * Math.min(Math.max(radius, 0.5), 3) / 1.5;

  // Planet color based on habitability + temp
  let col1, col2, atmoCol, ringCol = null;
  if (isHabitable) {
    col1 = '#1a6b3a'; col2 = '#0d3b20'; atmoCol = 'rgba(0,212,170,0.22)';
  } else if (eqt > 600) {
    col1 = '#8b1a00'; col2 = '#3d0800'; atmoCol = 'rgba(255,80,0,0.25)';
  } else if (eqt < 150) {
    col1 = '#1a3a6b'; col2 = '#0d1f3b'; atmoCol = 'rgba(150,200,255,0.20)';
    ringCol = 'rgba(180,220,255,0.18)';
  } else if (radius > 2.5) {
    col1 = '#6b4a1a'; col2 = '#3b2008'; atmoCol = 'rgba(255,180,60,0.18)';
    ringCol = 'rgba(200,150,80,0.22)';
  } else {
    col1 = '#3a2a6b'; col2 = '#1a1035'; atmoCol = 'rgba(120,80,255,0.18)';
  }

  // Planet type label
  const typeEl = document.getElementById('planetTypeLabel');
  if (typeEl) {
    if (isHabitable)        typeEl.textContent = '🌍 Earth-like';
    else if (radius > 2.5)  typeEl.textContent = '🪐 Gas Giant';
    else if (eqt > 600)     typeEl.textContent = '🔥 Lava World';
    else if (eqt < 150)     typeEl.textContent = '❄️ Ice World';
    else                    typeEl.textContent = '🪨 Rocky Planet';
  }

  let clouds = [];
  for (let i = 0; i < 8; i++) {
    clouds.push({ a: Math.random() * Math.PI * 2, lat: (Math.random() - 0.5) * 1.4,
                  len: 0.15 + Math.random() * 0.2, spd: 0.003 + Math.random() * 0.003 });
  }

  let t = 0;
  function frame() {
    ctx.clearRect(0, 0, W, H);
    t += 0.012;

    // Star glow from top-left
    const sg = ctx.createRadialGradient(cx * 0.3, cy * 0.3, 0, cx * 0.3, cy * 0.3, pr * 2.2);
    sg.addColorStop(0, 'rgba(255,220,120,0.06)');
    sg.addColorStop(1, 'transparent');
    ctx.fillStyle = sg; ctx.fillRect(0, 0, W, H);

    // Atmosphere glow
    const atmo = ctx.createRadialGradient(cx, cy, pr * 0.85, cx, cy, pr * 1.32);
    atmo.addColorStop(0, atmoCol);
    atmo.addColorStop(1, 'transparent');
    ctx.fillStyle = atmo; ctx.beginPath();
    ctx.arc(cx, cy, pr * 1.32, 0, Math.PI * 2); ctx.fill();

    // Planet body
    const pg = ctx.createRadialGradient(cx - pr * 0.28, cy - pr * 0.28, pr * 0.05, cx, cy, pr);
    pg.addColorStop(0, col1); pg.addColorStop(0.6, col2);
    pg.addColorStop(1, '#000');
    ctx.beginPath(); ctx.arc(cx, cy, pr, 0, Math.PI * 2);
    ctx.fillStyle = pg; ctx.fill();

    // Surface bands (latitude lines)
    ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, pr, 0, Math.PI * 2); ctx.clip();
    for (let i = 0; i < 6; i++) {
      const yb = cy - pr + (pr * 2 * i / 5);
      const wb = Math.sqrt(Math.max(0, pr * pr - (yb - cy) ** 2));
      if (wb < 4) continue;
      ctx.beginPath();
      ctx.ellipse(cx + Math.sin(t * 0.4 + i) * pr * 0.06, yb, wb * 0.96, pr * 0.07, 0, 0, Math.PI * 2);
      ctx.fillStyle = isHabitable
        ? `rgba(30,160,80,${0.08 + 0.04 * Math.sin(t + i)})`
        : `rgba(255,180,60,${0.05 + 0.03 * Math.sin(t + i)})`;
      ctx.fill();
    }
    // Clouds
    clouds.forEach(cl => {
      cl.a += cl.spd;
      const cx2 = cx + pr * Math.cos(cl.a) * 0.85;
      const cy2 = cy + pr * Math.sin(cl.lat) * 0.9 * Math.cos(cl.lat);
      const cw  = pr * cl.len;
      const cg  = ctx.createRadialGradient(cx2, cy2, 0, cx2, cy2, cw);
      cg.addColorStop(0, 'rgba(255,255,255,0.15)');
      cg.addColorStop(1, 'transparent');
      ctx.fillStyle = cg; ctx.beginPath();
      ctx.ellipse(cx2, cy2, cw, cw * 0.35, cl.a, 0, Math.PI * 2); ctx.fill();
    });
    ctx.restore();

    // Ring (if gas giant / ice world)
    if (ringCol) {
      ctx.save();
      ctx.translate(cx, cy); ctx.scale(1, 0.32);
      ctx.beginPath();
      ctx.arc(0, 0, pr * 1.55, 0, Math.PI * 2);
      ctx.strokeStyle = ringCol; ctx.lineWidth = pr * 0.22; ctx.stroke();
      // inner ring gap
      ctx.beginPath(); ctx.arc(0, 0, pr * 1.05, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0,0,0,0.55)'; ctx.lineWidth = pr * 0.12; ctx.stroke();
      ctx.restore();
    }

    // Specular highlight
    const hl = ctx.createRadialGradient(cx - pr * 0.32, cy - pr * 0.32, 0, cx - pr * 0.2, cy - pr * 0.2, pr * 0.5);
    hl.addColorStop(0, 'rgba(255,255,255,0.13)');
    hl.addColorStop(1, 'transparent');
    ctx.fillStyle = hl; ctx.beginPath(); ctx.arc(cx, cy, pr, 0, Math.PI * 2); ctx.fill();

    // Shadow (night side)
    const shadow = ctx.createRadialGradient(cx + pr * 0.4, cy + pr * 0.2, pr * 0.2, cx + pr * 0.55, cy, pr * 1.1);
    shadow.addColorStop(0, 'rgba(0,0,0,0.72)');
    shadow.addColorStop(1, 'transparent');
    ctx.fillStyle = shadow; ctx.beginPath(); ctx.arc(cx, cy, pr, 0, Math.PI * 2); ctx.fill();

    // Rim glow
    ctx.beginPath(); ctx.arc(cx, cy, pr, 0, Math.PI * 2);
    ctx.strokeStyle = isHabitable ? 'rgba(0,212,170,0.45)' : 'rgba(255,100,50,0.35)';
    ctx.lineWidth = 1.5; ctx.stroke();

    p3anim = requestAnimationFrame(frame);
  }
  frame();
}

// ── Temperature Zone ─────────────────────────────────
function updateTempZone(eqt) {
  const needle = document.getElementById('tzNeedle');
  const curr   = document.getElementById('tzCurrent');
  if (!needle || !curr) return;
  curr.textContent = eqt > 0 ? `${eqt.toFixed(0)} K` : '— K';

  // Map eqt to 0–100% across bar (50K=0%, 800K=100%)
  const pct = Math.max(0, Math.min(100, ((eqt - 50) / 750) * 100));
  needle.style.left = pct + '%';

  // Glow color
  let zoneColor;
  if (eqt < 150)       zoneColor = '#b8e8ff';
  else if (eqt < 220)  zoneColor = '#6dd5ed';
  else if (eqt <= 340) zoneColor = '#00d4aa';
  else if (eqt <= 500) zoneColor = '#f59e0b';
  else                 zoneColor = '#f43f5e';
  needle.style.background = zoneColor;
  needle.style.boxShadow  = `0 0 10px ${zoneColor}`;
  curr.style.color = zoneColor;
}

// conf = certainty of prediction (already correct — 0-1)
// isH  = boolean, used only for label color
function updateConfidence(conf, isH) {
  const pct  = document.getElementById('confPct');
  const fill = document.getElementById('confBarFill');
  const desc = document.getElementById('confDesc');
  if (!pct || !fill || !desc) return;

  const val = (conf * 100).toFixed(1);
  pct.textContent = val + '%';

  let color, label;
  if (conf >= 0.90)      { color = '#10b981'; label = '🟢 Very High Confidence — Strong prediction signal'; }
  else if (conf >= 0.75) { color = '#00d4aa'; label = '🔵 High Confidence — Reliable prediction'; }
  else if (conf >= 0.60) { color = '#f59e0b'; label = '🟡 Moderate Confidence — Some uncertainty'; }
  else                   { color = '#f43f5e'; label = '🔴 Low Confidence — Borderline case, verify manually'; }

  fill.style.width      = val + '%';
  fill.style.background = `linear-gradient(90deg,${color}88,${color})`;
  fill.style.boxShadow  = `0 0 12px ${color}55`;
  pct.style.color       = color;
  desc.textContent      = label;
  desc.style.color      = color;
}

// ── Score Breakdown Bars ─────────────────────────────
function renderScoreBars(scores) {
  const list = document.getElementById('scoreBarsList');
  if (!list) return;
  list.innerHTML = Object.entries(scores).map(([label, val]) => {
    const pct = Math.round(val * 100);
    let col = pct >= 70 ? '#10b981' : pct >= 50 ? '#f59e0b' : '#f43f5e';
    return `<div class="sb-row">
      <span class="sb-lbl">${label}</span>
      <div class="sb-bar-outer">
        <div class="sb-bar-fill" style="width:${pct}%;background:linear-gradient(90deg,${col}88,${col});box-shadow:0 0 8px ${col}44"></div>
      </div>
      <span class="sb-pct" style="color:${col}">${pct}%</span>
    </div>`;
  }).join('');
}

// ── Live Visual Indicators (update on input) ─────────
function updateVisualIndicators() {
  const rade  = parseFloat(document.getElementById('pl_rade')?.value);
  const eqt   = parseFloat(document.getElementById('pl_eqt')?.value);
  const per   = parseFloat(document.getElementById('pl_orbper')?.value);
  const steff = parseFloat(document.getElementById('st_teff')?.value);
  const mass  = parseFloat(document.getElementById('pl_bmasse')?.value);

  function setVI(id, valEl, val, unit, okMin, okMax) {
    const badge = document.getElementById(id);
    const span  = document.getElementById(valEl);
    if (!badge || !span) return;
    if (isNaN(val)) { span.textContent = '—'; badge.className = 'vi-badge'; return; }
    span.textContent = val + unit;
    const ok = val >= okMin && val <= okMax;
    const warn = ok ? false : Math.abs(val - (okMin + okMax) / 2) > (okMax - okMin);
    badge.className = 'vi-badge ' + (ok ? 'vi-ok' : warn ? 'vi-bad' : 'vi-warn');
  }

  setVI('vi_radius', 'vi_radius_val', rade,  ' R⊕', 0.5, 2.0);
  setVI('vi_temp',   'vi_temp_val',   eqt,   ' K',  220, 340);
  setVI('vi_period', 'vi_period_val', per,   ' d',  100, 700);
  setVI('vi_star',   'vi_star_val',   steff, ' K',  4800, 6500);
  setVI('vi_mass',   'vi_mass_val',   mass,  ' M⊕', 0.1, 5);
}

// ── Sample Data Loader ───────────────────────────────
async function loadSampleData(type) {
  let d;
  if (type === 'habitable')         d = SAMPLE_DATA.habitable[Math.floor(Math.random() * SAMPLE_DATA.habitable.length)];
  else if (type === 'nonHabitable') d = SAMPLE_DATA.nonHabitable[Math.floor(Math.random() * SAMPLE_DATA.nonHabitable.length)];
  else {
    const all = [...SAMPLE_DATA.habitable, ...SAMPLE_DATA.nonHabitable];
    d = all[Math.floor(Math.random() * all.length)];
  }

  _fillFormFields(d);

  // Glow animation
  const color = type === 'habitable' ? 'var(--teal)' : type === 'nonHabitable' ? 'var(--red)' : 'var(--amber)';
  document.querySelectorAll('.fin').forEach(el => {
    el.style.borderColor = color;
    setTimeout(() => { el.style.borderColor = ''; }, 1000);
  });

  updateVisualIndicators();
  await _showLivePreview(d, type);
}

// ── Zone-specific sample loader ──────────────────────
// Loads a planet that specifically falls in the Frozen or Warm temperature zone
async function loadZoneSample(zone) {
  // ❄️ Frozen — pick from SAMPLE_DATA (eqt < 150K, all scores low)
  const d = SAMPLE_DATA.nonHabitable.find(p => p.pl_eqt < 150) || SAMPLE_DATA.nonHabitable[0];

  _fillFormFields(d);

  document.querySelectorAll('.fin').forEach(el => {
    el.style.borderColor = '#b8e8ff';
    setTimeout(() => { el.style.borderColor = ''; }, 1000);
  });

  updateVisualIndicators();
  await _showLivePreview(d, 'nonHabitable');
}

// ── Fill all form fields from a data object ──────────
function _fillFormFields(d) {
  ['pl_rade','pl_bmasse','pl_orbper','pl_orbsmax','pl_eqt',
   'st_teff','st_rad','st_mass','st_met','st_lum',
   'temp_score','dist_score','lum_score','stellar_temp_score',
   'stellar_mass_score','stellar_compatibility_index','orbital_stability'
  ].forEach(f => { const el = document.getElementById(f); if (el && d[f] !== undefined) el.value = d[f]; });
  ['spec_F','spec_G','spec_K','spec_M','spec_m'].forEach(f => {
    const el = document.getElementById(f); if (el) el.checked = d[f] === 1;
  });
}

// ── Live Preview — calls same pipeline as Predict button (RF model) ──────
async function _showLivePreview(d, type) {
  const eqt   = d.pl_eqt  || 288;
  const rade  = d.pl_rade  || 1;
  const steff = d.st_teff  || 5772;

  // Try the same Flask API first (same as Predict button)
  let result;
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(d),
      signal: AbortSignal.timeout(4000)
    });
    const json = await res.json();
    if (res.ok) result = json;
    else throw new Error(json.message);
  } catch {
    // Flask offline — use RF simulator (same algorithm, approximate weights)
    result = predictLocally(d);
  }

  const isH  = result.prediction_result === 1;
  // prob_habitable = P(habitable) → used for Hab Score display
  // confidence_score = certainty of prediction → used for confidence meter
  const pHab       = result.prob_habitable   ?? result.confidence_score ?? (isH ? 0.93 : 0.07);
  const confidence = result.confidence_score ?? (isH ? pHab : 1 - pHab);

  // Show result panel
  const resultsDiv = document.getElementById('results');
  if (resultsDiv) resultsDiv.classList.remove('d-none');

  // Banner
  const alertEl = document.getElementById('resultAlert');
  const sourceNote = result.local
    ? `&nbsp;<span style="opacity:.6;font-size:.8em">(RF Simulator — Flask band hai, real result ke liye python app.py chalao)</span>`
    : `&nbsp;<span style="opacity:.6;font-size:.8em">(click Predict Habitability to confirm)</span>`;
  if (alertEl) {
    alertEl.className = `res-banner ${isH ? 'res-ok' : 'res-err'}`;
    alertEl.innerHTML = `<i class="fas fa-flask"></i>&nbsp;
      <strong>Sample Preview:</strong>&nbsp;${d.pl_name || 'Unknown Planet'}&nbsp;—&nbsp;
      ${isH ? '🌍 Predicted HABITABLE' : '✗ Predicted NON-HABITABLE'}
      ${sourceNote}`;
  }

  // Habitable Score = P(habitable), Non-Habitable Score = 1 - P(habitable)
  const hPct = (pHab * 100).toFixed(1);
  const nPct = ((1 - pHab) * 100).toFixed(1);
  const hScoreEl = document.getElementById('habitableScore');
  const nScoreEl = document.getElementById('nonHabitableScore');
  if (hScoreEl) hScoreEl.textContent = hPct + '%';
  if (nScoreEl) nScoreEl.textContent = nPct + '%';

  // Score bars
  const scores = {
    'Temp Score':    d.temp_score    || 0,
    'Dist Score':    d.dist_score    || 0,
    'Lum Score':     d.lum_score     || 0,
    'Stellar Temp':  d.stellar_temp_score  || 0,
    'Stellar Mass':  d.stellar_mass_score  || 0,
    'Compatibility': d.stellar_compatibility_index || 0,
    'Orbital Stab':  d.orbital_stability || 0,
  };

  // All 4 visual components
  draw3DPlanet(isH, rade, eqt, steff);
  updateTempZone(eqt);
  updateConfidence(confidence, isH);  // confidence = certainty of prediction
  renderScoreBars(scores);

  // Scroll into view
  setTimeout(() => {
    resultsDiv?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 250);
}

// ════════════════════════════════════════════════════
//  VALIDATION
// ════════════════════════════════════════════════════
function validateFormData(data) {
  const req = ['pl_rade','pl_bmasse','pl_orbper','pl_orbsmax','pl_eqt','st_teff','st_rad','st_mass','st_met','st_lum','temp_score','dist_score','lum_score','stellar_temp_score','stellar_mass_score','stellar_compatibility_index','orbital_stability'];
  for (const f of req) {
    if (!(f in data) || isNaN(data[f])) return { isValid:false, message:`Please fill: ${f.replace(/_/g,' ')}` };
  }
  const scores = ['temp_score','dist_score','lum_score','stellar_temp_score','stellar_mass_score','stellar_compatibility_index','orbital_stability'];
  for (const f of scores) {
    if (data[f] < 0 || data[f] > 1) return { isValid:false, message:`${f.replace(/_/g,' ')} must be between 0 and 1` };
  }
  return { isValid:true };
}

function showFormError(msg) {
  const div   = document.getElementById('results');
  const alert = document.getElementById('resultAlert');
  if (div && alert) {
    div.classList.remove('d-none');
    alert.className = 'res-banner res-err';
    alert.innerHTML = `<i class="fas fa-exclamation-triangle"></i>  ${msg}`;
    div.scrollIntoView({ behavior:'smooth' });
  }
}

// ════════════════════════════════════════════════════
//  AUTO FILL
// ════════════════════════════════════════════════════
async function autoFillHabitableData() {
  const d = SAMPLE_DATA.habitable[0];
  _fillFormFields(d);
  // Glow effect
  document.querySelectorAll('.fin').forEach(el => {
    el.style.borderColor = 'var(--teal)';
    setTimeout(() => { el.style.borderColor = ''; }, 1200);
  });
  updateVisualIndicators();
  await _showLivePreview(d, 'habitable');
}

// ════════════════════════════════════════════════════
//  LOADING STATE
// ════════════════════════════════════════════════════
function setLoadingState(btn, loading) {
  if (!btn) return;
  btn.disabled = loading;
  btn.innerHTML = loading
    ? '<i class="fas fa-spinner fa-spin"></i> Analyzing...'
    : '<i class="fas fa-rocket"></i> Predict Habitability';
}

// ════════════════════════════════════════════════════
//  DOWNLOAD SAMPLE CSV
// ════════════════════════════════════════════════════
function downloadSampleCSV(type) {
  const rows = type === 'habitable' ? SAMPLE_DATA.habitable : SAMPLE_DATA.nonHabitable;
  const headers = ['pl_rade','pl_bmasse','pl_orbper','pl_orbsmax','pl_eqt','st_teff','st_rad','st_mass','st_met','st_lum','temp_score','dist_score','lum_score','stellar_temp_score','stellar_mass_score','stellar_compatibility_index','orbital_stability','spec_F','spec_G','spec_K','spec_M','spec_m'];
  let csv = headers.join(',') + '\n';
  rows.forEach(r => { csv += headers.map(h => r[h] ?? '').join(',') + '\n'; });
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([csv],{type:'text/csv'})),
    download: `sample_${type}.csv`
  });
  document.body.appendChild(a); a.click(); a.remove();
}
// ════════════════════════════════════════════════════
//  ANALYTICS DASHBOARD — Full Implementation
// ════════════════════════════════════════════════════

// ── Data store ────────────────────────────────────
let _anlHistory = [];
try { _anlHistory = JSON.parse(localStorage.getItem('exo_anl_v2') || '[]'); } catch(e) {}

function anlSaveEntry(entry) {
  const now = new Date();
  // Planet type: physical params checked FIRST, then habitability
  const rade = entry.rade || 1;
  const eqt  = entry.eqt  || 300;
  let type;
  if      (rade > 2.5)   type = '🪐 Gas Giant';
  else if (eqt  > 600)   type = '🔥 Lava World';
  else if (eqt  < 175)   type = '❄️ Ice World';
  else if (entry.isH)    type = '🌍 Earth-like';
  else                   type = '🪨 Rocky Planet';

  _anlHistory.unshift({
    ...entry,
    time: now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}),
    date: now.toLocaleDateString(),
    id:   now.getTime(),
    type
  });
  if (_anlHistory.length > 500) _anlHistory.pop();
  try { localStorage.setItem('exo_anl_v2', JSON.stringify(_anlHistory)); } catch(e) {}
}


// ════════════════════════════════════════════════════
//  ANALYTICS DASHBOARD — COMPLETE v2 (Futuristic UI)
// ════════════════════════════════════════════════════

/* ── Reset Dashboard ─────────────────────────────── */
function anlResetDashboard() {
  if (!confirm('Reset all prediction history? This will clear all analytics data.')) return;

  // Clear data
  _anlHistory = [];
  try {
    localStorage.removeItem('exo_anl_v2');
    localStorage.removeItem('exo_anl');
  } catch(e) {}

  // Update nav badge
  const badge = document.getElementById('anlNavBadge');
  if (badge) badge.style.display = 'none';

  // Refresh to show empty state
  anlRefresh();

  // Flash confirmation
  const syncEl = document.getElementById('anlSyncTime');
  if (syncEl) {
    syncEl.textContent = 'Dashboard reset ✓';
    syncEl.style.color = '#10b981';
    setTimeout(() => {
      syncEl.textContent = 'Synced just now';
      syncEl.style.color = '';
    }, 2500);
  }
}

/* ── Main Refresh ────────────────────────────────── */
function anlRefresh() {
  const data  = _anlHistory;
  const n     = data.length;
  const habN  = data.filter(d => d.isH).length;
  const nhN   = n - habN;
  const rate  = n ? (habN / n * 100) : 0;
  const avgConf = n ? (data.reduce((s,d)=>s+d.conf,0)/n*100).toFixed(1) : '—';
  const fprN  = data.filter(d => !d.isH && d.conf > 0.5).length;
  const fpr   = nhN > 0 ? (fprN/nhN*100).toFixed(1)+'%' : '0%';

  // Stat cards
  _anlAnimate('anlTotal', n);
  _anlAnimate('anlHab',   habN);
  _s2('anlHabTrend',   n ? rate.toFixed(1)+'% habitable rate' : '—% habitable rate');
  _s2('anlTotalTrend', n ? `${n} planet${n!==1?'s':''} this session` : 'Session data');
  _s2('anlSubtitle',   n ? `Comprehensive insights into ${n} analyzed planet${n!==1?'s':''}` : 'Comprehensive insights into exoplanet habitability predictions');
  _s2('dsTotalPlanets', n);
  _s2('dsHabPlanets',   habN);
  _s2('dsNHPlanets',    nhN);
  _s2('dsAvgConf',      n ? avgConf+'%' : '—');
  _s2('dsAvgHab',       n ? (data.reduce((s,d)=>s+(d.habScore||0),0)/n*100).toFixed(1)+'%' : '—');
  _s2('anlLegH',        habN);
  _s2('anlLegNH',       nhN);
  _s2('anlDonutTotal',  n);
  _s2('anlSyncTime',    n ? `Synced — ${new Date().toLocaleTimeString()}` : 'Synced just now');

  // Rate ring SVG animation
  const circle = document.getElementById('anlRateCircle');
  const ratePct = document.getElementById('anlRatePct');
  if (circle && n > 0) {
    const circumference = 2 * Math.PI * 32; // r=32
    const dash = (rate/100) * circumference;
    circle.style.strokeDasharray = `${dash.toFixed(1)} ${circumference.toFixed(1)}`;
    if (ratePct) ratePct.textContent = rate.toFixed(1)+'%';
  } else if (ratePct) {
    ratePct.textContent = '0%';
  }

  // Charts
  _drawFeatBars();
  _drawDist(data);
  _drawDonut2(habN, nhN);
  _draw3DBgPlanet();
  _drawBarChart(data);
  _drawLineChart(data);
  _drawScatterChart(data);

  // Table
  const tblWrap = document.getElementById('anlTableWrap');
  const emptyEl = document.getElementById('anlEmpty');
  if (tblWrap) tblWrap.style.display = n ? 'block' : 'none';
  if (emptyEl) emptyEl.style.display = n ? 'none' : 'flex';
  _drawTable2(data);

  // Tooltips
  _initTooltips();
}

function _s2(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function _anlAnimate(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = parseInt(el.textContent) || 0;
  if (start === target) return;
  const step = Math.ceil(Math.abs(target - start) / 20);
  let cur = start;
  const timer = setInterval(() => {
    cur = target > cur ? Math.min(cur + step, target) : Math.max(cur - step, target);
    el.textContent = cur;
    if (cur === target) clearInterval(timer);
  }, 30);
}

/* ── Feature Importance — CSS Bars ──────────────── */
function _drawFeatBars() {
  const container = document.getElementById('anlFeatList');
  if (!container) return;

  // === ACTUAL feature importances from Tuned Random Forest model ===
  const features = [
    { name:'Planet Radius (pl_rade)',        val:0.4346, col:'#e040fb' },
    { name:'Planet Mass (pl_bmasse)',         val:0.2445, col:'#c060fb' },
    { name:'Stellar Radius (st_rad)',         val:0.0504, col:'#a040f0' },
    { name:'Stellar Metallicity (st_met)',    val:0.0453, col:'#9030e0' },
    { name:'Stellar Mass (st_mass)',          val:0.0439, col:'#7c20d8' },
    { name:'Stellar Luminosity (st_lum)',     val:0.0420, col:'#6a10c8' },
    { name:'Eq. Temperature (pl_eqt)',        val:0.0205, col:'#5800b8' },
    { name:'Orbital Period (pl_orbper)',      val:0.0190, col:'#4800a8' },
    { name:'Stellar Temp (st_teff)',          val:0.0189, col:'#380098' },
    { name:'Stellar Temp Score',              val:0.0170, col:'#280088' },
  ];

  // Normalize to max=1 for bar display
  const maxVal = features[0].val;
  container.innerHTML = features.map((f, i) => {
    const barPct = ((f.val / maxVal) * 100).toFixed(1);
    return `<div class="anl-feat-row" style="animation-delay:${i*60}ms">
      <div class="anl-feat-name" title="${f.name}">${f.name}</div>
      <div class="anl-feat-track">
        <div class="anl-feat-bar" style="width:0%;background:linear-gradient(90deg,${f.col}55,${f.col});box-shadow:0 0 10px ${f.col}44;transition:width 1.2s cubic-bezier(.4,0,.2,1) ${i*60}ms"
          data-width="${barPct}"></div>
      </div>
      <div class="anl-feat-val" style="color:${f.col}">${(f.val*100).toFixed(2)}%</div>
    </div>`;
  }).join('');

  // Animate bars in with stagger
  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll('.anl-feat-bar').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 80);
  });
}

/* ── Score Distribution Canvas ───────────────────── */
function _drawDist(data) {
  const canvas = document.getElementById('anlDistChart');
  if (!canvas) return;
  const W = canvas.offsetWidth || 420, H = 220;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const buckets = new Array(10).fill(0);
  if (data.length) {
    data.forEach(d => {
      const idx = Math.min(9, Math.floor((d.habScore || 0) * 10));
      buckets[idx]++;
    });
  } else {
    // Demo: RF model tends to cluster predictions near 0.85–0.95
    // Buckets: 0-10%, 10-20%, ..., 90-100%
    [3, 2, 4, 3, 5, 4, 8, 14, 22, 35].forEach((v,i) => buckets[i]=v);
  }

  const maxV = Math.max(...buckets, 1);
  const pL=44, pR=12, pT=22, pB=30;
  const cW=W-pL-pR, cH=H-pT-pB;
  const bW=cW/10;

  // Animate the chart drawing
  const startTime = performance.now();
  const animDur   = 1000;

  function drawFrame(now) {
    const t = Math.min((now - startTime) / animDur, 1);
    const ease = 1 - Math.pow(1 - t, 3);

    ctx.clearRect(0, 0, W, H);

    // Grid lines
    for (let i=0; i<=4; i++) {
      const y = pT + cH - cH*i/4;
      ctx.beginPath(); ctx.moveTo(pL,y); ctx.lineTo(W-pR,y);
      ctx.strokeStyle='rgba(255,255,255,.05)'; ctx.lineWidth=.5; ctx.stroke();
      ctx.fillStyle='#7aadcc'; ctx.font='9px monospace'; ctx.textAlign='right';
      ctx.fillText(Math.round(maxV*i/4), pL-4, y+3);
    }

    // Animated points
    const animPoints = buckets.map((v,i) => ({
      x: pL + (i+0.5)*bW,
      y: pT + cH - (v/maxV)*cH * ease
    }));

    // Fill area
    const areaGrad = ctx.createLinearGradient(0, pT, 0, pT+cH);
    areaGrad.addColorStop(0, 'rgba(0,212,170,0.4)');
    areaGrad.addColorStop(1, 'rgba(0,212,170,0.02)');
    ctx.beginPath();
    ctx.moveTo(animPoints[0].x, pT+cH);
    animPoints.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(animPoints[animPoints.length-1].x, pT+cH);
    ctx.closePath();
    ctx.fillStyle = areaGrad; ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(animPoints[0].x, animPoints[0].y);
    animPoints.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.strokeStyle='#00d4aa'; ctx.lineWidth=2.5; ctx.stroke();

    // Glow dots
    animPoints.forEach(p => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 3.5, 0, Math.PI*2);
      ctx.fillStyle='#00d4aa';
      ctx.shadowColor='#00d4aa'; ctx.shadowBlur=10;
      ctx.fill(); ctx.shadowBlur=0;
    });

    // X labels
    ['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%']
      .forEach((lbl,i) => {
        ctx.fillStyle='#6a9cb8'; ctx.font='7px monospace'; ctx.textAlign='center';
        ctx.fillText(lbl, pL+(i+0.5)*bW, H-4);
      });

    // Legend
    ctx.fillStyle='rgba(0,212,170,.7)';
    ctx.fillRect(W-130, pT-12, 10, 7);
    ctx.fillStyle='#8ec4dc'; ctx.font='9px monospace'; ctx.textAlign='left';
    ctx.fillText('Number of Planets', W-116, pT-6);

    if (t < 1) requestAnimationFrame(drawFrame);
  }

  requestAnimationFrame(drawFrame);
}

/* ── Donut Chart v2 — Animated ───────────────────── */
let _donutAF = null;
function _drawDonut2(habN, nhN) {
  const canvas = document.getElementById('anlDonutChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W=220, H=220, cx=W/2, cy=H/2, r=80, lw=28;
  canvas.width=W; canvas.height=H;

  const total  = habN + nhN || 1;
  const habFrac= habN / total;
  const TAU    = Math.PI * 2;

  // Cancel any previous animation
  if (_donutAF) { cancelAnimationFrame(_donutAF); _donutAF = null; }

  const startTime = performance.now();
  const duration  = 900; // ms

  function frame(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    // Ease out cubic
    const t = 1 - Math.pow(1 - progress, 3);

    ctx.clearRect(0, 0, W, H);

    const habAngle = habFrac * TAU * t;
    const nhAngle  = TAU * t - habAngle;

    // Background ring
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, TAU);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = lw;
    ctx.shadowBlur = 0; ctx.stroke();

    // Non-habitable arc (red)
    if (nhN > 0 && nhAngle > 0.001) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, -Math.PI/2, -Math.PI/2 + nhAngle);
      ctx.strokeStyle = '#f43f5e'; ctx.lineWidth = lw;
      ctx.shadowColor = '#f43f5e'; ctx.shadowBlur = 16;
      ctx.lineCap = 'round'; ctx.stroke(); ctx.shadowBlur = 0;
    }

    // Habitable arc (green)
    if (habN > 0 && habAngle > 0.001) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, -Math.PI/2 + nhAngle, -Math.PI/2 + nhAngle + habAngle);
      ctx.strokeStyle = '#10b981'; ctx.lineWidth = lw;
      ctx.shadowColor = '#10b981'; ctx.shadowBlur = 16;
      ctx.lineCap = 'round'; ctx.stroke(); ctx.shadowBlur = 0;
    }

    // Inner glow ring
    ctx.beginPath(); ctx.arc(cx, cy, r - lw/2 - 4, 0, TAU);
    ctx.strokeStyle = 'rgba(0,212,170,0.08)'; ctx.lineWidth = 2;
    ctx.shadowBlur = 0; ctx.stroke();

    if (progress < 1) {
      _donutAF = requestAnimationFrame(frame);
    } else {
      _donutAF = null;
    }
  }

  _donutAF = requestAnimationFrame(frame);
}

/* ── Prediction Table v2 ─────────────────────────── */
function _drawTable2(data) {
  const tbody = document.getElementById('anlTableBody');
  if (!tbody || !data.length) { if(tbody) tbody.innerHTML=''; return; }
  tbody.innerHTML = data.slice(0,50).map((d,i) => {
    const sc = d.habScore>=0.7?'#10b981':d.habScore>=0.5?'#00d4aa':d.habScore>=0.3?'#f59e0b':'#f43f5e';
    return `<tr>
      <td style="color:var(--amber);font-family:var(--fh);font-weight:700">#${i+1}</td>
      <td style="color:#c8e8f5">${d.type||'Unknown'}</td>
      <td style="color:#c8e8f5">${d.rade??'—'} R⊕</td>
      <td style="color:#c8e8f5">${d.eqt??'—'} K</td>
      <td style="color:#c8e8f5">${d.steff??'—'} K</td>
      <td style="color:${sc};font-weight:700;font-family:var(--fh)">${((d.habScore||0)*100).toFixed(1)}%</td>
      <td style="color:${d.isH?'#10b981':'#f43f5e'};font-weight:600">${((d.conf||0)*100).toFixed(1)}%</td>
      <td><span class="${d.isH?'bg':'br'}">${d.isH?'✓ Habitable':'✗ Non-Hab'}</span></td>
    </tr>`;
  }).join('');
}

/* ── 3D Rotating BG Planet ───────────────────────── */
let _bgPlanetT = 0, _bgPlanetAF = null;
function _draw3DBgPlanet() {
  const canvas = document.getElementById('anlBgPlanet');
  if (!canvas) return;
  if (_bgPlanetAF) return; // already running

  function frame() {
    const W = canvas.offsetWidth, H = canvas.offsetHeight;
    if (!W || !H) { _bgPlanetAF = requestAnimationFrame(frame); return; }
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,W,H);
    _bgPlanetT += 0.003;
    const cx=W*0.82, cy=H*0.25, r=Math.min(W,H)*0.22;

    // Atmosphere
    const ag=ctx.createRadialGradient(cx,cy,r*.7,cx,cy,r*1.6);
    ag.addColorStop(0,'rgba(100,0,200,0.15)'); ag.addColorStop(1,'transparent');
    ctx.fillStyle=ag; ctx.beginPath(); ctx.arc(cx,cy,r*1.6,0,Math.PI*2); ctx.fill();

    // Body
    const bg=ctx.createRadialGradient(cx-r*.3,cy-r*.3,r*.05,cx,cy,r);
    bg.addColorStop(0,'#2a0a5a'); bg.addColorStop(.5,'#150530'); bg.addColorStop(1,'#05000f');
    ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.fillStyle=bg; ctx.fill();

    // Surface bands
    ctx.save(); ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.clip();
    for(let i=0;i<7;i++){
      const yb=cy-r+r*2*(i/6);
      const wb=Math.sqrt(Math.max(0,r*r-(yb-cy)**2));
      if(wb<4)continue;
      ctx.beginPath();
      ctx.ellipse(cx+Math.sin(_bgPlanetT+i)*.03*r,yb,wb*.96,r*.05,0,0,Math.PI*2);
      ctx.fillStyle=`rgba(150,50,255,${.04+.02*Math.sin(_bgPlanetT+i)})`; ctx.fill();
    }
    ctx.restore();

    // Specular
    const hl=ctx.createRadialGradient(cx-r*.3,cy-r*.3,0,cx-r*.15,cy-r*.15,r*.45);
    hl.addColorStop(0,'rgba(255,255,255,.12)'); hl.addColorStop(1,'transparent');
    ctx.fillStyle=hl; ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.fill();

    // Shadow
    const sh=ctx.createRadialGradient(cx+r*.38,cy+r*.15,r*.1,cx+r*.5,cy,r*1.1);
    sh.addColorStop(0,'rgba(0,0,0,.8)'); sh.addColorStop(1,'transparent');
    ctx.fillStyle=sh; ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.fill();

    // Rim
    ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.strokeStyle='rgba(180,50,255,0.4)'; ctx.lineWidth=1.5; ctx.stroke();
    const rg=ctx.createRadialGradient(cx,cy,r-1,cx,cy,r+12);
    rg.addColorStop(0,'rgba(180,50,255,.2)'); rg.addColorStop(1,'transparent');
    ctx.strokeStyle=rg; ctx.lineWidth=16; ctx.stroke();

    _bgPlanetAF = requestAnimationFrame(frame);
  }
  frame();
}

/* ── Tooltip System ──────────────────────────────── */
function _initTooltips() {
  const tip = document.getElementById('anlTooltip');
  if (!tip) return;
  document.querySelectorAll('[data-tip]').forEach(el => {
    el.addEventListener('mouseenter', e => {
      tip.textContent = el.dataset.tip;
      tip.classList.add('anl-tip-show');
      const r = el.getBoundingClientRect();
      tip.style.left = (r.left + r.width/2) + 'px';
      tip.style.top  = (r.bottom + 8) + 'px';
    });
    el.addEventListener('mouseleave', () => tip.classList.remove('anl-tip-show'));
  });
}

/* ── Chart.js Instances ──────────────────────────── */
let _cjsBar = null, _cjsLine = null, _cjsScatter = null;

const _CJS_DEFAULTS = {
  font: { family: "'Exo 2', sans-serif", size: 11 },
  color: '#7aadcc',
  grid: 'rgba(255,255,255,0.05)',
  tick: '#6a9cb8',
};

/* ── Bar Chart: Stellar Type Distribution ─────────── */
function _drawBarChart(data) {
  const canvas = document.getElementById('anlBarChart');
  if (!canvas) return;

  // Stellar type counts from prediction history
  const counts = { 'F-type': 0, 'G-type': 0, 'K-type': 0, 'M-type': 0, 'M-dwarf': 0 };
  const habCounts = { 'F-type': 0, 'G-type': 0, 'K-type': 0, 'M-type': 0, 'M-dwarf': 0 };

  if (data.length > 0) {
    data.forEach(d => {
      if (d.specF) { counts['F-type']++; if (d.isH) habCounts['F-type']++; }
      else if (d.specG) { counts['G-type']++; if (d.isH) habCounts['G-type']++; }
      else if (d.specK) { counts['K-type']++; if (d.isH) habCounts['K-type']++; }
      else if (d.specM) { counts['M-type']++; if (d.isH) habCounts['M-type']++; }
      else if (d.specm) { counts['M-dwarf']++; if (d.isH) habCounts['M-dwarf']++; }
      else { counts['G-type']++; if (d.isH) habCounts['G-type']++; } // default
    });
  } else {
    // Demo data — realistic NASA distribution (G/K stars most common habitable hosts)
    Object.assign(counts,    { 'F-type': 8,  'G-type': 52, 'K-type': 38, 'M-type': 26, 'M-dwarf': 6 });
    Object.assign(habCounts, { 'F-type': 1,  'G-type': 31, 'K-type': 22, 'M-type': 4,  'M-dwarf': 0 });
  }

  const labels = Object.keys(counts);
  const totalData = labels.map(k => counts[k]);
  const habData   = labels.map(k => habCounts[k]);

  if (_cjsBar) { _cjsBar.destroy(); _cjsBar = null; }

  _cjsBar = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Total Planets',
          data: totalData,
          backgroundColor: 'rgba(30,144,255,0.25)',
          borderColor: '#1e90ff',
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        },
        {
          label: 'Habitable',
          data: habData,
          backgroundColor: 'rgba(16,185,129,0.4)',
          borderColor: '#10b981',
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 900, easing: 'easeOutQuart', delay: (ctx) => ctx.dataIndex * 80 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: _CJS_DEFAULTS.color, font: _CJS_DEFAULTS.font, boxWidth: 12, padding: 16 }
        },
        tooltip: {
          backgroundColor: 'rgba(4,21,37,0.95)',
          borderColor: 'rgba(0,212,170,0.3)',
          borderWidth: 1,
          titleColor: '#00d4aa',
          bodyColor: '#c8e8f5',
          padding: 12,
          callbacks: {
            afterBody: (items) => {
              const total = items.find(i => i.datasetIndex === 0)?.raw || 0;
              const hab   = items.find(i => i.datasetIndex === 1)?.raw || 0;
              return total > 0 ? [`Hab Rate: ${(hab/total*100).toFixed(1)}%`] : [];
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: _CJS_DEFAULTS.tick, font: _CJS_DEFAULTS.font },
          grid: { color: _CJS_DEFAULTS.grid },
          border: { color: 'rgba(255,255,255,0.1)' }
        },
        y: {
          beginAtZero: true,
          ticks: { color: _CJS_DEFAULTS.tick, font: _CJS_DEFAULTS.font, stepSize: 10 },
          grid: { color: _CJS_DEFAULTS.grid },
          border: { color: 'rgba(255,255,255,0.1)' }
        }
      }
    }
  });
}

/* ── Line Chart: Prediction History Trend ─────────── */
function _drawLineChart(data) {
  const canvas = document.getElementById('anlLineChart');
  if (!canvas) return;

  let points, labels;

  if (data.length > 0) {
    const slice = data.slice(0, 20).reverse();
    labels = slice.map((_, i) => `#${i + 1}`);
    points = slice.map(d => +((d.habScore || 0) * 100).toFixed(1));
  } else {
    // Demo data — realistic RF model probability scores (most planets near 0.9 threshold)
    labels = Array.from({ length: 20 }, (_, i) => `#${i+1}`);
    points = [87, 92, 45, 91, 38, 95, 93, 22, 90, 88, 41, 96, 85, 30, 92, 94, 18, 91, 89, 97];
  }

  if (_cjsLine) { _cjsLine.destroy(); _cjsLine = null; }

  _cjsLine = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Habitability Score (%)',
        data: points,
        borderColor: '#f59e0b',
        backgroundColor: (ctx) => {
          const gradient = ctx.chart.ctx.createLinearGradient(0, 0, 0, 260);
          gradient.addColorStop(0, 'rgba(245,158,11,0.35)');
          gradient.addColorStop(1, 'rgba(245,158,11,0.01)');
          return gradient;
        },
        borderWidth: 2.5,
        pointBackgroundColor: (ctx) => {
          const v = ctx.raw;
          return v >= 90 ? '#10b981' : v >= 70 ? '#f59e0b' : '#f43f5e';
        },
        pointBorderColor: 'rgba(0,0,0,0)',
        pointRadius: 5,
        pointHoverRadius: 8,
        fill: true,
        tension: 0.4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1200, easing: 'easeInOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: _CJS_DEFAULTS.color, font: _CJS_DEFAULTS.font, boxWidth: 12, padding: 16 }
        },
        tooltip: {
          backgroundColor: 'rgba(4,21,37,0.95)',
          borderColor: 'rgba(245,158,11,0.3)',
          borderWidth: 1,
          titleColor: '#f59e0b',
          bodyColor: '#c8e8f5',
          padding: 12,
          callbacks: {
            label: (item) => {
              const v = item.raw;
              const status = v >= 90 ? '🌍 Habitable (≥90%)' : v >= 70 ? '⚠️ Borderline' : '✗ Non-Habitable';
              return ` Score: ${v}% — ${status}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: _CJS_DEFAULTS.tick, font: _CJS_DEFAULTS.font, maxTicksLimit: 10 },
          grid: { color: _CJS_DEFAULTS.grid },
          border: { color: 'rgba(255,255,255,0.1)' }
        },
        y: {
          min: 0, max: 100,
          ticks: { color: _CJS_DEFAULTS.tick, font: _CJS_DEFAULTS.font, callback: v => v + '%' },
          grid: { color: _CJS_DEFAULTS.grid },
          border: { color: 'rgba(255,255,255,0.1)' }
        }
      }
    }
  });
}

/* ── Scatter Chart: Planet Mass vs Orbital Distance ── */
function _drawScatterChart(data) {
  const canvas = document.getElementById('anlScatterChart');
  if (!canvas) return;

  let habPoints, nonHabPoints;

  if (data.length > 0) {
    habPoints    = data.filter(d => d.isH).map(d  => ({ x: +(d.orbsmax||1).toFixed(3), y: +(d.mass||1).toFixed(2) }));
    nonHabPoints = data.filter(d => !d.isH).map(d => ({ x: +(d.orbsmax||1).toFixed(3), y: +(d.mass||1).toFixed(2) }));
  } else {
    // Demo points spanning the HZ region and beyond
    habPoints = [
      {x:1.00,y:1.00},{x:0.95,y:0.90},{x:1.10,y:1.20},{x:0.88,y:0.77},{x:1.03,y:1.08},
      {x:1.15,y:1.30},{x:0.92,y:1.05},{x:0.98,y:0.85},{x:1.08,y:0.95},{x:1.05,y:1.10}
    ];
    nonHabPoints = [
      {x:0.20,y:8.20},{x:0.15,y:12.5},{x:3.50,y:1.10},{x:0.50,y:15.0},{x:0.08,y:5.50},
      {x:0.10,y:9.50},{x:4.20,y:0.60},{x:0.30,y:3.80},{x:5.00,y:1.30},{x:0.12,y:7.20}
    ];
  }

  if (_cjsScatter) { _cjsScatter.destroy(); _cjsScatter = null; }

  _cjsScatter = new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Habitable',
          data: habPoints,
          backgroundColor: 'rgba(16,185,129,0.65)',
          borderColor: '#10b981',
          borderWidth: 1.5,
          pointRadius: 6,
          pointHoverRadius: 9,
        },
        {
          label: 'Non-Habitable',
          data: nonHabPoints,
          backgroundColor: 'rgba(244,63,94,0.55)',
          borderColor: '#f43f5e',
          borderWidth: 1.5,
          pointRadius: 5,
          pointHoverRadius: 8,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1000, easing: 'easeOutBack' },
      plugins: {
        legend: {
          labels: { color: _CJS_DEFAULTS.color, font: _CJS_DEFAULTS.font, boxWidth: 12, padding: 16 }
        },
        tooltip: {
          backgroundColor: 'rgba(4,21,37,0.95)',
          borderColor: 'rgba(224,64,251,0.3)',
          borderWidth: 1,
          titleColor: '#e040fb',
          bodyColor: '#c8e8f5',
          padding: 12,
          callbacks: {
            label: (item) => ` Mass: ${item.raw.y} M⊕ | Dist: ${item.raw.x} AU`,
          }
        },
        annotation: {}
      },
      scales: {
        x: {
          title: { display: true, text: 'Semi-major Axis (AU)', color: '#6a9cb8', font: _CJS_DEFAULTS.font },
          ticks: { color: _CJS_DEFAULTS.tick, font: _CJS_DEFAULTS.font },
          grid: { color: _CJS_DEFAULTS.grid },
          border: { color: 'rgba(255,255,255,0.1)' }
        },
        y: {
          title: { display: true, text: 'Planet Mass (M⊕)', color: '#6a9cb8', font: _CJS_DEFAULTS.font },
          ticks: { color: _CJS_DEFAULTS.tick, font: _CJS_DEFAULTS.font },
          grid: { color: _CJS_DEFAULTS.grid },
          border: { color: 'rgba(255,255,255,0.1)' }
        }
      }
    }
  });
}

/* ══════════════════════════════════════════════════════
   PDF REPORT — Captures all live charts + full data
══════════════════════════════════════════════════════ */
async function anlExportPDF() {
  // Refresh analytics charts first so canvases are fully rendered
  anlRefresh();
  // Wait for all Chart.js animations to complete (bar=900ms, line=1200ms, scatter=1000ms)
  await new Promise(r => setTimeout(r, 1400));

  const d   = _anlHistory;
  const n   = d.length;
  const habN = d.filter(x => x.isH).length;
  const nhN  = n - habN;
  const rate = n ? (habN / n * 100).toFixed(2) : '0';
  const avgC = n ? (d.reduce((s,x) => s + x.conf, 0) / n * 100).toFixed(1) : '—';
  const avgH = n ? (d.reduce((s,x) => s + (x.habScore||0), 0) / n * 100).toFixed(1) : '—';
  const now  = new Date().toLocaleString();

  /* ── Capture every live canvas as base64 PNG with dark background ── */
  function snap(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    try {
      // Create offscreen canvas with dark background to capture Chart.js canvas
      const oc = document.createElement('canvas');
      oc.width  = el.width  || el.offsetWidth  || 600;
      oc.height = el.height || el.offsetHeight || 280;
      const ctx = oc.getContext('2d');
      ctx.fillStyle = '#041525';
      ctx.fillRect(0, 0, oc.width, oc.height);
      ctx.drawImage(el, 0, 0, oc.width, oc.height);
      return oc.toDataURL('image/png');
    } catch(e) { return null; }
  }

  /* Offscreen donut for report (white-bg) */
  function makeDonutImg() {
    const oc = document.createElement('canvas');
    oc.width = 300; oc.height = 300;
    const ctx = oc.getContext('2d');
    ctx.fillStyle = '#041525';
    ctx.fillRect(0,0,300,300);
    const cx=150,cy=150,r=100,lw=34;
    const total = habN + nhN || 1;
    const habA  = (habN/total)*Math.PI*2;
    // Non-hab arc
    ctx.beginPath(); ctx.arc(cx,cy,r,-Math.PI/2,-Math.PI/2+(Math.PI*2-habA));
    ctx.strokeStyle='#f43f5e'; ctx.lineWidth=lw; ctx.lineCap='round'; ctx.stroke();
    // Hab arc
    if (habN > 0) {
      ctx.beginPath(); ctx.arc(cx,cy,r,-Math.PI/2+(Math.PI*2-habA),-Math.PI/2+Math.PI*2);
      ctx.strokeStyle='#10b981'; ctx.lineWidth=lw; ctx.lineCap='round'; ctx.stroke();
    }
    // Center text
    ctx.fillStyle='#e2f0fa'; ctx.font='bold 28px monospace'; ctx.textAlign='center';
    ctx.fillText(n, cx, cy+6);
    ctx.font='10px monospace'; ctx.fillStyle='#7aadcc';
    ctx.fillText('TOTAL', cx, cy+22);
    return oc.toDataURL('image/png');
  }

  /* Offscreen score distribution line chart */
  function makeDistImg() {
    const oc = document.createElement('canvas');
    oc.width=600; oc.height=240;
    const ctx=oc.getContext('2d');
    ctx.fillStyle='#041525'; ctx.fillRect(0,0,600,240);
    const buckets=new Array(10).fill(0);
    if(n>0) d.forEach(x=>{ const i=Math.min(9,Math.floor((x.habScore||0)*10)); buckets[i]++; });
    else [3,2,4,3,5,4,8,14,22,35].forEach((v,i)=>buckets[i]=v);
    const maxV=Math.max(...buckets,1), pL=44,pR=12,pT=22,pB=30;
    const cW=600-pL-pR, cH=240-pT-pB;
    // Grid
    for(let i=0;i<=4;i++){
      const y=pT+cH-cH*i/4;
      ctx.beginPath();ctx.moveTo(pL,y);ctx.lineTo(600-pR,y);
      ctx.strokeStyle='rgba(255,255,255,.07)';ctx.lineWidth=.5;ctx.stroke();
      ctx.fillStyle='#7aadcc';ctx.font='9px monospace';ctx.textAlign='right';
      ctx.fillText(Math.round(maxV*i/4),pL-4,y+3);
    }
    const bW=cW/10;
    const pts=buckets.map((v,i)=>({x:pL+(i+.5)*bW, y:pT+cH-(v/maxV)*cH}));
    const ag=ctx.createLinearGradient(0,pT,0,pT+cH);
    ag.addColorStop(0,'rgba(0,212,170,.38)'); ag.addColorStop(1,'rgba(0,212,170,.02)');
    ctx.beginPath(); ctx.moveTo(pts[0].x,pT+cH);
    pts.forEach(p=>ctx.lineTo(p.x,p.y));
    ctx.lineTo(pts[9].x,pT+cH); ctx.closePath();
    ctx.fillStyle=ag; ctx.fill();
    ctx.beginPath(); ctx.moveTo(pts[0].x,pts[0].y);
    pts.forEach(p=>ctx.lineTo(p.x,p.y));
    ctx.strokeStyle='#00d4aa';ctx.lineWidth=2.5;ctx.stroke();
    pts.forEach(p=>{
      ctx.beginPath();ctx.arc(p.x,p.y,3.5,0,Math.PI*2);
      ctx.fillStyle='#00d4aa';ctx.shadowColor='#00d4aa';ctx.shadowBlur=8;ctx.fill();ctx.shadowBlur=0;
    });
    ['0-10%','10-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70-80%','80-90%','90-100%']
      .forEach((l,i)=>{ctx.fillStyle='#6a9cb8';ctx.font='7px monospace';ctx.textAlign='center';ctx.fillText(l,pL+(i+.5)*bW,236);});
    return oc.toDataURL('image/png');
  }

  /* Capture Chart.js canvases */
  const barImg     = snap('anlBarChart');
  const lineImg    = snap('anlLineChart');
  const scatterImg = snap('anlScatterChart');
  const donutImg   = makeDonutImg();
  const distImg    = makeDistImg();

  /* ── Feature importance bars — ACTUAL values from Tuned Random Forest ── */
  const features = [
    {name:'Planet Radius (pl_rade)',       val:0.4346,col:'#e040fb'},
    {name:'Planet Mass (pl_bmasse)',        val:0.2445,col:'#c060fb'},
    {name:'Stellar Radius (st_rad)',        val:0.0504,col:'#a040f0'},
    {name:'Stellar Metallicity (st_met)',   val:0.0453,col:'#9030e0'},
    {name:'Stellar Mass (st_mass)',         val:0.0439,col:'#7c20d8'},
    {name:'Stellar Luminosity (st_lum)',    val:0.0420,col:'#6a10c8'},
    {name:'Eq. Temperature (pl_eqt)',       val:0.0205,col:'#5800b8'},
    {name:'Orbital Period (pl_orbper)',     val:0.0190,col:'#4800a8'},
    {name:'Stellar Temp (st_teff)',         val:0.0189,col:'#380098'},
    {name:'Stellar Temp Score',             val:0.0170,col:'#280088'},
  ];
  const maxFI = features[0].val;
  const featRows = features.map(f => `
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <div style="width:220px;font-size:11px;color:#c8e8f5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${f.name}</div>
      <div style="flex:1;height:10px;background:rgba(255,255,255,.05);border-radius:5px;overflow:hidden">
        <div style="width:${((f.val/maxFI)*100).toFixed(0)}%;height:100%;background:linear-gradient(90deg,${f.col}66,${f.col});border-radius:5px"></div>
      </div>
      <div style="width:52px;text-align:right;font-size:11px;color:${f.col};font-family:monospace">${(f.val*100).toFixed(2)}%</div>
    </div>`).join('');

  /* ── Prediction table rows ── */
  const rows = d.slice(0,100).map((x,i) => `
    <tr style="background:${i%2?'#091624':'#0d1f35'}">
      <td style="padding:6px 10px;color:#fcd34d;font-weight:bold;font-family:monospace">#${i+1}</td>
      <td style="padding:6px 10px;color:#c8e8f5">${x.type||'Unknown'}</td>
      <td style="padding:6px 10px;color:#c8e8f5;font-family:monospace">${x.rade??'—'} R⊕</td>
      <td style="padding:6px 10px;color:#c8e8f5;font-family:monospace">${x.eqt??'—'} K</td>
      <td style="padding:6px 10px;color:#c8e8f5;font-family:monospace">${x.steff??'—'} K</td>
      <td style="padding:6px 10px;color:#c8e8f5;font-family:monospace">${x.orbsmax??'—'} AU</td>
      <td style="padding:6px 10px;color:${x.isH?'#10b981':'#f43f5e'};font-weight:bold;font-family:monospace">${((x.habScore||0)*100).toFixed(1)}%</td>
      <td style="padding:6px 10px;color:${x.isH?'#10b981':'#f43f5e'};font-weight:bold;font-family:monospace">${((x.conf||0)*100).toFixed(1)}%</td>
      <td style="padding:6px 10px"><span style="background:${x.isH?'rgba(16,185,129,.15)':'rgba(244,63,94,.15)'};color:${x.isH?'#10b981':'#f43f5e'};padding:2px 8px;border-radius:4px;font-size:10px;border:1px solid ${x.isH?'rgba(16,185,129,.4)':'rgba(244,63,94,.4)'}">${x.isH?'✓ HABITABLE':'✗ NON-HAB'}</span></td>
      <td style="padding:6px 10px;color:#8ec4dc;font-size:11px">${x.date||''} ${x.time||''}</td>
    </tr>`).join('');

  /* ── Stellar type breakdown for report ── */
  const stCounts={F:0,G:0,K:0,M:0,m:0};
  const stHab={F:0,G:0,K:0,M:0,m:0};
  d.forEach(x=>{
    if(x.specF){stCounts.F++;if(x.isH)stHab.F++;}
    else if(x.specG){stCounts.G++;if(x.isH)stHab.G++;}
    else if(x.specK){stCounts.K++;if(x.isH)stHab.K++;}
    else if(x.specM){stCounts.M++;if(x.isH)stHab.M++;}
    else if(x.specm){stCounts.m++;if(x.isH)stHab.m++;}
    else{stCounts.G++;if(x.isH)stHab.G++;}
  });
  const stRows=[
    {label:'F-type',total:stCounts.F,hab:stHab.F,col:'#f59e0b'},
    {label:'G-type (Sun-like)',total:stCounts.G,hab:stHab.G,col:'#fcd34d'},
    {label:'K-type',total:stCounts.K,hab:stHab.K,col:'#1e90ff'},
    {label:'M-type',total:stCounts.M,hab:stHab.M,col:'#e040fb'},
    {label:'M-dwarf',total:stCounts.m,hab:stHab.m,col:'#f43f5e'},
  ].map(s=>`<tr style="background:#091624">
    <td style="padding:7px 12px;color:${s.col};font-weight:bold">${s.label}</td>
    <td style="padding:7px 12px;color:#c8e8f5;text-align:center;font-family:monospace">${s.total}</td>
    <td style="padding:7px 12px;color:#10b981;text-align:center;font-family:monospace">${s.hab}</td>
    <td style="padding:7px 12px;color:#f43f5e;text-align:center;font-family:monospace">${s.total-s.hab}</td>
    <td style="padding:7px 12px;text-align:center">
      <div style="background:rgba(255,255,255,.05);border-radius:4px;height:8px;width:120px;display:inline-block;overflow:hidden;vertical-align:middle">
        <div style="width:${s.total?((s.hab/s.total)*100).toFixed(0):0}%;height:100%;background:${s.col};border-radius:4px"></div>
      </div>
      <span style="color:${s.col};font-size:10px;font-family:monospace;margin-left:6px">${s.total?((s.hab/s.total)*100).toFixed(0):0}%</span>
    </td>
  </tr>`).join('');

  /* ── Assemble the full HTML report ── */
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ExoHabitAI Analytics Report</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#020d1a;color:#e2f0fa;font-size:14px;line-height:1.6}
  .page{max-width:1100px;margin:0 auto;padding:40px 36px}

  /* Header */
  .rpt-header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid rgba(0,212,170,.2)}
  .rpt-logo{font-size:28px;font-weight:900;font-family:monospace;letter-spacing:2px;background:linear-gradient(90deg,#e040fb,#00d4aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
  .rpt-meta{text-align:right;font-size:11px;color:#7aadcc;line-height:1.8}
  .rpt-badge{display:inline-block;background:rgba(0,212,170,.12);border:1px solid rgba(0,212,170,.3);color:#00d4aa;padding:3px 10px;border-radius:20px;font-size:10px;letter-spacing:1px;margin-top:4px}

  /* Section heading */
  .sh{font-size:12px;color:#00d4aa;font-family:monospace;letter-spacing:2.5px;text-transform:uppercase;margin:28px 0 12px;padding-bottom:7px;border-bottom:1px solid rgba(0,212,170,.12)}
  .sh-icon{margin-right:6px}

  /* Stat card rows */
  .sr{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:4px}
  .sb{flex:1;min-width:120px;background:#041525;border:1px solid rgba(0,212,170,.18);border-radius:10px;padding:14px 16px}
  .sb.red{border-color:rgba(244,63,94,.25)}
  .sb.amber{border-color:rgba(245,158,11,.25)}
  .sl{font-size:9px;color:#7aadcc;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px}
  .sv{font-size:24px;font-weight:900;font-family:monospace}

  /* Performance bars */
  .perf-row{display:flex;align-items:center;gap:10px;margin-bottom:10px}
  .perf-name{width:90px;font-size:11px;color:#8ec4dc}
  .perf-track{flex:1;height:9px;background:rgba(255,255,255,.05);border-radius:5px;overflow:hidden}
  .perf-bar{height:100%;border-radius:5px}
  .perf-pct{width:52px;text-align:right;font-size:11px;font-family:monospace}

  /* Chart images */
  .chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:4px}
  .chart-box{background:#041525;border:1px solid rgba(0,212,170,.15);border-radius:10px;padding:14px}
  .chart-box.full{grid-column:1/-1}
  .chart-title{font-size:10px;color:#7aadcc;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px}
  .chart-box img{width:100%;border-radius:6px;display:block}
  .donut-row{display:flex;align-items:center;gap:20px}
  .donut-row img{width:160px;height:160px;border-radius:50%}
  .donut-legend{display:flex;flex-direction:column;gap:10px}
  .dl-item{display:flex;align-items:center;gap:8px;font-size:12px}
  .dl-dot{width:12px;height:12px;border-radius:50%}

  /* Tables */
  table{width:100%;border-collapse:collapse}
  th{background:#041525;padding:7px 10px;text-align:left;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#7aadcc;border-bottom:1px solid rgba(0,212,170,.12)}
  td{border-bottom:1px solid rgba(255,255,255,.04)}

  /* Footer */
  .ft{margin-top:36px;text-align:center;font-size:11px;color:#3d6b82;border-top:1px solid rgba(0,212,170,.1);padding-top:14px}

  @media print{body{background:#020d1a!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
</style>
</head>
<body>
<div class="page">

  <!-- ── HEADER ── -->
  <div class="rpt-header">
    <div>
      <div class="rpt-logo">🚀 ExoHabitAI</div>
      <div style="font-size:12px;color:#7aadcc;margin-top:4px">Analytics Report — NASA Exoplanet Habitability Predictor</div>
      <div class="rpt-badge">Tuned Random Forest · 22 Features · Threshold 0.9</div>
    </div>
    <div class="rpt-meta">
      <div>Generated: <strong style="color:#c8e8f5">${now}</strong></div>
      <div>Session Predictions: <strong style="color:#e040fb">${n}</strong></div>
      <div>Habitable Found: <strong style="color:#10b981">${habN}</strong></div>
      <div>Habitability Rate: <strong style="color:#f59e0b">${rate}%</strong></div>
    </div>
  </div>

  <!-- ── SUMMARY STATS ── -->
  <div class="sh"><span class="sh-icon">📊</span>Session Summary</div>
  <div class="sr">
    <div class="sb"><div class="sl">Total Analyzed</div><div class="sv" style="color:#e040fb">${n}</div></div>
    <div class="sb"><div class="sl">Habitable</div><div class="sv" style="color:#10b981">${habN}</div></div>
    <div class="sb red"><div class="sl">Non-Habitable</div><div class="sv" style="color:#f43f5e">${nhN}</div></div>
    <div class="sb amber"><div class="sl">Hab. Rate</div><div class="sv" style="color:#f59e0b">${rate}%</div></div>
    <div class="sb"><div class="sl">Avg Confidence</div><div class="sv" style="color:#00d4aa">${avgC}${avgC!=='—'?'%':''}</div></div>
    <div class="sb"><div class="sl">Avg Hab. Score</div><div class="sv" style="color:#1e90ff">${avgH}${avgH!=='—'?'%':''}</div></div>
  </div>

  <!-- ── MODEL PERFORMANCE ── -->
  <div class="sh"><span class="sh-icon">🧠</span>Model Performance Metrics</div>
  <div style="background:#041525;border:1px solid rgba(0,212,170,.15);border-radius:10px;padding:18px 20px">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 40px">
      ${[
        {name:'Accuracy',   val:98.89, col:'#00d4aa'},
        {name:'Recall',     val:97.50, col:'#e040fb'},
        {name:'F1 Score',   val:98.73, col:'#10b981'},
        {name:'Precision',  val:100.0, col:'#f59e0b'},
        {name:'ROC-AUC',    val:99.94, col:'#1e90ff'},
        {name:'Features',   val:100.0, col:'#b8e8ff', label:'22'},
      ].map(p=>`
        <div class="perf-row">
          <div class="perf-name">${p.name}</div>
          <div class="perf-track"><div class="perf-bar" style="width:${p.val}%;background:linear-gradient(90deg,${p.col}55,${p.col})"></div></div>
          <div class="perf-pct" style="color:${p.col}">${p.label||p.val+'%'}</div>
        </div>`).join('')}
    </div>
    <div style="display:flex;gap:16px;margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,.05)">
      <div style="font-size:11px;color:#7aadcc">Model: <span style="color:#c8e8f5">Tuned Random Forest</span></div>
      <div style="font-size:11px;color:#7aadcc">Threshold: <span style="color:#f59e0b">0.9000</span></div>
      <div style="font-size:11px;color:#7aadcc">Training: <span style="color:#c8e8f5">NASA Exoplanet Archive</span></div>
    </div>
  </div>

  <!-- ── CHARTS SECTION ── -->
  <div class="sh"><span class="sh-icon">📈</span>Analytics Charts</div>
  <div class="chart-grid">

    <!-- Donut + Score Distribution -->
    <div class="chart-box">
      <div class="chart-title">🍩 Classification Results — Habitable vs Non-Habitable</div>
      <div class="donut-row">
        <img src="${donutImg}" alt="Donut Chart" style="width:140px;height:140px">
        <div class="donut-legend">
          <div class="dl-item"><div class="dl-dot" style="background:#10b981"></div><div><div style="color:#7aadcc;font-size:10px">Habitable</div><div style="color:#10b981;font-size:20px;font-weight:900;font-family:monospace">${habN}</div></div></div>
          <div class="dl-item"><div class="dl-dot" style="background:#f43f5e"></div><div><div style="color:#7aadcc;font-size:10px">Non-Habitable</div><div style="color:#f43f5e;font-size:20px;font-weight:900;font-family:monospace">${nhN}</div></div></div>
          <div style="margin-top:4px;padding:8px 12px;background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);border-radius:8px;text-align:center">
            <div style="font-size:10px;color:#7aadcc">Habitability Rate</div>
            <div style="font-size:22px;font-weight:900;color:#10b981;font-family:monospace">${rate}%</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Score Distribution -->
    <div class="chart-box">
      <div class="chart-title">📉 Habitability Score Distribution</div>
      ${distImg ? `<img src="${distImg}" alt="Score Distribution">` : '<div style="color:#5a8099;text-align:center;padding:30px">Chart not available</div>'}
    </div>

    <!-- Bar Chart -->
    <div class="chart-box">
      <div class="chart-title">📊 Stellar Type Distribution (Total vs Habitable)</div>
      ${barImg ? `<img src="${barImg}" alt="Stellar Type Bar Chart">` : '<div style="color:#5a8099;text-align:center;padding:30px">No data yet — run predictions first</div>'}
    </div>

    <!-- Line Chart -->
    <div class="chart-box">
      <div class="chart-title">📈 Prediction History Trend (Habitability Score)</div>
      ${lineImg ? `<img src="${lineImg}" alt="History Trend Line Chart">` : '<div style="color:#5a8099;text-align:center;padding:30px">No data yet — run predictions first</div>'}
    </div>

    <!-- Scatter Plot full width -->
    <div class="chart-box full">
      <div class="chart-title">⚫ Scatter Plot — Planet Mass (M⊕) vs Orbital Distance (AU)</div>
      ${scatterImg ? `<img src="${scatterImg}" alt="Mass vs Distance Scatter">` : '<div style="color:#5a8099;text-align:center;padding:30px">No data yet — run predictions first</div>'}
    </div>

  </div>

  <!-- ── FEATURE IMPORTANCE ── -->
  <div class="sh"><span class="sh-icon">🔬</span>Feature Importance (Top 10)</div>
  <div style="background:#041525;border:1px solid rgba(0,212,170,.15);border-radius:10px;padding:18px 20px">
    ${featRows}
  </div>

  <!-- ── STELLAR TYPE TABLE ── -->
  ${n ? `
  <div class="sh"><span class="sh-icon">⭐</span>Stellar Type Breakdown</div>
  <table>
    <thead><tr><th>Stellar Type</th><th style="text-align:center">Total Planets</th><th style="text-align:center">Habitable</th><th style="text-align:center">Non-Habitable</th><th style="text-align:center">Hab. Rate</th></tr></thead>
    <tbody>${stRows}</tbody>
  </table>` : ''}

  <!-- ── PREDICTION HISTORY ── -->
  <div class="sh"><span class="sh-icon">🕐</span>Prediction History (${Math.min(n,100)} records)</div>
  ${n ? `
  <table>
    <thead><tr>
      <th>#</th><th>Planet Type</th><th>Radius</th><th>Eq. Temp</th><th>Star Temp</th>
      <th>Orb. Dist</th><th>Hab. Score</th><th>Confidence</th><th>Status</th><th>Date / Time</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <div style="text-align:right;font-size:10px;color:#5a8099;margin-top:6px">Showing ${Math.min(n,100)} of ${n} total records</div>`
  : '<div style="text-align:center;color:#5a8099;padding:24px;background:#041525;border-radius:8px;border:1px solid rgba(255,255,255,.05)">No predictions recorded yet — head to the Predict section to analyze planets.</div>'}

  <!-- ── FOOTER ── -->
  <div class="ft">
    <strong style="color:#00d4aa">ExoHabitAI</strong> — NASA Exoplanet Habitability Predictor<br>
    Tuned Random Forest ML · 22 Astronomical Features<br>
    <span style="color:#3d6b82">Accuracy 98.89% · Recall 97.5% · F1 98.73% · ROC-AUC 0.9994 · Threshold 0.9</span>
  </div>

</div>
</body></html>`;

  _dlBlob(html, 'text/html', `ExoHabitAI_Report_${Date.now()}.html`);
}

/* ── Export: CSV ─────────────────────────────────── */
function anlExportCSV() {
  const d=_anlHistory;
  const hdr=['#','Type','Radius_R-Earth','EqTemp_K','StarTemp_K','HabScore_%','Confidence_%','Status','Date','Time'];
  const lines=d.map((x,i)=>[i+1,`"${x.type}"`,x.rade??'',x.eqt??'',x.steff??'',
    ((x.habScore||0)*100).toFixed(2),((x.conf||0)*100).toFixed(2),
    x.isH?'Habitable':'Non-Habitable',x.date,x.time].join(','));
  const csv=[
    '# ExoHabitAI Analytics Export',
    `# Generated: ${new Date().toLocaleString()}`,
    ` Accuracy: 98.89%| Accuracy: 98.89% | Recall: 97.5% | F1: 98.73% | ROC-AUC: 0.9994 | Threshold: 0.5`,
    `# Total: ${d.length} | Habitable: ${d.filter(x=>x.isH).length} | Non-Habitable: ${d.filter(x=>!x.isH).length}`,
    '',hdr.join(','),...lines].join('\n');
  _dlBlob(csv,'text/csv',`ExoHabitAI_Data_${Date.now()}.csv`);
}

/* ── Export: JSON ────────────────────────────────── */
function anlExportJSON() {
  const d=_anlHistory;
  const payload={
    meta:{generated:new Date().toISOString(),model:'Tuned Random Forest',features:39,
      threshold:0.9,accuracy:0.9889,recall:0.975,precision:1.0,f1:0.9873,rocAuc:0.9994},
    summary:{total:d.length,habitable:d.filter(x=>x.isH).length,nonHabitable:d.filter(x=>!x.isH).length,
      avgConfidence:d.length?+(d.reduce((s,x)=>s+x.conf,0)/d.length).toFixed(4):null,
      avgHabScore:d.length?+(d.reduce((s,x)=>s+(x.habScore||0),0)/d.length).toFixed(4):null},
    predictions:d};
  _dlBlob(JSON.stringify(payload,null,2),'application/json',`ExoHabitAI_Export_${Date.now()}.json`);
}

/* ── Helpers ─────────────────────────────────────── */
function _dlBlob(content,mimeType,filename){
  const blob=new Blob([content],{type:mimeType});
  const url=URL.createObjectURL(blob);
  const a=Object.assign(document.createElement('a'),{href:url,download:filename});
  document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
}

// Init on DOM load
document.addEventListener('DOMContentLoaded',()=>{
  try{_anlHistory=JSON.parse(localStorage.getItem('exo_anl_v2')||localStorage.getItem('exo_anl')||'[]');}catch(e){}
  const badge=document.getElementById('anlNavBadge');
  if(badge&&_anlHistory.length){badge.textContent=_anlHistory.length;badge.style.display='inline-block';}
});
