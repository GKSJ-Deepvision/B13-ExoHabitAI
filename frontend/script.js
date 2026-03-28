/* ═══════════════════════════════════════════════════════
   ExoHabitAI — Frontend JavaScript
   Full stack: Starfield · Sample Picker · Predict ·
               Batch Upload · Rankings · Toast
   ═══════════════════════════════════════════════════════ */

"use strict";

const API = "/api";   // Flask serves frontend at root; API at /api/*

// ══════════════════════════════════════════════════════════
// STARFIELD
// ══════════════════════════════════════════════════════════
(function initStarfield() {
  const cv  = document.getElementById("starfield");
  const ctx = cv.getContext("2d");
  let stars = [];

  function resize() {
    cv.width  = window.innerWidth;
    cv.height = window.innerHeight;
  }

  function make(n) {
    return Array.from({ length: n }, () => ({
      x:  Math.random() * cv.width,
      y:  Math.random() * cv.height,
      r:  Math.random() * 1.3 + 0.2,
      a:  Math.random(),
      da: (Math.random() * 0.003 + 0.001) * (Math.random() > 0.5 ? 1 : -1),
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (const s of stars) {
      s.a += s.da;
      if (s.a <= 0.05 || s.a >= 1) s.da *= -1;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(180,205,255,${s.a})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  resize();
  stars = make(250);
  draw();
  window.addEventListener("resize", () => { resize(); stars = make(250); });
})();

// ══════════════════════════════════════════════════════════
// SVG GRADIENT DEF (injected once)
// ══════════════════════════════════════════════════════════
(function injectSvgDefs() {
  const svg = document.createElementNS("http://www.w3.org/2000/svg","svg");
  svg.style.cssText = "position:absolute;width:0;height:0";
  svg.innerHTML = `<defs>
    <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#3968e8"/>
      <stop offset="100%" stop-color="#22e5a0"/>
    </linearGradient>
  </defs>`;
  document.body.prepend(svg);
})();

// ══════════════════════════════════════════════════════════
// NAV SCROLL ACTIVE STATE
// ══════════════════════════════════════════════════════════
(function initNavHighlight() {
  const links    = document.querySelectorAll(".nav-links .nav-link");
  const sections = document.querySelectorAll(".page-section");

  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const id = e.target.id;
        links.forEach(l => {
          l.classList.toggle("active", l.getAttribute("data-page") === id);
        });
      }
    });
  }, { threshold: 0.3 });

  sections.forEach(s => obs.observe(s));
})();

// ══════════════════════════════════════════════════════════
// TOAST
// ══════════════════════════════════════════════════════════
function toast(msg, type = "info", duration = 4000) {
  const icons   = { info:"ℹ️", success:"✅", error:"❌", warn:"⚠️" };
  const colors  = { info:"#5a8cff", success:"#22e5a0", error:"#ff4f6b", warn:"#ffb340" };
  const tc      = document.getElementById("toastContainer");
  const el      = document.createElement("div");
  el.className  = "exo-toast";
  el.style.borderLeftColor = colors[type] || colors.info;
  el.style.borderLeftWidth = "3px";
  el.innerHTML  = `<span>${icons[type]||"ℹ️"}</span><span>${msg}</span>`;
  tc.appendChild(el);
  setTimeout(() => {
    el.classList.add("toast-exit");
    setTimeout(() => el.remove(), 320);
  }, duration);
}

// ══════════════════════════════════════════════════════════
// NAVIGATION HELPERS
// ══════════════════════════════════════════════════════════
function scrollToPredictor() {
  document.getElementById("predictor").scrollIntoView({ behavior: "smooth" });
}

function scrollTo(sel) {
  document.querySelector(sel)?.scrollIntoView({ behavior: "smooth" });
}

// ══════════════════════════════════════════════════════════
// FORM FIELDS
// ══════════════════════════════════════════════════════════
const FIELDS = [
  "planet_name",
  "pl_rade","pl_bmasse","pl_orbper","pl_orbsmax",
  "pl_eqt","pl_dens","st_teff","st_lum","st_met","st_spectype"
];

// Fields that are NOT sent to the API (UI-only)
const UI_ONLY_FIELDS = new Set(["planet_name"]);

function getFormData() {
  const d = {};
  FIELDS.forEach(id => {
    const el  = document.getElementById(id);
    const val = el ? el.value.trim() : "";
    if (id === "st_spectype" || id === "planet_name") {
      d[id] = val;
    } else {
      d[id] = val === "" ? "" : parseFloat(val);
    }
  });
  return d;
}

function getApiPayload(formData) {
  // Strip UI-only fields before sending to Flask
  const payload = { ...formData };
  UI_ONLY_FIELDS.forEach(k => delete payload[k]);
  return payload;
}

function setFormData(data) {
  FIELDS.forEach(id => {
    const el = document.getElementById(id);
    if (el && data[id] !== undefined && data[id] !== null) {
      el.value = data[id];
    }
  });
}

function clearForm() {
  FIELDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
}

function validateForm(d) {
  const numFields = FIELDS.filter(f => f !== "st_spectype");
  for (const f of numFields) {
    if (d[f] === "" || isNaN(d[f])) return `Please enter a valid number for "${f}".`;
  }
  if (!d.st_spectype) return "Please enter a spectral type (e.g. G2 V, K5, M3 V).";
  return null;
}

// ══════════════════════════════════════════════════════════
// UI STATES
// ══════════════════════════════════════════════════════════
function showState(id) {
  ["stateIdle","stateLoading","stateResult"].forEach(s => {
    document.getElementById(s).classList.toggle("d-none", s !== id);
  });
}

// ══════════════════════════════════════════════════════════
// RESULT RENDERING
// ══════════════════════════════════════════════════════════
function renderResult(result, formData) {
  const ok   = result.habitable === 1;
  const pct  = result.probability;
  const band = result.confidence_band;
  const name = formData.planet_name || "Unknown Planet";

  // Emoji + planet name + label
  document.getElementById("resEmoji").textContent = ok ? "🌍" : "☄️";

  const nameEl = document.getElementById("resPlanetName");
  nameEl.textContent = name;
  nameEl.style.display = name === "Unknown Planet" ? "none" : "block";
  const lbl = document.getElementById("resLabel");
  lbl.textContent = result.label;
  lbl.style.color = ok ? "var(--green)" : "var(--red)";

  // Badge
  const badge = document.getElementById("resBadge");
  badge.textContent = band + " Confidence";
  badge.className   = "res-badge " +
    (band === "High" ? "badge-high" : band === "Moderate" ? "badge-moderate" : "badge-low");

  // SVG Arc: circumference = 2π × 50 ≈ 314.16
  const circ = 314.16;
  const fill = (pct / 100) * circ;
  document.getElementById("arcFill").setAttribute("stroke-dasharray", `${fill} ${circ}`);
  document.getElementById("resPct").textContent = pct.toFixed(1) + "%";

  // Metric tiles
  document.getElementById("mVerdict").textContent  = ok ? "Habitable ✅" : "Not Habitable ❌";
  document.getElementById("mVerdict").style.color  = ok ? "var(--green)" : "var(--red)";
  document.getElementById("mConf").textContent     = band;
  document.getElementById("mTemp").textContent     = formData.pl_eqt !== "" ? formData.pl_eqt : "—";
  document.getElementById("mStar").textContent     = formData.st_spectype || "—";

  showState("stateResult");
}

// ══════════════════════════════════════════════════════════
// PREDICT (single)
// ══════════════════════════════════════════════════════════
document.getElementById("btnPredict").addEventListener("click", async () => {
  const data = getFormData();
  const err  = validateForm(data);
  if (err) { toast(err, "error"); return; }

  showState("stateLoading");
  document.getElementById("btnPredict").disabled = true;

  try {
    const res  = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getApiPayload(data)),
    });
    const json = await res.json();
    if (!res.ok || json.status === "error") throw new Error(json.message || "Server error");
    renderResult(json, data);
  } catch (e) {
    showState("stateIdle");
    toast("Prediction failed: " + e.message, "error");
  } finally {
    document.getElementById("btnPredict").disabled = false;
  }
});

// Reset
document.getElementById("btnReset").addEventListener("click", () => {
  clearForm();
  showState("stateIdle");
});

// ══════════════════════════════════════════════════════════
// SAMPLE PICKER
// ══════════════════════════════════════════════════════════
let samplesCache = null;

async function loadSamples() {
  if (samplesCache) return samplesCache;
  try {
    const res  = await fetch(`${API}/sample`);
    const json = await res.json();
    samplesCache = json.samples || [];
    return samplesCache;
  } catch {
    return [];
  }
}

function renderSampleGrid(samples) {
  const grid = document.getElementById("sampleGrid");
  if (!samples.length) {
    grid.innerHTML = '<p class="text-center text-muted" style="font-size:.8rem">No samples available</p>';
    return;
  }
  grid.innerHTML = samples.map((s, i) => `
    <div class="sample-card" data-idx="${i}">
      <div class="sample-label">${s.label || (s.habitability_binary === 1 ? "🌍 Habitable" : "☄️ Non-Hab")}</div>
      <div class="sample-prob">${s.habitability_probability}%</div>
      <div class="sample-spec">${s.st_spectype}</div>
    </div>
  `).join("");

  grid.querySelectorAll(".sample-card").forEach(card => {
    card.addEventListener("click", () => {
      const s = samples[parseInt(card.dataset.idx)];
      setFormData(s);
      document.getElementById("samplePicker").classList.add("d-none");
      toast("Sample loaded! Click Analyse Habitability.", "success");
    });
  });
}

document.getElementById("btnSampleToggle").addEventListener("click", async () => {
  const picker = document.getElementById("samplePicker");
  picker.classList.toggle("d-none");
  if (!picker.classList.contains("d-none")) {
    const samples = await loadSamples();
    renderSampleGrid(samples);
  }
});

document.getElementById("btnSampleClose").addEventListener("click", () => {
  document.getElementById("samplePicker").classList.add("d-none");
});

// Random fill
document.getElementById("btnRandom").addEventListener("click", async () => {
  const btn = document.getElementById("btnRandom");
  btn.disabled = true;
  try {
    const res  = await fetch(`${API}/sample/random`);
    const json = await res.json();
    if (!res.ok || json.status === "error") throw new Error(json.message);
    setFormData(json.sample);
    toast("Random sample loaded!", "success");
  } catch (e) {
    toast("Could not load random sample: " + e.message, "error");
  } finally {
    btn.disabled = false;
  }
});

// ══════════════════════════════════════════════════════════
// BATCH UPLOAD
// ══════════════════════════════════════════════════════════
const dropZone  = document.getElementById("dropZone");
const batchFile = document.getElementById("batchFile");
const batchStat = document.getElementById("batchStatus");

dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) processBatchFile(file);
});
batchFile.addEventListener("change", () => {
  if (batchFile.files[0]) processBatchFile(batchFile.files[0]);
});

async function processBatchFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["csv","json"].includes(ext)) {
    toast("Only .csv or .json files are supported.", "error");
    return;
  }

  batchStat.classList.remove("d-none");
  batchStat.innerHTML = `<div class="text-center text-muted py-2"><span class="spinner-border spinner-border-sm me-2"></span>Parsing ${file.name}…</div>`;

  try {
    const text = await file.text();
    let rows;

    if (ext === "json") {
      rows = JSON.parse(text);
      if (!Array.isArray(rows)) rows = [rows];
    } else {
      rows = parseCSV(text);
    }

    if (!rows.length) throw new Error("File is empty or could not be parsed.");
    if (rows.length > 500) throw new Error("File exceeds 500 row limit.");

    batchStat.innerHTML = `<div class="text-center text-muted py-2"><span class="spinner-border spinner-border-sm me-2"></span>Running inference on ${rows.length} planets…</div>`;

    const res  = await fetch(`${API}/predict/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rows),
    });
    const json = await res.json();
    if (!res.ok || json.status === "error") throw new Error(json.message);

    renderBatchResults(json, file.name);
  } catch (e) {
    batchStat.innerHTML = `<div class="batch-result-panel" style="border-color:var(--red)">❌ ${e.message}</div>`;
    toast("Batch error: " + e.message, "error");
  }
}

function parseCSV(text) {
  const lines  = text.trim().split("\n");
  const headers = lines[0].split(",").map(h => h.trim().replace(/"/g,""));
  return lines.slice(1).map(line => {
    const vals = line.split(",").map(v => v.trim().replace(/"/g,""));
    const obj  = {};
    headers.forEach((h, i) => {
      obj[h] = isNaN(vals[i]) || h === "st_spectype" ? vals[i] : parseFloat(vals[i]);
    });
    return obj;
  }).filter(r => Object.keys(r).length > 0);
}

function renderBatchResults(json, filename) {
  const s = json.summary;
  batchStat.innerHTML = `
    <div class="batch-result-panel">
      <div class="mb-2" style="font-weight:700;color:var(--green)">
        ✅ Batch complete — ${filename}
      </div>
      <div class="batch-stat"><span class="batch-stat-key">Total Planets</span><span class="batch-stat-val">${s.total}</span></div>
      <div class="batch-stat"><span class="batch-stat-key">Habitable</span><span class="batch-stat-val" style="color:var(--green)">${s.habitable_count}</span></div>
      <div class="batch-stat"><span class="batch-stat-key">Not Habitable</span><span class="batch-stat-val" style="color:var(--red)">${s.non_habitable}</span></div>
      <div class="batch-stat"><span class="batch-stat-key">Avg Probability</span><span class="batch-stat-val">${s.avg_probability}%</span></div>
      <button class="btn-ghost mt-2" onclick="downloadBatchCSV(${JSON.stringify(json.results).replace(/"/g,"'").slice(0,0)+'window.__lastBatch'})">
        <i class="bi bi-download me-1"></i>Download Results
      </button>
    </div>`;
  window.__lastBatch = json.results;
  // Attach click after render
  batchStat.querySelector("button").onclick = () => downloadBatchCSV(json.results);
  toast(`Batch done: ${s.habitable_count}/${s.total} habitable`, "success");
}

function downloadBatchCSV(results) {
  const cols  = ["row","habitable","probability","label","confidence_band"];
  const lines = [cols.join(",")];
  results.forEach(r => {
    lines.push(cols.map(c => `"${r[c] ?? ""}"`).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type:"text/csv" });
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = "exohabitai_batch_results.csv";
  a.click();
}

// ══════════════════════════════════════════════════════════
// RANKINGS
// ══════════════════════════════════════════════════════════
document.getElementById("btnLoadRank").addEventListener("click", loadRankings);

async function loadRankings() {
  const btn  = document.getElementById("btnLoadRank");
  const body = document.getElementById("rankBody");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Loading…';
  body.innerHTML = `<tr><td colspan="10" class="rank-empty">
    <span class="spinner-border spinner-border-sm me-2"></span>Fetching top planets…
  </td></tr>`;

  try {
    const res  = await fetch(`${API}/rank?n=10`);
    const json = await res.json();
    if (!res.ok || json.status === "error") throw new Error(json.message);

    if (!json.planets.length) {
      body.innerHTML = `<tr><td colspan="10" class="rank-empty">No data available.</td></tr>`;
      return;
    }

    body.innerHTML = json.planets.map(p => {
      const pct  = p.habitability_probability;
      const cls  = pct >= 75 ? "pill-green" : pct >= 45 ? "pill-amber" : "pill-red";
      const fmt  = v => (v !== undefined && v !== null && !isNaN(v)) ? parseFloat(v).toFixed(3) : "—";
      return `<tr>
        <td class="rank-num">${p.rank}</td>
        <td style="font-size:.8rem;color:var(--accent-glow)">${p.planet_id}</td>
        <td><span class="prob-pill ${cls}">${pct.toFixed(1)}%</span></td>
        <td>${fmt(p.pl_rade)}</td>
        <td>${fmt(p.pl_bmasse)}</td>
        <td>${fmt(p.pl_orbper)}</td>
        <td>${fmt(p.pl_eqt)}</td>
        <td>${fmt(p.st_teff)}</td>
        <td style="font-size:.78rem">${p.st_spectype}</td>
        <td class="${p.habitable ? "status-hab" : "status-not"}">${p.habitable ? "✅ Yes" : "❌ No"}</td>
      </tr>`;
    }).join("");

    toast(`Top ${json.count} planets loaded`, "success");
  } catch (e) {
    body.innerHTML = `<tr><td colspan="10" class="rank-empty" style="color:var(--red)">Error: ${e.message}</td></tr>`;
    toast("Rankings error: " + e.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>Refresh';
  }
}

// ══════════════════════════════════════════════════════════
// HEALTH CHECK ON LOAD
// ══════════════════════════════════════════════════════════
(async function checkHealth() {
  try {
    const res  = await fetch(`${API}/health`);
    const json = await res.json();
    if (json.status === "ok") {
      console.log(`✅ ExoHabitAI API ready — ${json.feature_count} features loaded`);
    }
  } catch {
    toast("Backend not reachable — ensure Flask is running on port 5000.", "warn", 6000);
  }
})();
