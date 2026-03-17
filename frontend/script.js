const API_BASE_URL = 'http://127.0.0.1:5000';
let predictionCount = 0;

// ── Checking the api for loading─────────────────
window.addEventListener('load', checkAPIHealth);

async function checkAPIHealth() {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      dot.classList.add('online');
      text.textContent = 'API: Online';
    } else {
      throw new Error();
    }
  } catch {
    dot.classList.add('offline');
    text.textContent = 'API: Offline';
  }
}

// ── Sample Data Load  ────────────────────────────────
const SAMPLE_PLANETS = [
    // ✅ HABITABLE PLANETS
    {
      planet_name: 'Earth',
      planet_radius: 1.0,
      orbital_period: 365.25,
      equilibrium_temperature: 255,
      semi_major_axis: 1.0,
      stellar_luminosity: 1.0,
      stellar_mass: 1.0
    },
    {
      planet_name: 'Kepler-452b',
      planet_radius: 1.6,
      orbital_period: 384.8,
      equilibrium_temperature: 265,
      semi_major_axis: 1.05,
      stellar_luminosity: 1.2,
      stellar_mass: 1.04
    },
    {
      planet_name: 'Kepler-442b',
      planet_radius: 1.34,
      orbital_period: 112.3,
      equilibrium_temperature: 233,
      semi_major_axis: 0.409,
      stellar_luminosity: 0.112,
      stellar_mass: 0.61
    },
    {
      planet_name: 'Kepler-62f',
      planet_radius: 1.41,
      orbital_period: 267.3,
      equilibrium_temperature: 208,
      semi_major_axis: 0.718,
      stellar_luminosity: 0.25,
      stellar_mass: 0.69
    },
    {
      planet_name: 'Kepler-22b',
      planet_radius: 2.4,
      orbital_period: 289.9,
      equilibrium_temperature: 262,
      semi_major_axis: 0.849,
      stellar_luminosity: 0.79,
      stellar_mass: 0.97
    },
    {
      planet_name: 'Kepler-296e',
      planet_radius: 1.48,
      orbital_period: 34.1,
      equilibrium_temperature: 243,
      semi_major_axis: 0.165,
      stellar_luminosity: 0.02,
      stellar_mass: 0.497
    },
    {
      planet_name: 'Kepler-186f',
      planet_radius: 1.17,
      orbital_period: 129.9,
      equilibrium_temperature: 188,
      semi_major_axis: 0.432,
      stellar_luminosity: 0.04,
      stellar_mass: 0.544
    },
    {
      planet_name: 'TRAPPIST-1e',
      planet_radius: 0.91,
      orbital_period: 6.1,
      equilibrium_temperature: 251,
      semi_major_axis: 0.0293,
      stellar_luminosity: 0.000553,
      stellar_mass: 0.0898
    },
    {
      planet_name: 'TRAPPIST-1f',
      planet_radius: 1.04,
      orbital_period: 9.2,
      equilibrium_temperature: 219,
      semi_major_axis: 0.0385,
      stellar_luminosity: 0.000553,
      stellar_mass: 0.0898
    },
    {
      planet_name: 'TRAPPIST-1g',
      planet_radius: 1.13,
      orbital_period: 12.4,
      equilibrium_temperature: 198,
      semi_major_axis: 0.0469,
      stellar_luminosity: 0.000553,
      stellar_mass: 0.0898
    },
    {
      planet_name: 'Proxima Centauri b',
      planet_radius: 1.08,
      orbital_period: 11.2,
      equilibrium_temperature: 234,
      semi_major_axis: 0.0485,
      stellar_luminosity: 0.0017,
      stellar_mass: 0.1221
    },
    {
      planet_name: 'GJ 667Cc',
      planet_radius: 1.54,
      orbital_period: 28.1,
      equilibrium_temperature: 277,
      semi_major_axis: 0.125,
      stellar_luminosity: 0.013,
      stellar_mass: 0.33
    },
    {
      planet_name: 'HD 40307g',
      planet_radius: 1.89,
      orbital_period: 197.8,
      equilibrium_temperature: 226,
      semi_major_axis: 0.6,
      stellar_luminosity: 0.23,
      stellar_mass: 0.77
    },
    {
      planet_name: 'Tau Ceti e',
      planet_radius: 1.65,
      orbital_period: 168.1,
      equilibrium_temperature: 271,
      semi_major_axis: 0.538,
      stellar_luminosity: 0.52,
      stellar_mass: 0.783
    },
    {
      planet_name: 'Kepler-1229b',
      planet_radius: 1.4,
      orbital_period: 86.8,
      equilibrium_temperature: 213,
      semi_major_axis: 0.298,
      stellar_luminosity: 0.056,
      stellar_mass: 0.54
    },
  
    // ⚠️ BORDERLINE PLANETS
    {
      planet_name: 'Mars',
      planet_radius: 0.53,
      orbital_period: 687,
      equilibrium_temperature: 210,
      semi_major_axis: 1.52,
      stellar_luminosity: 1.0,
      stellar_mass: 1.0
    },
    {
      planet_name: 'Super-Earth K2-18b',
      planet_radius: 2.27,
      orbital_period: 32.9,
      equilibrium_temperature: 265,
      semi_major_axis: 0.1429,
      stellar_luminosity: 0.035,
      stellar_mass: 0.3593
    },
    {
      planet_name: 'Kepler-62e',
      planet_radius: 1.61,
      orbital_period: 122.4,
      equilibrium_temperature: 270,
      semi_major_axis: 0.427,
      stellar_luminosity: 0.25,
      stellar_mass: 0.69
    },
    {
      planet_name: 'Wolf 1061c',
      planet_radius: 1.66,
      orbital_period: 17.9,
      equilibrium_temperature: 228,
      semi_major_axis: 0.089,
      stellar_luminosity: 0.01,
      stellar_mass: 0.294
    },
    {
      planet_name: 'Gliese 163c',
      planet_radius: 1.8,
      orbital_period: 25.6,
      equilibrium_temperature: 277,
      semi_major_axis: 0.125,
      stellar_luminosity: 0.022,
      stellar_mass: 0.4
    },
  
    // ❌ NOT HABITABLE PLANETS
    {
      planet_name: 'Hot-Jupiter-X',
      planet_radius: 11.2,
      orbital_period: 3.5,
      equilibrium_temperature: 800,
      semi_major_axis: 0.05,
      stellar_luminosity: 2.5,
      stellar_mass: 1.1
    },
    {
      planet_name: 'Venus-like',
      planet_radius: 0.95,
      orbital_period: 225,
      equilibrium_temperature: 737,
      semi_major_axis: 0.72,
      stellar_luminosity: 1.0,
      stellar_mass: 1.0
    },
    {
      planet_name: 'Jupiter',
      planet_radius: 11.2,
      orbital_period: 4333,
      equilibrium_temperature: 110,
      semi_major_axis: 5.2,
      stellar_luminosity: 1.0,
      stellar_mass: 1.0
    },
    {
      planet_name: 'Mercury',
      planet_radius: 0.38,
      orbital_period: 88,
      equilibrium_temperature: 440,
      semi_major_axis: 0.39,
      stellar_luminosity: 1.0,
      stellar_mass: 1.0
    },
    {
      planet_name: '55 Cancri e',
      planet_radius: 1.88,
      orbital_period: 0.74,
      equilibrium_temperature: 2400,
      semi_major_axis: 0.0154,
      stellar_luminosity: 0.582,
      stellar_mass: 0.905
    },
    {
      planet_name: 'WASP-12b',
      planet_radius: 15.4,
      orbital_period: 1.09,
      equilibrium_temperature: 2500,
      semi_major_axis: 0.0229,
      stellar_luminosity: 1.657,
      stellar_mass: 1.35
    },
    {
      planet_name: 'HD 189733b',
      planet_radius: 12.7,
      orbital_period: 2.2,
      equilibrium_temperature: 1200,
      semi_major_axis: 0.031,
      stellar_luminosity: 0.36,
      stellar_mass: 0.846
    },
    {
      planet_name: 'Kepler-7b',
      planet_radius: 14.6,
      orbital_period: 4.9,
      equilibrium_temperature: 1540,
      semi_major_axis: 0.062,
      stellar_luminosity: 3.9,
      stellar_mass: 1.36
    }
  ];
  
  function loadSample(index) {
    const s = SAMPLE_PLANETS[index];
    if (!s) return;
    document.getElementById('planet_name').value             = s.planet_name;
    document.getElementById('planet_radius').value           = s.planet_radius;
    document.getElementById('orbital_period').value          = s.orbital_period;
    document.getElementById('equilibrium_temperature').value = s.equilibrium_temperature;
    document.getElementById('semi_major_axis').value         = s.semi_major_axis;
    document.getElementById('stellar_luminosity').value      = s.stellar_luminosity;
    document.getElementById('stellar_mass').value            = s.stellar_mass;
  }

// ── Form Submit ────────────────────────────────────────────
document.getElementById('predictForm')
  .addEventListener('submit', async function (e) {
    e.preventDefault();

    const inputData = {
      planet_name:             document.getElementById('planet_name').value || 'Unknown',
      planet_radius:           document.getElementById('planet_radius').value,
      orbital_period:          document.getElementById('orbital_period').value,
      equilibrium_temperature: document.getElementById('equilibrium_temperature').value,
      semi_major_axis:         document.getElementById('semi_major_axis').value,
      stellar_luminosity:      document.getElementById('stellar_luminosity').value,
      stellar_mass:            document.getElementById('stellar_mass').value
    };

    // Validation 
    const rules = {
      planet_radius:           [0.1, 30],
      orbital_period:          [0.1, 100000],
      equilibrium_temperature: [50, 1000],
      semi_major_axis:         [0.001, 100],
      stellar_luminosity:      [0.0001, 1000],
      stellar_mass:            [0.01, 150]
    };

    let valid = true;
    Object.entries(rules).forEach(([field, [min, max]]) => {
      const el  = document.getElementById(field);
      const val = parseFloat(el.value);
      if (!el.value || isNaN(val) || val < min || val > max) {
        el.classList.add('is-invalid');
        valid = false;
      } else {
        el.classList.remove('is-invalid');
      }
    });

    if (!valid) {
      showState('error');
      document.getElementById('errorMessage').textContent =
        'Please fill all required fields with valid values.';
      return;
    }

    showState('loading');

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(inputData)
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Backend error');
      }

      displayResults(result, inputData);
      addToHistory(inputData, result);
      showState('results');

    } catch (err) {
      showState('error');
      if (err.message.includes('fetch')) {
        document.getElementById('errorMessage').textContent =
          '❌ Cannot connect to backend. Make sure Flask is running on port 5000.';
      } else {
        document.getElementById('errorMessage').textContent = err.message;
      }
    }
  });

// ── Results Display  ─────────────────────────────────
function displayResults(result, inputData) {
  const isHabitable = result.prediction === 1;
  const score       = result.habitability_score;

  // Planet icon
  document.getElementById('resultIcon').textContent =
    isHabitable ? '🌍' : '🔴';

  // Status
  const statusEl = document.getElementById('resultStatus');
  statusEl.textContent = result.habitability_status;
  statusEl.style.color = isHabitable
    ? 'var(--accent-green)' : 'var(--accent-red)';

  // Confidence badge
  const badge = document.getElementById('confidenceBadge');
  badge.textContent  = result.confidence;
  badge.style.background = isHabitable
    ? 'rgba(74,222,128,0.2)' : 'rgba(248,113,113,0.2)';
  badge.style.color = isHabitable
    ? 'var(--accent-green)' : 'var(--accent-red)';

  // Score bar animate 
  const bar = document.getElementById('scoreBar');
  setTimeout(() => {
    bar.style.width       = score + '%';
    bar.style.background  = isHabitable
      ? 'linear-gradient(90deg,#4ade80,#22c55e)'
      : 'linear-gradient(90deg,#f87171,#ef4444)';
  }, 100);

  document.getElementById('scoreLabel').textContent = score + '%';
  document.getElementById('scoreBig').textContent   = score + '%';
  document.getElementById('scoreBig').style.color   = isHabitable
    ? 'var(--accent-green)' : 'var(--accent-red)';

  // Data table
  document.getElementById('resultTableBody').innerHTML = `
    <tr><td><strong>Planet Radius</strong></td>
        <td>${inputData.planet_radius} R⊕</td></tr>
    <tr><td><strong>Orbital Period</strong></td>
        <td>${inputData.orbital_period} days</td></tr>
    <tr><td><strong>Temperature</strong></td>
        <td>${inputData.equilibrium_temperature} K</td></tr>
    <tr><td><strong>Semi-Major Axis</strong></td>
        <td>${inputData.semi_major_axis} AU</td></tr>
    <tr><td><strong>Stellar Luminosity</strong></td>
        <td>${inputData.stellar_luminosity} L☉</td></tr>
    <tr><td><strong>Stellar Mass</strong></td>
        <td>${inputData.stellar_mass} M☉</td></tr>
  `;

  // Explanation
  document.getElementById('explanationText').textContent = isHabitable
    ? '✅ This planet falls within parameters consistent with liquid water and potential atmospheric stability. The temperature, size, and stellar conditions are favorable for life as we know it.'
    : '❌ This planet\'s parameters fall outside the conventional habitable zone. The temperature, size, or stellar conditions make it unlikely to support life as we know it.';
}

// ── History Table ──────────────────────────────────────────
function addToHistory(inputData, result) {
  predictionCount++;
  const isHabitable = result.prediction === 1;
  const now = new Date();
  const time = now.toTimeString().slice(0, 8);

  const noRow = document.getElementById('noHistoryRow');
  if (noRow) noRow.style.display = 'none';

  const tbody = document.getElementById('historyBody');

  // Max 10 rows 
  const rows = tbody.querySelectorAll('tr:not(#noHistoryRow)');
  if (rows.length >= 10) rows[rows.length - 1].remove();

  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${predictionCount}</td>
    <td>${inputData.planet_name}</td>
    <td>${inputData.planet_radius}</td>
    <td>${inputData.equilibrium_temperature}</td>
    <td>${inputData.orbital_period}</td>
    <td><strong>${result.habitability_score}%</strong></td>
    <td>
      <span class="badge"
            style="background:${isHabitable
              ? 'rgba(74,222,128,0.2);color:#4ade80'
              : 'rgba(248,113,113,0.2);color:#f87171'}">
        ${isHabitable ? '✅ Habitable' : '❌ Not Habitable'}
      </span>
    </td>
    <td><small>${time}</small></td>
  `;
  tbody.prepend(row);
}

// ── Helper Functions ───────────────────────────────────────
function showState(state) {
  ['placeholderState','loadingState','errorState','resultsState']
    .forEach(id => document.getElementById(id).classList.add('d-none'));

  const btn = document.getElementById('submitBtn');

  if (state === 'loading') {
    document.getElementById('loadingState').classList.remove('d-none');
    btn.disabled    = true;
    btn.innerHTML   = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
  } else {
    btn.disabled  = false;
    btn.innerHTML = '<i class="fas fa-rocket me-2"></i> Predict Habitability';
    document.getElementById(state + 'State').classList.remove('d-none');
  }
}

function resetToPlaceholder() {
  showState('placeholder');
}

function resetForm() {
  document.getElementById('predictForm').reset();
  ['planet_radius','orbital_period','equilibrium_temperature',
   'semi_major_axis','stellar_luminosity','stellar_mass']
    .forEach(id => document.getElementById(id).classList.remove('is-invalid'));
  showState('placeholder');
  document.getElementById('scoreBar').style.width = '0%';
}

function clearHistory() {
  const tbody = document.getElementById('historyBody');
  tbody.innerHTML = `
    <tr id="noHistoryRow">
      <td colspan="8" class="text-center text-muted py-4">No predictions yet</td>
    </tr>
  `;
  predictionCount = 0;
}