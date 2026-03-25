// Global Variables
let barChart;
let polarChart;  // ← replaced radarChart
let isAnalyzing = false;
let currentPredictionResult = null;

// DOM Elements
const form = document.getElementById('planetForm');
const predictBtn = document.getElementById('predictBtn');
const btnText = predictBtn.querySelector('.btn-text');
const btnLoading = predictBtn.querySelector('.btn-loading');
const scoreCircle = document.getElementById('scoreCircle');
const scoreValue = scoreCircle.querySelector('.score-value');
const statusTitle = document.getElementById('statusTitle');
const statusIcon = document.querySelector('.status-icon i');
const rankingBody = document.getElementById('rankingBody');

// Slider Elements
const sliders = ['massSlider', 'periodSlider', 'distanceSlider', 'tempSlider', 'radiusSlider'];
const valueDisplays = ['massValue', 'periodValue', 'distanceValue', 'tempValue', 'radiusValue'];

// Preset Configurations
const PRESETS = {
    earth: { mass: 1.0, period: 365, distance: 1.0, temp: 5772, radius: 1.0 },
    mars:  { mass: 0.107, period: 687, distance: 1.524, temp: 5772, radius: 0.53 },
    venus: { mass: 0.815, period: 225, distance: 0.723, temp: 5772, radius: 0.95 }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', function () {
    initializeSliders();
    initializeCharts();
    loadRankingData();
    loadAnalyticsCharts();
    setupEventListeners();
    updatePolarChart(); // Initial update
});

// Initialize Sliders
function initializeSliders() {
    sliders.forEach((sliderId, index) => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(valueDisplays[index]);

        valueDisplay.textContent = slider.value;

        slider.addEventListener('input', function () {
            valueDisplay.textContent = this.value;
            updatePolarChart();
        });
    });
}

// Initialize Charts
function initializeCharts() {
    // Bar Chart (unchanged)
    const barCtx = document.getElementById('habitabilityChart').getContext('2d');
    barChart = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: ['Mass', 'Orbit', 'Temperature', 'Stellar Age', 'Atmosphere', 'Field'],
            datasets: [{
                label: 'Habitability Factors',
                data: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                backgroundColor: [
                    'rgba(99, 102, 241, 0.7)',
                    'rgba(139, 92, 246, 0.7)',
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(245, 158, 11, 0.7)',
                    'rgba(239, 68, 68, 0.7)'
                ],
                borderColor: [
                    '#6366f1', '#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444'
                ],
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#94a3b8', callback: value => value.toFixed(1) }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 11 } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: { label: context => `${context.dataset.label}: ${context.raw.toFixed(2)}` }
                }
            }
        }
    });

    // Polar Area Chart (new impressive feature)
    const polarCtx = document.getElementById('parameterPolar').getContext('2d');
    polarChart = new Chart(polarCtx, {
        type: 'polarArea',
        data: {
            labels: ['Mass', 'Orbit Period', 'Distance', 'Star Temp', 'Star Radius'],
            datasets: [{
                label: 'Current Parameters',
                data: [0.5, 0.5, 0.5, 0.5, 0.5],
                backgroundColor: [
                    'rgba(99, 102, 241, 0.55)',
                    'rgba(139, 92, 246, 0.55)',
                    'rgba(59, 130, 246, 0.55)',
                    'rgba(16, 185, 129, 0.55)',
                    'rgba(245, 158, 11, 0.55)'
                ],
                borderColor: [
                    '#6366f1',
                    '#8b5cf6',
                    '#3b82f6',
                    '#10b981',
                    '#f59e0b'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2,
                        color: 'rgba(255, 255, 255, 0.6)',
                        backdropColor: 'transparent'
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.15)' }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: 'rgba(255, 255, 255, 0.9)',
                        font: { size: 12 }
                    }
                }
            }
        }
    });
}

// Update Polar Chart
function updatePolarChart() {
    if (!polarChart) return;

    const mass = parseFloat(document.getElementById('massSlider').value);
    const period = parseFloat(document.getElementById('periodSlider').value);
    const distance = parseFloat(document.getElementById('distanceSlider').value);
    const temp = parseFloat(document.getElementById('tempSlider').value);
    const radius = parseFloat(document.getElementById('radiusSlider').value);

    const normalized = [
        Math.min(mass / 20, 1),
        Math.min(period / 1000, 1),
        Math.min(distance / 10, 1),
        Math.min(temp / 10000, 1),
        Math.min(radius / 10, 1)
    ];

    polarChart.data.datasets[0].data = normalized;
    polarChart.update();
}

// Setup Event Listeners
function setupEventListeners() {
    form.addEventListener('submit', handlePrediction);
    
    // Export buttons
    document.getElementById("exportExcel")?.addEventListener("click", function() {
        window.open("/export/excel", "_blank");
    });

    document.getElementById("exportPDF")?.addEventListener("click", function() {
        window.open("/export/pdf", "_blank");
    });
}

// Handle Prediction
async function handlePrediction(e) {
    e.preventDefault();

    if (isAnalyzing) return;
    isAnalyzing = true;
    showLoadingState();

    const planetName = document.getElementById('planetName').value.trim() || "Unnamed Planet";

    const payload = {
        planet_mass_earth:   parseFloat(document.getElementById('massSlider').value),
        orbital_period_days: parseFloat(document.getElementById('periodSlider').value),
        orbit_distance_au:   parseFloat(document.getElementById('distanceSlider').value),
        star_temperature_k:  parseFloat(document.getElementById('tempSlider').value),
        star_radius_solar:   parseFloat(document.getElementById('radiusSlider').value)
    };

    try {
        const predictResponse = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!predictResponse.ok) {
            throw new Error(`Prediction failed: ${predictResponse.status}`);
        }

        const result = await predictResponse.json();

        const uiResult = {
            score: result.habitability_score,
            habitable: result.habitable === 1,
            category: getCategory(result.habitability_score),
            confidence: (result.habitability_score * 100).toFixed(1),
            factors: {
                mass: result.habitability_score,
                orbit: result.habitability_score,
                temperature: result.habitability_score,
                stellar_age: result.habitability_score * 0.9,
                atmosphere: result.habitability_score * 0.85,
                magnetic_field: result.habitability_score * 0.8
            },
            name: planetName,
            payload: payload
        };

        currentPredictionResult = uiResult;

        displayResults(uiResult);
        showSaveButton();

    } catch (error) {
        console.error(error);
        showErrorState("Failed to connect to backend API");
    } finally {
        isAnalyzing = false;
        hideLoadingState();
    }
}

// Show "Save to Ranking" button
function showSaveButton() {
    let saveBtn = document.getElementById('saveToRankingBtn');

    if (!saveBtn) {
        saveBtn = document.createElement('button');
        saveBtn.id = 'saveToRankingBtn';
        saveBtn.type = 'button';
        saveBtn.className = 'btn btn-success btn-lg py-3 mt-3';
        saveBtn.innerHTML = '<i class="fas fa-save me-2"></i>Save to Ranking';

        const actionDiv = document.querySelector('#planetForm .d-grid');
        if (actionDiv) actionDiv.appendChild(saveBtn);
    }

    saveBtn.style.display = 'block';
    saveBtn.onclick = handleSaveToRanking;
}

// Manual save handler
async function handleSaveToRanking() {
    if (!currentPredictionResult) return;

    const savePayload = {
        name: currentPredictionResult.name,
        planet_mass_earth: parseFloat(currentPredictionResult.payload.planet_mass_earth),
        orbital_period_days: parseFloat(currentPredictionResult.payload.orbital_period_days),
        orbit_distance_au: parseFloat(currentPredictionResult.payload.orbit_distance_au),
        star_temperature_k: parseFloat(currentPredictionResult.payload.star_temperature_k),
        star_radius_solar: parseFloat(currentPredictionResult.payload.star_radius_solar),
        habitability_score: parseFloat(currentPredictionResult.score),
        is_habitable: currentPredictionResult.habitable ? 1 : 0,
        category: currentPredictionResult.category
    };

    console.log("Saving payload:", savePayload);

    try {
        const response = await fetch("/add_planet", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(savePayload)
        });

        const result = await response.json();
        console.log("Save response:", result);

        if (response.ok) {
            alert("Planet saved successfully!");
            loadRankingData();
            loadAnalyticsCharts();
            document.getElementById('saveToRankingBtn').style.display = 'none';
            currentPredictionResult = null;
        } else {
            alert("Save failed: " + (result.error || "Unknown error"));
        }
    } catch (err) {
        console.error("Save error:", err);
        alert("Error connecting to server");
    }
}

function displayResults(result) {
    updateScoreCircle(result.score);
    updateStatus(result.habitable, result.category, result.confidence);
    updateChart(result.factors);
    updateAnalysisText(result);
    updatePolarChart(); // Updated call
}

function updateScoreCircle(score) {
    const formattedScore = score.toFixed(3);
    scoreValue.textContent = formattedScore;

    const hue = score * 120;
    scoreCircle.style.background = `conic-gradient(
        hsl(${hue}, 70%, 50%) 0deg,
        hsl(${hue}, 70%, 50%) ${score * 360}deg,
        rgba(255,255,255,0.1) ${score * 360}deg,
        rgba(255,255,255,0.1) 360deg
    )`;

    const catEl = document.getElementById('scoreCategory');
    if (catEl) {
        catEl.textContent = getCategory(score);
        catEl.className = `badge bg-${score >= 0.7 ? 'success' : score >= 0.5 ? 'warning' : 'danger'}`;
    }
}

function updateStatus(isHabitable, category, confidence) {
    const config = {
        'Earth-like':          { icon: 'fa-seedling',        color: 'success', title: 'Highly Habitable' },
        'Potentially Habitable': { icon: 'fa-leaf',           color: 'success', title: 'Potentially Habitable' },
        'Marginal':            { icon: 'fa-temperature-high', color: 'warning', title: 'Marginal Habitability' },
        'Hostile':             { icon: 'fa-skull-crossbones', color: 'danger',  title: 'Not Habitable' }
    }[category] || { icon: 'fa-temperature-high', color: 'warning', title: 'Marginal Habitability' };

    statusIcon.className = `fas ${config.icon} fa-3x`;
    statusIcon.style.color = `var(--${config.color})`;
    statusTitle.textContent = config.title;
    statusTitle.className = `fw-bold text-${config.color}`;

    document.getElementById('confidenceValue').textContent = `${confidence}%`;
    const bar = document.getElementById('confidenceBar');
    bar.style.width = `${confidence}%`;
    bar.className = `progress-bar bg-${config.color}`;
}

function updateChart(factors) {
    barChart.data.datasets[0].data = [
        factors.mass, factors.orbit, factors.temperature,
        factors.stellar_age, factors.atmosphere, factors.magnetic_field
    ];
    barChart.update();
}

function updateAnalysisText(result) {
    const el = document.getElementById('habitabilityAnalysis');
    if (!el) return;

    let text = '', cls = '';

    if (result.score >= 0.8) {
        text = `This planet shows <strong>strong Earth-like characteristics</strong>. Score: ${result.score.toFixed(3)}.`;
        cls = 'success';
    } else if (result.score >= 0.7) {
        text = `Potentially habitable. Score: ${result.score.toFixed(3)}.`;
        cls = 'info';
    } else if (result.score >= 0.5) {
        text = `Marginal habitability. Score: ${result.score.toFixed(3)}.`;
        cls = 'warning';
    } else {
        text = `Likely uninhabitable. Score: ${result.score.toFixed(3)}.`;
        cls = 'danger';
    }

    el.innerHTML = `<div class="alert alert-${cls}"><i class="fas fa-microscope me-2"></i>${text}</div>`;
}
async function loadRankingData() {
    try {
        const res = await fetch("/ranking");
        if (!res.ok) throw new Error(`Failed to fetch ranking: ${res.status}`);

        const planets = await res.json();
        console.log("Ranking data:", planets);  // debug - check this in browser console

        const tbody = document.getElementById('rankingBody');
        tbody.innerHTML = '';

        if (!planets || planets.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No planets ranked yet</td></tr>';
            return;
        }

        planets.forEach(p => {
            const score = parseFloat(p.habitability_score ?? p.score ?? 0);
            const color = score >= 0.8 ? 'success' : score >= 0.7 ? 'warning' : 'danger';

            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="fw-bold">${p.name || 'Unnamed'}</td>
                <td>
                    <div class="d-flex align-items-center">
                        <div class="progress me-2" style="width:120px;height:8px">
                            <div class="progress-bar bg-${color}" style="width:${score * 100}%"></div>
                        </div>
                        <span class="ms-2 fw-bold">${score.toFixed(3)}</span>
                    </div>
                </td>
                <td>${p.orbit_distance_au ? p.orbit_distance_au.toFixed(3) + ' AU' : '—'}</td>
                <td><span class="badge bg-dark">${p.category || 'Custom'}</span></td>
                <td>—</td>  <!-- No created_at column, so show dash -->
            `;
            tbody.appendChild(row);
        });

    } catch (err) {
        console.error("Ranking error:", err);
        document.getElementById('rankingBody').innerHTML = 
            '<tr><td colspan="5" class="text-center text-danger">Could not load ranking data</td></tr>';
    }
}async function loadAnalyticsCharts() {
    try {
        const resp = await fetch("/analytics_charts");
        if (!resp.ok) throw new Error("Charts fetch failed");

        const data = await resp.json();

        const loadingEl  = document.getElementById('analyticsLoading');
        const noDataEl   = document.getElementById('noDataMessage');
        const chartsEl   = document.getElementById('chartsContainer');

        if (!loadingEl || !noDataEl || !chartsEl) {
            console.warn("Analytics DOM elements not found");
            return;
        }

        loadingEl.classList.add('d-none');

        if (!data.has_data) {
            noDataEl.classList.remove('d-none');
            chartsEl.classList.add('d-none');
            return;
        }

        noDataEl.classList.add('d-none');
        chartsEl.classList.remove('d-none');

        if (data.feature_importance) {
            document.getElementById('featImg').src = `data:image/png;base64,${data.feature_importance}`;
        }
        if (data.score_distribution) {
            document.getElementById('distImg').src = `data:image/png;base64,${data.score_distribution}`;
        }
        if (data.correlation) {
            document.getElementById('corrImg').src = `data:image/png;base64,${data.correlation}`;
        }

    } catch (err) {
        console.error("Analytics load error:", err);
        const loadingEl = document.getElementById('analyticsLoading');
        if (loadingEl) {
            loadingEl.innerHTML = '<div class="alert alert-danger">Could not load analytics charts</div>';
        }
    }
}

function showLoadingState() {
    btnText.classList.add('d-none');
    btnLoading.classList.remove('d-none');
    predictBtn.disabled = true;
}

function hideLoadingState() {
    btnText.classList.remove('d-none');
    btnLoading.classList.add('d-none');
    predictBtn.disabled = false;
}

function showErrorState(message) {
    const el = document.getElementById('habitabilityAnalysis');
    if (el) {
        el.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-triangle me-2"></i>${message}</div>`;
    }
}

function getCategory(score) {
    if (score >= 0.8) return 'Earth-like';
    if (score >= 0.7) return 'Potentially Habitable';
    if (score >= 0.5) return 'Marginal';
    return 'Hostile';
}

// Make functions available globally for tab switching
window.loadRankingData = loadRankingData;
window.loadAnalyticsCharts = loadAnalyticsCharts;