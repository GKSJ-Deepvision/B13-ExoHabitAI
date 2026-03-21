/**
 * ExoHabit — Frontend JavaScript
 * Handles API calls, form validation, result display, and rankings.
 */

// ═══════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════

const API_BASE_URL = 'http://127.0.0.1:5000';


const NUMERIC_FEATURES = [
    'Planet_Radius',
    'Planet_Mass',
    'Orbital_Period',
    'Semi_Major_Axis',
    'Equilibrium_Temp',
    'Planet_Density',
    'Stellar_Temp',
    'Stellar_Luminosity',
    'Stellar_Metallicity'
];


const EXAMPLE_HABITABLE = {
    Planet_Radius: 2.759,
    Planet_Mass: 6.63,
    Orbital_Period: 9.15059,
    Semi_Major_Axis: 0.0814,
    Equilibrium_Temp: 828,
    Planet_Density: 1.73,
    Stellar_Temp: 5320,
    Stellar_Luminosity: 0.519,
    Stellar_Metallicity: -0.02,
    StarType: 'K'
};

const EXAMPLE_NOT_HABITABLE = {
    Planet_Radius: 16.14,
    Planet_Mass: 327.35,
    Orbital_Period: 1.51,
    Semi_Major_Axis: 0.054,
    Equilibrium_Temp: 1898,
    Planet_Density: 0.38,
    Stellar_Temp: 5950,
    Stellar_Luminosity: 0.941,
    Stellar_Metallicity: -0.3,
    StarType: 'G'
};


// ═══════════════════════════════════════════════
// DOM Elements
// ═══════════════════════════════════════════════

const predictionForm = document.getElementById('prediction-form');
const predictBtn = document.getElementById('predict-btn');
const resetBtn = document.getElementById('reset-btn');
const resultContainer = document.getElementById('result-container');
const resultContent = document.getElementById('result-content');
const errorContainer = document.getElementById('error-container');
const errorMessage = document.getElementById('error-message');
const loadingContainer = document.getElementById('loading-container');
const fillExampleBtn = document.getElementById('fill-example-btn');
const fillExampleBtn2 = document.getElementById('fill-example-btn-2');

const loadRankingsBtn = document.getElementById('load-rankings-btn');
const topNSelect = document.getElementById('top-n-select');
const rankingsContainer = document.getElementById('rankings-container');
const rankingsBody = document.getElementById('rankings-body');
const rankingsLoading = document.getElementById('rankings-loading');
const rankingsError = document.getElementById('rankings-error');
const rankingsErrorMessage = document.getElementById('rankings-error-message');


// ═══════════════════════════════════════════════
// Event Listeners
// ═══════════════════════════════════════════════

predictionForm.addEventListener('submit', handlePredict);
resetBtn.addEventListener('click', handleReset);
fillExampleBtn.addEventListener('click', () => fillForm(EXAMPLE_HABITABLE));
fillExampleBtn2.addEventListener('click', () => fillForm(EXAMPLE_NOT_HABITABLE));
loadRankingsBtn.addEventListener('click', handleLoadRankings);


// ═══════════════════════════════════════════════
// Prediction Logic
// ═══════════════════════════════════════════════

async function handlePredict(e) {
    e.preventDefault();

    
    hideElement(resultContainer);
    hideElement(errorContainer);

    
    if (!validateForm()) {
        return;
    }

    
    const payload = buildPayload();
    if (!payload) return;

    
    showElement(loadingContainer);
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            displayResult(data);
        } else {
            showError(data.message || 'Prediction failed. Please check your inputs.');
        }
    } catch (err) {
        showError(
            'Could not connect to the backend server. ' +
            'Please ensure the Flask API is running at ' + API_BASE_URL + '. ' +
            'Start it with: python backend/app.py'
        );
        console.error('Prediction error:', err);
    } finally {
        hideElement(loadingContainer);
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="bi bi-cpu me-2"></i>Predict Habitability';
    }
}


function validateForm() {
    let isValid = true;

    
    NUMERIC_FEATURES.forEach(id => {
        const input = document.getElementById(id);
        const value = input.value.trim();

        if (value === '' || isNaN(parseFloat(value))) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
        }
    });

    
    const starType = document.getElementById('StarType');
    if (!starType.value) {
        starType.classList.add('is-invalid');
        isValid = false;
    } else {
        starType.classList.remove('is-invalid');
    }

    if (!isValid) {
        showError('Please fill in all required fields with valid numeric values.');
    }

    return isValid;
}


function buildPayload() {
    const payload = {};

    
    NUMERIC_FEATURES.forEach(id => {
        payload[id] = parseFloat(document.getElementById(id).value);
    });

    
    const starType = document.getElementById('StarType').value;
    payload['StarType_A'] = starType === 'A';
    payload['StarType_F'] = starType === 'F';
    payload['StarType_G'] = starType === 'G';
    payload['StarType_K'] = starType === 'K';
    payload['StarType_M'] = starType === 'M';

    return payload;
}


// ═══════════════════════════════════════════════
// Display Results
// ═══════════════════════════════════════════════

function displayResult(data) {
    const pred = data.prediction;
    const isHabitable = pred.habitable;
    const score = pred.habitability_score;
    const scorePercent = (score * 100).toFixed(2);
    const cssClass = isHabitable ? 'result-habitable' : 'result-not-habitable';
    const icon = isHabitable ? 'bi-check-circle-fill' : 'bi-x-circle-fill';
    const barClass = isHabitable ? 'habitable' : 'not-habitable';

    resultContent.innerHTML = `
        <div class="${cssClass}">
            <div class="result-header">
                <div class="result-icon"><i class="bi ${icon}"></i></div>
                <div class="result-label">${pred.label}</div>
                <div class="result-score">${data.message}</div>
            </div>

            <div class="score-bar-container">
                <div class="d-flex justify-content-between mb-1">
                    <small class="text-secondary">Habitability Score</small>
                    <small style="color: ${isHabitable ? 'var(--accent-green)' : 'var(--accent-red)'}; font-weight: 600;">
                        ${scorePercent}%
                    </small>
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill ${barClass}" style="width: 0%;" id="score-bar"></div>
                </div>
            </div>

            <div class="result-details">
                <div class="detail-item">
                    <div class="detail-label">Prediction</div>
                    <div class="detail-value">${pred.label}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Class</div>
                    <div class="detail-value">${pred.class}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Confidence</div>
                    <div class="detail-value">${pred.confidence}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Raw Score</div>
                    <div class="detail-value">${score.toFixed(6)}</div>
                </div>
            </div>
        </div>
    `;

    showElement(resultContainer);

    
    requestAnimationFrame(() => {
        setTimeout(() => {
            document.getElementById('score-bar').style.width = `${scorePercent}%`;
        }, 100);
    });

    
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
}


// ═══════════════════════════════════════════════
// Rankings Logic
// ═══════════════════════════════════════════════

async function handleLoadRankings() {
    hideElement(rankingsContainer);
    hideElement(rankingsError);
    showElement(rankingsLoading);
    loadRankingsBtn.disabled = true;

    const topN = parseInt(topNSelect.value);
    let url = `${API_BASE_URL}/rank`;
    if (topN > 0) {
        url += `?top_n=${topN}`;
    }

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (response.ok && data.status === 'success') {
            displayRankings(data.rankings);
        } else {
            showRankingsError(data.message || 'Failed to load rankings.');
        }
    } catch (err) {
        showRankingsError(
            'Could not connect to the backend server. ' +
            'Ensure the Flask API is running at ' + API_BASE_URL
        );
        console.error('Rankings error:', err);
    } finally {
        hideElement(rankingsLoading);
        loadRankingsBtn.disabled = false;
    }
}


function displayRankings(rankings) {
    rankingsBody.innerHTML = '';

    rankings.forEach(item => {
        const row = document.createElement('tr');
        const isTop3 = item.rank <= 3;

        row.innerHTML = `
            <td><span class="rank-number ${isTop3 ? 'top-3' : ''}">#${item.rank}</span></td>
            <td><strong>${(item.habitability_score * 100).toFixed(3)}%</strong></td>
            <td>${item.planet_radius}</td>
            <td>${item.planet_mass.toFixed(2)}</td>
            <td>${item.orbital_period.toFixed(2)}</td>
            <td>${item.equilibrium_temp.toFixed(0)}</td>
            <td>${item.planet_density.toFixed(2)}</td>
            <td>${item.stellar_temp.toFixed(0)}</td>
            <td><span class="badge-habitable">Habitable</span></td>
        `;

        rankingsBody.appendChild(row);
    });

    showElement(rankingsContainer);
    rankingsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


// ═══════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════

function fillForm(exampleData) {
    NUMERIC_FEATURES.forEach(id => {
        const input = document.getElementById(id);
        if (exampleData[id] !== undefined) {
            input.value = exampleData[id];
            input.classList.remove('is-invalid');
        }
    });

    if (exampleData.StarType) {
        const select = document.getElementById('StarType');
        select.value = exampleData.StarType;
        select.classList.remove('is-invalid');
    }

    
    hideElement(resultContainer);
    hideElement(errorContainer);
}

function handleReset() {
    hideElement(resultContainer);
    hideElement(errorContainer);
    hideElement(loadingContainer);

    
    document.querySelectorAll('.custom-input').forEach(el => {
        el.classList.remove('is-invalid');
    });
}

function showError(msg) {
    errorMessage.innerHTML = `<i class="bi bi-exclamation-triangle me-2"></i>${msg}`;
    showElement(errorContainer);
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function showRankingsError(msg) {
    rankingsErrorMessage.innerHTML = `<i class="bi bi-exclamation-triangle me-2"></i>${msg}`;
    showElement(rankingsError);
}

function showElement(el) {
    el.style.display = 'block';
}

function hideElement(el) {
    el.style.display = 'none';
}
