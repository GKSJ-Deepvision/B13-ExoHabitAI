// ============================================================
// STORAGE UTILITIES
// ============================================================

class Storage {
  static USERS_KEY = 'exohabitat_users';
  static CURRENT_USER_KEY = 'exohabitat_current_user';
  static PREDICTIONS_KEY = 'exohabitat_predictions';

  static getUserFromStorage(email, password) {
    const users = JSON.parse(localStorage.getItem(this.USERS_KEY) || '[]');
    const userRecord = users.find(u => u.email === email && u.password === password);
    return userRecord?.user || null;
  }

  static saveUserToStorage(email, password, name) {
    const users = JSON.parse(localStorage.getItem(this.USERS_KEY) || '[]');
    
    const user = {
      id: Date.now().toString(),
      email,
      name,
      createdAt: new Date().toISOString(),
    };

    users.push({ email, password, user });
    localStorage.setItem(this.USERS_KEY, JSON.stringify(users));
    localStorage.setItem(this.CURRENT_USER_KEY, JSON.stringify(user));
    
    return user;
  }

  static setCurrentUser(user) {
    if (user) {
      localStorage.setItem(this.CURRENT_USER_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(this.CURRENT_USER_KEY);
    }
  }

  static getCurrentUser() {
    const stored = localStorage.getItem(this.CURRENT_USER_KEY);
    return stored ? JSON.parse(stored) : null;
  }

  static emailExists(email) {
    const users = JSON.parse(localStorage.getItem(this.USERS_KEY) || '[]');
    return users.some(u => u.email === email);
  }

  static savePrediction(prediction) {
    const predictions = JSON.parse(localStorage.getItem(this.PREDICTIONS_KEY) || '[]');
    predictions.push(prediction);
    localStorage.setItem(this.PREDICTIONS_KEY, JSON.stringify(predictions));
  }

  static getPredictionsForUser(userId) {
    const predictions = JSON.parse(localStorage.getItem(this.PREDICTIONS_KEY) || '[]');
    return predictions
      .filter(p => p.userId === userId)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }

  static getAllPredictions() {
    return JSON.parse(localStorage.getItem(this.PREDICTIONS_KEY) || '[]');
  }

  static deletePrediction(id) {
    const predictions = JSON.parse(localStorage.getItem(this.PREDICTIONS_KEY) || '[]');
    const filtered = predictions.filter(p => p.id !== id);
    localStorage.setItem(this.PREDICTIONS_KEY, JSON.stringify(filtered));
  }

  static initializeDemoAccount() {
    const users = JSON.parse(localStorage.getItem(this.USERS_KEY) || '[]');
    if (users.length === 0) {
      this.saveUserToStorage('demo@exohabitat.com', 'demo123', 'Demo Astronomer');
    }
  }
}

// ============================================================
// HABITABILITY CALCULATOR
// ============================================================

class HabitabilityCalculator {
  static calculateHabitability(params) {
    const sizeScore = this.calculateSizeFactor(params.radius);
    const orbitScore = this.calculateOrbitFactor(params.distance);
    const atmosphereScore = params.atmosphere ? 25 : 0;
    const temperatureScore = this.calculateTemperatureFactor(params.temperature);
    const waterScore = Math.min(params.waterPresence, 100) * 0.25;
    
    const totalScore = (sizeScore + orbitScore + atmosphereScore + temperatureScore + waterScore) / 1.25;
    const finalScore = Math.min(Math.round(totalScore), 100);
    
    const classification = this.getClassification(finalScore);
    
    const factors = {
      size: {
        score: sizeScore,
        description: this.getSizeDescription(params.radius)
      },
      orbit: {
        score: orbitScore,
        description: this.getOrbitDescription(params.distance)
      },
      atmosphere: {
        score: atmosphereScore,
        description: params.atmosphere ? 'Atmosphere detected' : 'No atmosphere detected'
      },
      temperature: {
        score: temperatureScore,
        description: this.getTemperatureDescription(params.temperature)
      },
      water: {
        score: waterScore,
        description: `Water presence probability: ${params.waterPresence.toFixed(1)}%`
      }
    };
    
    const recommendation = this.generateRecommendation(finalScore, classification, params);
    
    return {
      score: finalScore,
      classification,
      factors,
      recommendation
    };
  }

  static calculateSizeFactor(radius) {
    const optimal = 1.0;
    const deviation = Math.abs(radius - optimal);
    
    if (deviation < 0.2) return 25;
    if (deviation < 0.5) return 20;
    if (deviation < 1.0) return 12;
    return 5;
  }

  static calculateOrbitFactor(distance) {
    if (distance >= 0.95 && distance <= 1.37) return 25;
    if (distance >= 0.8 && distance <= 1.5) return 20;
    if (distance >= 0.7 && distance <= 1.7) return 12;
    return 5;
  }

  static calculateTemperatureFactor(temp) {
    const optimalMin = 250;
    const optimalMax = 310;
    const acceptableMin = 273;
    const acceptableMax = 373;
    
    if (temp >= optimalMin && temp <= optimalMax) return 25;
    if (temp >= acceptableMin && temp <= acceptableMax) return 20;
    if (temp >= 240 && temp <= 390) return 12;
    return 5;
  }

  static getSizeDescription(radius) {
    if (radius < 0.5) return 'Too small for significant atmosphere';
    if (radius < 0.8) return 'Small but potentially habitable';
    if (radius <= 1.2) return 'Optimal size range';
    if (radius < 2.0) return 'Larger Super-Earth, possibly habitable';
    return 'Too large, likely a mini-Neptune';
  }

  static getOrbitDescription(distance) {
    if (distance < 0.8) return 'Too close to star - likely too hot';
    if (distance >= 0.95 && distance <= 1.37) return 'Optimal habitable zone';
    if (distance <= 1.5) return 'Within habitable zone';
    if (distance <= 2.0) return 'Edge of habitable zone';
    return 'Too far from star - likely too cold';
  }

  static getTemperatureDescription(temp) {
    if (temp < 200) return 'Extremely cold';
    if (temp < 250) return 'Very cold';
    if (temp >= 250 && temp <= 310) return 'Optimal temperature range';
    if (temp <= 373) return 'Acceptable for liquid water';
    if (temp <= 450) return 'Very hot';
    return 'Extremely hot';
  }

  static getClassification(score) {
    if (score >= 80) return 'Highly Habitable';
    if (score >= 60) return 'Potentially Habitable';
    if (score >= 40) return 'Marginally Habitable';
    if (score >= 20) return 'Unlikely Habitable';
    return 'Not Habitable';
  }

  static generateRecommendation(score, classification, params) {
    if (score >= 80) {
      return `This exoplanet shows excellent potential for habitability. With ${classification.toLowerCase()} characteristics and a habitability index of ${score}, it should be prioritized for further study.`;
    } else if (score >= 60) {
      return `This exoplanet demonstrates promising habitability factors. While classified as ${classification.toLowerCase()}, targeted observation could reveal more about its potential to support life.`;
    } else if (score >= 40) {
      return `This exoplanet has some favorable conditions but significant challenges remain. Further research is needed to determine if life could adapt to these conditions.`;
    } else {
      return `This exoplanet presents considerable challenges to habitability. Current parameters suggest unfavorable conditions for known life forms.`;
    }
  }
}

// ============================================================
// UI UTILITIES
// ============================================================

class UI {
  static showPage(pageName) {
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    const page = document.getElementById(`${pageName}-page`);
    if (page) page.classList.add('active');
  }

  static showToast(message, type = 'info', duration = 2000) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
      toast.style.animation = 'fadeOut 0.3s ease-out forwards';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  static updateNavigation() {
    const currentUser = Storage.getCurrentUser();
    const userEmail = document.getElementById('userEmail');
    const authToggle = document.getElementById('authToggle');
    const exploreLink = document.getElementById('exploreLink');
    const predictorLink = document.getElementById('predictorLink');
    const dashboardLink = document.getElementById('dashboardLink');
    
    if (currentUser) {
      userEmail.textContent = currentUser.email;
      userEmail.style.display = 'inline';
      authToggle.textContent = 'Sign Out';
      exploreLink.style.display = 'inline-block';
      predictorLink.style.display = 'inline-block';
      dashboardLink.style.display = 'inline-block';
    } else {
      userEmail.style.display = 'none';
      authToggle.textContent = 'Sign In';
      exploreLink.style.display = 'none';
      predictorLink.style.display = 'none';
      dashboardLink.style.display = 'none';
    }
  }

  static renderPlanets(planets, container) {
    container.innerHTML = '';
    planets.forEach(planet => {
      const card = document.createElement('div');
      card.className = 'planet-card';
      card.innerHTML = `
        <h3 class="planet-name">${planet.name}</h3>
        <p class="planet-star">Around ${planet.star}</p>
        
        <div class="planet-stats">
          <div class="planet-stat">
            <span class="planet-stat-label">Radius</span>
            <span class="planet-stat-value">${planet.radius}R⊕</span>
          </div>
          <div class="planet-stat">
            <span class="planet-stat-label">Distance</span>
            <span class="planet-stat-value">${planet.distance}AU</span>
          </div>
        </div>

        <div class="habitability-section">
          <div class="habitability-label">
            <span>Habitability Score</span>
            <span>${planet.habitabilityScore}%</span>
          </div>
          <div class="habitability-bar">
            <div class="habitability-fill" style="width: ${planet.habitabilityScore}%"></div>
          </div>
        </div>

        <div class="planet-attrs">
          ${planet.atmosphere ? '<span class="atmosphere-badge">Has Atmosphere</span>' : ''}
          <button class="favorite-btn" data-planet-id="${planet.id}">
            ${planet.favorite ? '❤️' : '🤍'}
          </button>
        </div>
      `;
      container.appendChild(card);
    });
  }
}

// ============================================================
// EXOPLANET DATABASE
// ============================================================

const EXOPLANETS = [
  {
    id: 'trappist1e',
    name: 'TRAPPIST-1e',
    star: 'TRAPPIST-1',
    radius: 0.92,
    distance: 0.02925,
    atmosphere: true,
    temperature: 280,
    waterPresence: 95,
    habitabilityScore: 86,
    atmosphere: true,
  },
  {
    id: 'kepler452b',
    name: 'Kepler-452b',
    star: 'Kepler-452',
    radius: 1.6,
    distance: 1.046,
    atmosphere: true,
    temperature: 265,
    waterPresence: 80,
    habitabilityScore: 78,
    atmosphere: true,
  },
  {
    id: 'proxcenb',
    name: 'Proxima Centauri b',
    star: 'Proxima Centauri',
    radius: 1.1,
    distance: 0.0485,
    atmosphere: true,
    temperature: 234,
    waterPresence: 60,
    habitabilityScore: 52,
    atmosphere: true,
  },
  {
    id: 'kepler186f',
    name: 'Kepler-186f',
    star: 'Kepler-186',
    radius: 1.11,
    distance: 0.432,
    atmosphere: true,
    temperature: 188,
    waterPresence: 75,
    habitabilityScore: 64,
    atmosphere: true,
  },
  {
    id: 'rossb',
    name: 'Ross 128 b',
    star: 'Ross 128',
    radius: 1.35,
    distance: 0.0485,
    atmosphere: false,
    temperature: 213,
    waterPresence: 45,
    habitabilityScore: 38,
    atmosphere: false,
  },
  {
    id: 'toi700e',
    name: 'TOI-700 e',
    star: 'TOI-700',
    radius: 1.08,
    distance: 0.619,
    atmosphere: true,
    temperature: 276,
    waterPresence: 72,
    habitabilityScore: 71,
    atmosphere: true,
  },
];

// ============================================================
// PAGE HANDLERS
// ============================================================

class PageHandlers {
  static initSignup() {
    const form = document.getElementById('signupForm');
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const name = document.getElementById('signupName').value;
      const email = document.getElementById('signupEmail').value;
      const password = document.getElementById('signupPassword').value;
      const confirm = document.getElementById('signupConfirm').value;
      const errorDiv = document.getElementById('signupError');
      
      errorDiv.style.display = 'none';
      
      if (password !== confirm) {
        errorDiv.textContent = 'Passwords do not match';
        errorDiv.style.display = 'block';
        return;
      }
      
      if (Storage.emailExists(email)) {
        errorDiv.textContent = 'Email already exists';
        errorDiv.style.display = 'block';
        return;
      }
      
      Storage.saveUserToStorage(email, password, name);
      UI.updateNavigation();
      UI.showToast('Account created successfully!', 'success');
      UI.showPage('home');
      form.reset();
    });
  }

  static initSignin() {
    const form = document.getElementById('signinForm');
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      
      const email = document.getElementById('signinEmail').value;
      const password = document.getElementById('signinPassword').value;
      const errorDiv = document.getElementById('signinError');
      
      errorDiv.style.display = 'none';
      
      const user = Storage.getUserFromStorage(email, password);
      if (!user) {
        errorDiv.textContent = 'Invalid email or password';
        errorDiv.style.display = 'block';
        return;
      }
      
      Storage.setCurrentUser(user);
      UI.updateNavigation();
      UI.showToast('Signed in successfully!', 'success');
      UI.showPage('home');
      form.reset();
    });
  }

  static initPredictor() {
    // Update range values display
    const radius = document.getElementById('radius');
    const distance = document.getElementById('distance');
    const temperature = document.getElementById('temperature');
    const waterPresence = document.getElementById('waterPresence');
    
    radius.addEventListener('change', () => {
      document.getElementById('radiusValue').textContent = parseFloat(radius.value).toFixed(2);
    });
    
    distance.addEventListener('change', () => {
      document.getElementById('distanceValue').textContent = parseFloat(distance.value).toFixed(2);
    });
    
    temperature.addEventListener('change', () => {
      document.getElementById('temperatureValue').textContent = temperature.value;
    });
    
    waterPresence.addEventListener('change', () => {
      document.getElementById('waterValue').textContent = parseFloat(waterPresence.value).toFixed(1);
    });
    
    // Predict button
    document.getElementById('predictBtn').addEventListener('click', () => {
      const planetName = document.getElementById('planetName').value;
      if (!planetName.trim()) {
        UI.showToast('Please enter a planet name', 'error');
        return;
      }
      
      const params = {
        radius: parseFloat(radius.value),
        distance: parseFloat(distance.value),
        atmosphere: document.getElementById('atmosphere').checked,
        temperature: parseInt(temperature.value),
        waterPresence: parseFloat(waterPresence.value),
      };
      
      const result = HabitabilityCalculator.calculateHabitability(params);
      this.displayResults(result, planetName);
    });
    
    // Save button
    document.getElementById('saveBtn').addEventListener('click', () => {
      const currentUser = Storage.getCurrentUser();
      if (!currentUser) {
        UI.showToast('You must be signed in to save predictions', 'error');
        return;
      }
      
      const result = this.lastResult;
      const prediction = {
        id: Date.now().toString(),
        userId: currentUser.id,
        planetName: document.getElementById('planetName').value,
        parameters: {
          radius: parseFloat(radius.value),
          distance: parseFloat(distance.value),
          atmosphere: document.getElementById('atmosphere').checked,
          temperature: parseInt(temperature.value),
          waterPresence: parseFloat(waterPresence.value),
        },
        habitabilityScore: result.score,
        classification: result.classification,
        timestamp: new Date().toISOString(),
      };
      
      Storage.savePrediction(prediction);
      UI.showToast(`${document.getElementById('planetName').value} saved to dashboard!`, 'success');
      setTimeout(() => UI.showPage('dashboard'), 500);
    });
    
    // New analysis button
    document.getElementById('newAnalysisBtn').addEventListener('click', () => {
      document.getElementById('planetName').value = '';
      document.getElementById('radius').value = 1.0;
      document.getElementById('distance').value = 1.0;
      document.getElementById('temperature').value = 288;
      document.getElementById('waterPresence').value = 50;
      document.getElementById('atmosphere').checked = true;
      document.getElementById('resultsPanel').style.display = 'none';
      document.getElementById('emptyState').style.display = 'block';
      
      document.getElementById('radiusValue').textContent = '1.00';
      document.getElementById('distanceValue').textContent = '1.00';
      document.getElementById('temperatureValue').textContent = '288';
      document.getElementById('waterValue').textContent = '50.0';
    });
  }

  static displayResults(result, planetName) {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('resultsPanel').style.display = 'block';
    
    document.getElementById('scoreValue').textContent = result.score;
    document.getElementById('classificationValue').textContent = result.classification;
    document.getElementById('recommendationText').textContent = result.recommendation;
    
    // Animate progress bar with smooth animation
    const progressBar = document.getElementById('scoreProgress');
    AnimationHelper.animateProgressBar(progressBar, result.score, 1000);
    
    // Render factors
    const factorsContainer = document.getElementById('factorsContainer');
    factorsContainer.innerHTML = '';
    
    const factorLabels = {
      size: 'Planetary Size',
      orbit: 'Orbital Distance',
      atmosphere: 'Atmosphere',
      temperature: 'Surface Temperature',
      water: 'Water Presence'
    };
    
    let delay = 0;
    Object.entries(result.factors).forEach(([key, factor]) => {
      const factorEl = document.createElement('div');
      factorEl.className = 'factor-item';
      factorEl.style.animationDelay = `${delay}ms`;
      
      factorEl.innerHTML = `
        <h4>${factorLabels[key]} <span>${factor.score.toFixed(1)}/25</span></h4>
        <p>${factor.description}</p>
        <div class="factor-progress">
          <div class="factor-progress-fill" style="width: 0%"></div>
        </div>
      `;
      
      factorsContainer.appendChild(factorEl);
      
      // Animate factor progress bars
      setTimeout(() => {
        const fill = factorEl.querySelector('.factor-progress-fill');
        fill.style.width = `${(factor.score / 25) * 100}%`;
      }, delay + 200);
      
      delay += 100;
    });
    
    this.lastResult = result;
  }

  static initExplore() {
    const searchInput = document.getElementById('searchInput');
    const filterBtns = document.querySelectorAll('.filter-btn');
    const planetsGrid = document.getElementById('planetsGrid');
    
    let currentFilter = 'all';
    let searchTerm = '';
    let favorites = new Set(JSON.parse(localStorage.getItem('favorites') || '[]'));
    
    const renderPlanets = () => {
      let filtered = EXOPLANETS;
      
      // Apply search
      if (searchTerm) {
        filtered = filtered.filter(p => 
          p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          p.star.toLowerCase().includes(searchTerm.toLowerCase())
        );
      }
      
      // Apply filter
      if (currentFilter === 'high') {
        filtered = filtered.filter(p => p.habitabilityScore >= 80);
      } else if (currentFilter === 'medium') {
        filtered = filtered.filter(p => p.habitabilityScore >= 60 && p.habitabilityScore < 80);
      } else if (currentFilter === 'low') {
        filtered = filtered.filter(p => p.habitabilityScore < 60);
      }
      
      planetsGrid.innerHTML = '';
      filtered.forEach((planet, index) => {
        const isFavorite = favorites.has(planet.id);
        const card = document.createElement('div');
        card.className = 'planet-card';
        card.style.animationDelay = `${index * 50}ms`;
        card.innerHTML = `
          <h3 class="planet-name">${planet.name}</h3>
          <p class="planet-star">Around ${planet.star}</p>
          
          <div class="planet-stats">
            <div class="planet-stat">
              <span class="planet-stat-label">Radius</span>
              <span class="planet-stat-value">${planet.radius}R⊕</span>
            </div>
            <div class="planet-stat">
              <span class="planet-stat-label">Distance</span>
              <span class="planet-stat-value">${planet.distance}AU</span>
            </div>
          </div>

          <div class="habitability-section">
            <div class="habitability-label">
              <span>Habitability Score</span>
              <span>${planet.habitabilityScore}%</span>
            </div>
            <div class="habitability-bar">
              <div class="habitability-fill" style="width: ${planet.habitabilityScore}%"></div>
            </div>
          </div>

          <div class="planet-attrs">
            ${planet.atmosphere ? '<span class="atmosphere-badge">Has Atmosphere</span>' : ''}
            <button class="favorite-btn ${isFavorite ? 'liked' : ''}" data-planet-id="${planet.id}">
              ${isFavorite ? '❤️' : '🤍'}
            </button>
          </div>
        `;
        
        const favoriteBtn = card.querySelector('.favorite-btn');
        favoriteBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          if (favorites.has(planet.id)) {
            favorites.delete(planet.id);
            UI.showToast(`Removed ${planet.name} from favorites`, 'info');
          } else {
            favorites.add(planet.id);
            UI.showToast(`Added ${planet.name} to favorites`, 'success');
            favoriteBtn.style.animation = 'heartBeat 0.6s ease-out';
          }
          localStorage.setItem('favorites', JSON.stringify([...favorites]));
          renderPlanets();
        });
        
        planetsGrid.appendChild(card);
      });
    };
    
    searchInput.addEventListener('input', (e) => {
      searchTerm = e.target.value;
      renderPlanets();
    });
    
    filterBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        renderPlanets();
      });
    });
    
    renderPlanets();
  }

  static initDashboard() {
    const currentUser = Storage.getCurrentUser();
    if (!currentUser) {
      UI.showPage('signin');
      return;
    }
    
    document.getElementById('welcomeMessage').textContent = `Welcome back, ${currentUser.name}`;
    document.getElementById('userEmail').textContent = currentUser.email;
    
    const predictions = Storage.getPredictionsForUser(currentUser.id);
    const highlyHabitable = predictions.filter(p => p.habitabilityScore >= 80).length;
    const potentiallyHabitable = predictions.filter(p => p.habitabilityScore >= 60 && p.habitabilityScore < 80).length;
    
    // Animate counters
    const totalElement = document.getElementById('totalPredictions');
    const highlyElement = document.getElementById('highlyHabitable');
    const potentialElement = document.getElementById('potentiallyHabitable');
    
    totalElement.textContent = '0';
    highlyElement.textContent = '0';
    potentialElement.textContent = '0';
    
    // Stagger the animation
    setTimeout(() => AnimationHelper.animateCounter(totalElement, predictions.length), 0);
    setTimeout(() => AnimationHelper.animateCounter(highlyElement, highlyHabitable), 100);
    setTimeout(() => AnimationHelper.animateCounter(potentialElement, potentiallyHabitable), 200);
    
    document.getElementById('accountCreated').textContent = new Date(currentUser.createdAt).toLocaleDateString();
    
    const predictionsTable = document.getElementById('predictionsTable');
    if (predictions.length === 0) {
      predictionsTable.innerHTML = '<p style="text-align: center; color: var(--muted-foreground);">No predictions yet. Start analyzing planets!</p>';
      return;
    }
    
    predictionsTable.innerHTML = predictions.map((p, index) => `
      <div class="prediction-item prediction-row" style="animation-delay: ${index * 50}ms;">
        <div class="prediction-info">
          <h3>${p.planetName}</h3>
          <div class="prediction-meta">
            <span>${p.classification}</span>
            <span>Score: ${p.habitabilityScore}%</span>
            <span>${new Date(p.timestamp).toLocaleDateString()}</span>
          </div>
        </div>
        <div class="prediction-actions">
          <button class="delete-btn" data-prediction-id="${p.id}">Delete</button>
        </div>
      </div>
    `).join('');
    
    document.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const id = btn.dataset.predictionId;
        Storage.deletePrediction(id);
        UI.showToast('Prediction deleted', 'info');
        this.initDashboard(); // Refresh
      });
    });
  }
}

// ============================================================
// MAIN APP INITIALIZATION
// ============================================================

class App {
  static init() {
    // Initialize demo account
    Storage.initializeDemoAccount();
    
    // Setup navigation
    UI.updateNavigation();
    
    // Setup page navigation
    document.querySelectorAll('[data-page]').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const page = link.dataset.page;
        
        if (page === 'signup' || page === 'signin') {
          if (Storage.getCurrentUser()) {
            UI.showToast('You are already signed in', 'info');
            return;
          }
        }
        
        if (page === 'predictor' || page === 'dashboard' || page === 'explore') {
          if (!Storage.getCurrentUser()) {
            UI.showToast('You must be signed in first', 'error');
            UI.showPage('signin');
            return;
          }
        }
        
        UI.showPage(page);
        
        if (page === 'predictor') PageHandlers.initPredictor();
        if (page === 'explore') PageHandlers.initExplore();
        if (page === 'dashboard') PageHandlers.initDashboard();
      });
    });
    
    // Setup auth toggle
    document.getElementById('authToggle').addEventListener('click', () => {
      const currentUser = Storage.getCurrentUser();
      if (currentUser) {
        Storage.setCurrentUser(null);
        UI.updateNavigation();
        UI.showToast('Signed out successfully', 'success');
        UI.showPage('home');
      } else {
        UI.showPage('signin');
      }
    });
    
    // Setup theme toggle
    document.getElementById('themeToggle').addEventListener('click', () => {
      document.body.classList.toggle('dark');
      localStorage.setItem('theme', document.body.classList.contains('dark') ? 'dark' : 'light');
    });
    
    // Setup forms
    PageHandlers.initSignup();
    PageHandlers.initSignin();
    
    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'dark') {
      document.body.classList.add('dark');
    }
    
    // Show home page
    UI.showPage('home');
  }
}

// Start app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => App.init());
} else {
  App.init();
}

// ============================================================
// ANIMATION UTILITIES
// ============================================================

class AnimationHelper {
  static animateCounter(element, target, duration = 1500) {
    const startValue = 0;
    const increment = target / (duration / 16);
    let currentValue = startValue;

    const updateCounter = () => {
      currentValue += increment;
      if (currentValue < target) {
        element.textContent = Math.floor(currentValue);
        requestAnimationFrame(updateCounter);
      } else {
        element.textContent = target;
      }
    };

    updateCounter();
  }

  static animateProgressBar(element, percentage, duration = 1000) {
    let currentPercent = 0;
    const increment = percentage / (duration / 16);

    const updateProgress = () => {
      currentPercent += increment;
      if (currentPercent < percentage) {
        element.style.width = currentPercent + '%';
        requestAnimationFrame(updateProgress);
      } else {
        element.style.width = percentage + '%';
      }
    };

    updateProgress();
  }

  static fadeInElement(element, duration = 400) {
    element.style.opacity = '0';
    element.style.transition = `opacity ${duration}ms ease-out`;
    
    setTimeout(() => {
      element.style.opacity = '1';
    }, 10);
  }

  static slideUp(element, duration = 400) {
    element.style.transform = 'translateY(20px)';
    element.style.opacity = '0';
    element.style.transition = `all ${duration}ms ease-out`;
    
    setTimeout(() => {
      element.style.transform = 'translateY(0)';
      element.style.opacity = '1';
    }, 10);
  }

  static scaleIn(element, duration = 300) {
    element.style.transform = 'scale(0.95)';
    element.style.opacity = '0';
    element.style.transition = `all ${duration}ms ease-out`;
    
    setTimeout(() => {
      element.style.transform = 'scale(1)';
      element.style.opacity = '1';
    }, 10);
  }

  static shake(element) {
    element.style.animation = 'none';
    
    setTimeout(() => {
      element.style.animation = 'shake 0.5s ease-in-out';
    }, 10);
  }

  static pulse(element) {
    element.style.animation = 'none';
    
    setTimeout(() => {
      element.style.animation = 'pulse 0.6s ease-out';
    }, 10);
  }

  static addStaggerAnimation(elements, delay = 100) {
    elements.forEach((el, index) => {
      el.style.animation = 'none';
      el.style.opacity = '0';
      el.style.transform = 'translateY(20px)';
      
      setTimeout(() => {
        el.style.transition = 'all 0.5s ease-out';
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
      }, index * delay);
    });
  }
}

// Add shake animation to keyframes dynamically
const style = document.createElement('style');
style.textContent = `
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
  }
`;
document.head.appendChild(style);
