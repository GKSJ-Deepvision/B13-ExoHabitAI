# 🚀 ExoHabitAI — Frontend README

**Exoplanet Habitability Predictor** | ML-powered web app using NASA data

---

## 📁 Files

| File | Size | Purpose |
|------|------|---------|
| `index.html` | ~1230 lines | All pages (single-page scroll) |
| `script.js` | ~1453 lines | All JavaScript logic |
| `style.css` | ~1140 lines | All styles + animations |

---

## 🌐 Pages (Smooth Scroll)

| Section | Anchor | Description |
|---------|--------|-------------|
| Home | `#home` | Hero, ticker, Why ExoHabitAI, stats band |
| About | `#about` | Project info, model details, tech stack |
| Batch | `#batch` | CSV upload + JSON input for bulk analysis |
| Ranking | `#ranking` | Exoplanet leaderboard table |
| Predict | `#predict` | Single planet prediction form |
| Analytics | `#analytics` | Live dashboard with charts + history |

---

## ⚙️ Setup & Run

```bash
# No build step needed — pure HTML/CSS/JS
# Just open index.html in a browser

# For full ML predictions, run Flask backend:
python app.py         # starts at http://localhost:5000

```
**Flask API endpoint used:**
```
POST http://localhost:5000/predict    → single prediction
POST http://localhost:5000/rank       → planet rankings
```

---

## 🤖 ML Model Info

| Metric | Value |
|--------|-------|
| Model | Logistic Regression Pipeline |
| Accuracy | 98.89% |
| Recall | 97.5% |
| F1-Score | 98.73% |
| ROC-AUC | 0.9994 |
| Features | **22** |
| Threshold | 0.5 |

---

## 📊 Key Features

### 1. Predict Page
- Fill form manually **or** use sample buttons:
  - 🌱 **Habitable** — Earth-like planet values
  - 💀 **Non-Habitable** — hot/dead planet values
  - ❄️ **Frozen** — ultra-cold planet (eqt < 150K)
  - 🪄 **Auto Fill** — loads first habitable sample
- Shows instantly on button click:
  - 3D animated planet preview
  - Temperature zone needle
  - Confidence bar
  - Score breakdown bars
- Click **Predict Habitability** → calls Flask API (or local fallback)

### 2. Batch Analysis
- **CSV tab** — upload `.csv` file → shows top 4 results
- **JSON tab** — paste JSON or load sample → shows top 4 results
- Required columns: `pl_rade, pl_bmasse, pl_eqt, st_teff, temp_score, dist_score, lum_score, stellar_temp_score, stellar_mass_score, stellar_compatibility_index, orbital_stability`

### 3. Ranking Table
- 22-planet demo pool (13 habitable + 9 non-habitable)
- Filters: number of results, status (all/habitable/non-habitable), min score
- Score shown as % with 🥇🥈🥉 medals

### 4. Analytics Dashboard
- Auto-updates after every prediction
- Charts: Feature Importance, Score Distribution, Donut, Performance bars
- Export: **PDF Report**, **CSV Data**, **JSON**
- **Reset** button clears all history

---

## 🔑 Important JS Functions

```js
predictLocally(d)          // Local ML fallback — uses weighted scores + penalties
handleSinglePrediction(e)  // Form submit → Flask API → display result
loadSampleData(type)        // Fill form + show preview ('habitable'|'nonHabitable')
loadZoneSample(zone)        // Fill form for 'frozen' zone sample
_showLivePreview(d, type)   // Show planet/temp/confidence before submitting
loadRankingData()           // Fetch rankings (API or local 22-planet pool)
anlRefresh()                // Rebuild all analytics charts from history
anlResetDashboard()         // Clear localStorage prediction history
anlExportPDF()              // Download HTML report
anlExportCSV()              // Download CSV data
anlExportJSON()             // Download JSON export
```

---

## 🎨 CSS Variables (style.css)

```css
--bg        #020d1a    /* deep space background */
--teal      #00d4aa    /* primary accent */
--blue      #1e90ff    /* secondary accent */
--magenta   #e040fb    /* highlight */
--amber     #f59e0b    /* warning / warm */
--green     #10b981    /* habitable / success */
--red       #f43f5e    /* non-habitable / error */
--text      #dff0fa    /* body text */
--muted     #8ec4dc    /* secondary text */
--fh        'Orbitron' /* heading font */
--fb        'Exo 2'    /* body font */
```

---

## 🪐 Sample Data (SAMPLE_DATA in script.js)

**5 Habitable:** Kepler-442b-like, ExoEarth-Alpha, GreenWorld-7, TRAPPIST-1e-like, Kepler-186f-like

**Non-Habitable:** Frozen-Cryo (eqt:85K), HotJupiter-X1 (eqt:1200K), FrozenGiant-Beta, ScorchWorld-Zeta, DeadRock-Omega, IceGiant-Delta

> All non-habitable scores are kept very low (0.02–0.06) so Flask ML model correctly classifies them as ✗ Non-Habitable.

---

## 🐛 Known Behaviour

- **Score display:** `confidence_score` always = P(habitable). Habitable Score card = `conf × 100%`, Non-Habitable Score card = `(1 - conf) × 100%`
- **Preview vs Predict:** Both call the same Flask API → same result. If API is down, both use `predictLocally()` fallback
- **Analytics history** stored in `localStorage` key `exo_anl_v2`

---

## 🗂 Navbar Links

```
Intro | About | Batch | Ranking | Predict | [Dashboard]
```
All links use `href="#section-id"` smooth scroll. Active link highlighted by scroll position.

---

*ExoHabitAI · Logistic Regression ML · NASA Exoplanet Data · 22 Features*
