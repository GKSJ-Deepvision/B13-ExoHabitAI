// ═══════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════
const API_BASE_URL = 'http://127.0.0.1:5000';
let predictionCount = 0;

// ═══════════════════════════════════════════════════
// PLANET DATA — Real colors & textures
// ═══════════════════════════════════════════════════
const GALAXY_PLANETS = [
    {
      name: 'Mercury', score: 15, temp: 440, radius: 0.38,
      size: 18, speed: 0.0006, orbitR: 520, tilt: 0.12, angle: 0.5,
      desc: 'Smallest planet — extreme temperature swings',
      sampleIndex: 23,
      colors: ['#8c7b6b', '#a08878', '#6b5a4e', '#9a8070'],
      type: 'rocky'
    },
    {
      name: 'Venus', score: 12, temp: 737, radius: 0.95,
      size: 32, speed: 0.0004, orbitR: 440, tilt: 0.10, angle: 1.8,
      desc: 'Hottest planet — thick toxic atmosphere',
      sampleIndex: 21,
      colors: ['#e8c84a', '#d4a830', '#f0d060', '#c8a020'],
      type: 'rocky'
    },
    {
      name: 'Earth', score: 95, temp: 255, radius: 1.0,
      size: 38, speed: 0.00035, orbitR: 360, tilt: 0.13, angle: 3.2,
      desc: 'Our home — blue oceans and green continents',
      sampleIndex: 0,
      colors: ['#2a6fd4', '#1a9e3a', '#4a8fe8', '#1a7a2e'],
      type: 'earth'
    },
    {
      name: 'Mars', score: 38, temp: 210, radius: 0.53,
      size: 24, speed: 0.0003, orbitR: 480, tilt: 0.11, angle: 0.9,
      desc: 'The Red Planet — iron oxide rusty surface',
      sampleIndex: 15,
      colors: ['#c1440e', '#e05a1a', '#a03008', '#d04810'],
      type: 'rocky'
    },
    {
      name: 'Jupiter', score: 5, temp: 110, radius: 11.2,
      size: 72, speed: 0.00018, orbitR: 300, tilt: 0.15, angle: 2.1,
      desc: 'Largest planet — Great Red Spot for 350+ years',
      sampleIndex: 22,
      colors: ['#c88b3a', '#e8a84a', '#a06828', '#f0c060'],
      type: 'gas_giant',
      bands: true
    },
    {
      name: 'Saturn', score: 4, temp: 95, radius: 9.5,
      size: 60, speed: 0.00014, orbitR: 560, tilt: 0.09, angle: 4.5,
      desc: 'Ringed giant — rings span 282,000 km',
      sampleIndex: 22,
      colors: ['#e8d080', '#d4b860', '#f0e090', '#c8a840'],
      type: 'saturn',
      bands: true
    },
    {
      name: 'Uranus', score: 3, temp: 76, radius: 4.0,
      size: 42, speed: 0.0001, orbitR: 420, tilt: 0.11, angle: 1.3,
      desc: 'Ice giant — rotates on its side at 98°',
      sampleIndex: 22,
      colors: ['#7de8e8', '#5cd4d4', '#9af0f0', '#4ababa'],
      type: 'ice_giant'
    },
    {
      name: 'Neptune', score: 2, temp: 72, radius: 3.9,
      size: 40, speed: 0.00008, orbitR: 580, tilt: 0.08, angle: 3.7,
      desc: 'Windiest planet — storms up to 2100 km/h',
      sampleIndex: 22,
      colors: ['#2060c8', '#1848a8', '#3878e8', '#1050b0'],
      type: 'ice_giant'
    },
    {
      name: 'Kepler-452b', score: 82, temp: 265, radius: 1.6,
      size: 40, speed: 0.00028, orbitR: 390, tilt: 0.12, angle: 5.1,
      desc: "Earth's cousin — 6 billion year old system",
      sampleIndex: 1,
      colors: ['#3a7fd4', '#2a9e4a', '#5a9fe8', '#2a8e3e'],
      type: 'earth'
    },
    {
      name: 'TRAPPIST-1e', score: 66, temp: 251, radius: 0.91,
      size: 28, speed: 0.0004, orbitR: 330, tilt: 0.14, angle: 2.7,
      desc: 'Best life candidate — red dwarf system',
      sampleIndex: 7,
      colors: ['#4a90d8', '#3a7ac8', '#5aa8e8', '#2a60a8'],
      type: 'earth'
    },
    {
      name: 'WASP-12b', score: 1, temp: 2500, radius: 15.4,
      size: 62, speed: 0.0002, orbitR: 500, tilt: 0.10, angle: 0.2,
      desc: 'Ultra-hot Jupiter — being torn by its star',
      sampleIndex: 25,
      colors: ['#ff4400', '#ff6600', '#dd2200', '#ff8800'],
      type: 'hot_jupiter'
    },
    {
      name: 'Proxima Cen b', score: 58, temp: 234, radius: 1.08,
      size: 30, speed: 0.00032, orbitR: 460, tilt: 0.10, angle: 4.0,
      desc: 'Closest exoplanet — 4.2 light years away',
      sampleIndex: 10,
      colors: ['#50a8d8', '#408898', '#60c8e8', '#307888'],
      type: 'earth'
    },
  ];

// ═══════════════════════════════════════════════════
// DRAW PLANET TEXTURE
// ═══════════════════════════════════════════════════
function drawPlanet(ctx, p, sz) {
  const x = p.x;
  const y = p.y;
  const [c1, c2, c3, c4] = p.colors;

  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, sz, 0, Math.PI * 2);
  ctx.clip();

  if (p.type === 'earth') {
    // Ocean base
    const ocean = ctx.createRadialGradient(x-sz*0.2, y-sz*0.2, 0, x, y, sz);
    ocean.addColorStop(0, lightenColor(c1, 40));
    ocean.addColorStop(0.5, c1);
    ocean.addColorStop(1, darkenColor(c1, 30));
    ctx.fillStyle = ocean;
    ctx.fillRect(x-sz, y-sz, sz*2, sz*2);

    // Continents
    ctx.fillStyle = c2;
    const patches = [
      [x-sz*0.1, y-sz*0.3, sz*0.5, sz*0.35],
      [x+sz*0.2, y+sz*0.1, sz*0.4, sz*0.3],
      [x-sz*0.4, y+sz*0.2, sz*0.3, sz*0.25],
      [x+sz*0.0, y-sz*0.5, sz*0.35, sz*0.2],
    ];
    patches.forEach(([px, py, pw, ph]) => {
      ctx.beginPath();
      ctx.ellipse(px, py, pw*0.5, ph*0.5, Math.random()*0.5, 0, Math.PI*2);
      ctx.fill();
    });

    // Cloud wisps
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    [[-0.3,0.1,0.6,0.12], [0.1,-0.3,0.5,0.1], [-0.1,0.35,0.55,0.1]]
      .forEach(([dx,dy,w,h]) => {
        ctx.beginPath();
        ctx.ellipse(x+dx*sz, y+dy*sz, w*sz*0.5, h*sz*0.5, 0.3, 0, Math.PI*2);
        ctx.fill();
      });

  } else if (p.type === 'gas_giant' || p.type === 'hot_jupiter') {
    // Base
    const base = ctx.createRadialGradient(x-sz*0.3, y-sz*0.3, 0, x, y, sz);
    base.addColorStop(0, lightenColor(c1, 50));
    base.addColorStop(0.6, c1);
    base.addColorStop(1,   darkenColor(c1, 40));
    ctx.fillStyle = base;
    ctx.fillRect(x-sz, y-sz, sz*2, sz*2);

    // Horizontal bands
    const bandColors = [c2, c3, c4, c2, c1, c3];
    const numBands   = 8;
    for (let b = 0; b < numBands; b++) {
      const by     = y - sz + (b / numBands) * sz * 2;
      const bh     = (sz * 2) / numBands;
      const bc     = bandColors[b % bandColors.length];
      ctx.fillStyle = bc + '99';
      ctx.fillRect(x-sz, by, sz*2, bh * 0.7);
    }

    // Great Red Spot for Jupiter
    if (p.name === 'Jupiter') {
      ctx.beginPath();
      ctx.ellipse(x+sz*0.2, y+sz*0.1, sz*0.28, sz*0.16, 0, 0, Math.PI*2);
      ctx.fillStyle = '#cc3300cc';
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(x+sz*0.2, y+sz*0.1, sz*0.18, sz*0.10, 0, 0, Math.PI*2);
      ctx.fillStyle = '#aa2200aa';
      ctx.fill();
    }

    // Hot glow for WASP-12b
    if (p.type === 'hot_jupiter') {
      const hotGrd = ctx.createRadialGradient(x, y, sz*0.3, x, y, sz);
      hotGrd.addColorStop(0, 'rgba(255,200,100,0.6)');
      hotGrd.addColorStop(1, 'rgba(255,50,0,0)');
      ctx.fillStyle = hotGrd;
      ctx.fillRect(x-sz, y-sz, sz*2, sz*2);
    }

  } else if (p.type === 'saturn') {
    const base = ctx.createRadialGradient(x-sz*0.3, y-sz*0.3, 0, x, y, sz);
    base.addColorStop(0, lightenColor(c1, 50));
    base.addColorStop(0.6, c1);
    base.addColorStop(1,   darkenColor(c1, 35));
    ctx.fillStyle = base;
    ctx.fillRect(x-sz, y-sz, sz*2, sz*2);

    // Bands
    for (let b = 0; b < 7; b++) {
      const by = y - sz + (b/7)*sz*2;
      ctx.fillStyle = b%2===0 ? c2+'88' : c4+'66';
      ctx.fillRect(x-sz, by, sz*2, sz*2/7*0.6);
    }

  } else if (p.type === 'ice_giant') {
    const base = ctx.createRadialGradient(x-sz*0.25, y-sz*0.25, 0, x, y, sz);
    base.addColorStop(0, lightenColor(c1, 60));
    base.addColorStop(0.5, c1);
    base.addColorStop(1,   darkenColor(c1, 30));
    ctx.fillStyle = base;
    ctx.fillRect(x-sz, y-sz, sz*2, sz*2);

    // Swirls
    ctx.strokeStyle = lightenColor(c2, 40) + '88';
    ctx.lineWidth   = sz * 0.08;
    for (let s = 0; s < 3; s++) {
      ctx.beginPath();
      ctx.ellipse(x, y - sz*0.2 + s*sz*0.3, sz*0.8, sz*0.08, 0, 0, Math.PI*2);
      ctx.stroke();
    }

  } else {
    // Rocky planet
    const base = ctx.createRadialGradient(x-sz*0.3, y-sz*0.3, 0, x, y, sz);
    base.addColorStop(0, lightenColor(c1, 50));
    base.addColorStop(0.5, c1);
    base.addColorStop(1,   darkenColor(c1, 40));
    ctx.fillStyle = base;
    ctx.fillRect(x-sz, y-sz, sz*2, sz*2);

    // Rocky surface patches
    ctx.fillStyle = darkenColor(c1, 20) + 'aa';
    [[0.2,0.1,0.25], [-0.2,-0.15,0.2],
     [0.0,0.3,0.18], [-0.1,0.0,0.22]]
      .forEach(([dx, dy, r]) => {
        ctx.beginPath();
        ctx.arc(x+dx*sz, y+dy*sz, r*sz, 0, Math.PI*2);
        ctx.fill();
      });

    // Craters for Mercury
    if (p.name === 'Mercury') {
      ctx.strokeStyle = darkenColor(c1, 50) + 'cc';
      ctx.lineWidth   = 1;
      [[0.15, -0.2, 0.12], [-0.25, 0.1, 0.09],
       [0.0, 0.25, 0.07],  [0.3, 0.15, 0.06]]
        .forEach(([dx, dy, r]) => {
          ctx.beginPath();
          ctx.arc(x+dx*sz, y+dy*sz, r*sz, 0, Math.PI*2);
          ctx.stroke();
        });
    }
  }

  // Lighting overlay — darker on right side
  const light = ctx.createRadialGradient(x-sz*0.4, y-sz*0.4, 0, x+sz*0.3, y+sz*0.3, sz*1.4);
  light.addColorStop(0,   'rgba(255,255,255,0.15)');
  light.addColorStop(0.5, 'rgba(0,0,0,0)');
  light.addColorStop(1,   'rgba(0,0,0,0.55)');
  ctx.fillStyle = light;
  ctx.fillRect(x-sz, y-sz, sz*2, sz*2);

  ctx.restore();

  // Atmosphere rim
  const atmo = ctx.createRadialGradient(x, y, sz*0.78, x, y, sz*1.12);
  const atmoColor = p.type === 'earth' ? '60,140,255'
                  : p.type === 'hot_jupiter' ? '255,80,0'
                  : p.type === 'ice_giant'   ? '80,200,220'
                  : '100,120,160';
  atmo.addColorStop(0, `rgba(${atmoColor},0)`);
  atmo.addColorStop(0.6, `rgba(${atmoColor},0.2)`);
  atmo.addColorStop(1,   `rgba(${atmoColor},0)`);
  ctx.beginPath();
  ctx.arc(x, y, sz*1.12, 0, Math.PI*2);
  ctx.fillStyle = atmo;
  ctx.fill();
}

// ── Color helpers ─────────────────────────────────
function lightenColor(hex, amt) {
  const n = parseInt(hex.replace('#',''), 16);
  const r = Math.min(255, (n>>16)+amt);
  const g = Math.min(255, ((n>>8)&0xff)+amt);
  const b = Math.min(255, (n&0xff)+amt);
  return `rgb(${r},${g},${b})`;
}
function darkenColor(hex, amt) {
  const n = parseInt(hex.replace('#',''), 16);
  const r = Math.max(0, (n>>16)-amt);
  const g = Math.max(0, ((n>>8)&0xff)-amt);
  const b = Math.max(0, (n&0xff)-amt);
  return `rgb(${r},${g},${b})`;
}

// ═══════════════════════════════════════════════════
// FULLSCREEN GALAXY
// ═══════════════════════════════════════════════════
function initGalaxy() {
  const canvas  = document.getElementById('galaxyCanvas');
  if (!canvas) return;
  const ctx     = canvas.getContext('2d');
  const tooltip = document.getElementById('planetTooltip');
  let hoveredIdx = -1;

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    placePlanets();
  }

  function placePlanets() {
    const cx = canvas.width  / 2;
    const cy = canvas.height / 2;
    GALAXY_PLANETS.forEach(p => {
      p.cx = cx;
      p.cy = cy;
      // tilt low — planets will horizontally spread 
      const effectiveTilt = Math.min(p.tilt, canvas.height / (2 * p.orbitR) * 0.85);
      p._tilt = effectiveTilt;
      p.x  = cx + Math.cos(p.angle) * p.orbitR;
      p.y  = cy + Math.sin(p.angle) * p.orbitR * effectiveTilt;
    });
  }

  // Stars
  const STARS = Array.from({ length: 400 }, () => ({
    x: Math.random(), y: Math.random(),
    r: Math.random() * 1.6 + 0.2,
    o: Math.random() * 0.7 + 0.1,
    t: Math.random() * Math.PI * 2,
    speed: Math.random() * 0.02 + 0.005
  }));

  // Nebulas
  const NEBULAS = [
    { x:0.12, y:0.18, r:260, c:'rgba(74,144,226,0.08)'  },
    { x:0.85, y:0.78, r:220, c:'rgba(74,222,128,0.06)'  },
    { x:0.78, y:0.12, r:190, c:'rgba(139,92,246,0.07)'  },
    { x:0.18, y:0.82, r:200, c:'rgba(245,158,11,0.06)'  },
    { x:0.55, y:0.45, r:160, c:'rgba(255,100,100,0.04)' },
    { x:0.38, y:0.65, r:140, c:'rgba(100,200,255,0.05)' },
  ];

  function draw() {
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Deep space bg
    const bg = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, W*0.8);
    bg.addColorStop(0,   '#081828');
    bg.addColorStop(0.4, '#050e18');
    bg.addColorStop(1,   '#020608');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Nebulas
    NEBULAS.forEach(n => {
      const g = ctx.createRadialGradient(n.x*W, n.y*H, 0, n.x*W, n.y*H, n.r);
      g.addColorStop(0, n.c);
      g.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.arc(n.x*W, n.y*H, n.r, 0, Math.PI*2);
      ctx.fillStyle = g;
      ctx.fill();
    });

    // Stars twinkle
    STARS.forEach(s => {
      s.t += s.speed;
      const op = s.o * (0.5 + 0.5 * Math.sin(s.t));
      // Sparkle effect for bright stars
      if (s.r > 1.2) {
        ctx.beginPath();
        ctx.moveTo(s.x*W, s.y*H - s.r*3);
        ctx.lineTo(s.x*W, s.y*H + s.r*3);
        ctx.moveTo(s.x*W - s.r*3, s.y*H);
        ctx.lineTo(s.x*W + s.r*3, s.y*H);
        ctx.strokeStyle = `rgba(255,255,255,${op*0.4})`;
        ctx.lineWidth   = 0.5;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(s.x*W, s.y*H, s.r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(255,255,255,${op})`;
      ctx.fill();
    });

    const cx = W / 2;
    const cy = H / 2;

    // Galaxy core
    const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, 140);
    core.addColorStop(0,    'rgba(255,248,220,0.98)');
    core.addColorStop(0.12, 'rgba(200,220,255,0.7)');
    core.addColorStop(0.4,  'rgba(100,160,255,0.25)');
    core.addColorStop(0.8,  'rgba(74,144,226,0.08)');
    core.addColorStop(1,    'rgba(74,144,226,0)');
    ctx.beginPath();
    ctx.arc(cx, cy, 140, 0, Math.PI*2);
    ctx.fillStyle = core;
    ctx.fill();

    // Orbit rings
    GALAXY_PLANETS.forEach(p => {
      ctx.beginPath();
      ctx.ellipse(cx, cy, p.orbitR, p.orbitR * (p._tilt || p.tilt), 0, 0, Math.PI*2);
      ctx.strokeStyle = 'rgba(74,144,226,0.1)';
      ctx.lineWidth   = 0.7;
      ctx.setLineDash([5, 12]);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // Update + draw planets
    GALAXY_PLANETS.forEach((p, i) => {
      p.angle += p.speed;
      p.angle += p.speed;
      p.x = cx + Math.cos(p.angle) * p.orbitR;
      p.y = cy + Math.sin(p.angle) * p.orbitR * (p._tilt || p.tilt);

      const isH = hoveredIdx === i;
      const sz  = p.size * (isH ? 1.35 : 1);

      // Drop shadow
      ctx.beginPath();
      ctx.ellipse(p.x, p.y + sz*0.95, sz*0.7, sz*0.15, 0, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fill();

      // Hover outer glow
      if (isH) {
        const glow = ctx.createRadialGradient(p.x, p.y, sz, p.x, p.y, sz*4);
        glow.addColorStop(0, 'rgba(74,144,226,0.5)');
        glow.addColorStop(1, 'rgba(74,144,226,0)');
        ctx.beginPath();
        ctx.arc(p.x, p.y, sz*4, 0, Math.PI*2);
        ctx.fillStyle = glow;
        ctx.fill();
      }

      // Draw textured planet
      drawPlanet(ctx, { ...p, x: p.x, y: p.y }, sz);

      // Saturn rings (drawn after planet)
      if (p.name === 'Saturn') {
        // Back ring
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, sz*2.1, sz*0.38, 0, Math.PI, Math.PI*2);
        ctx.strokeStyle = 'rgba(210,185,110,0.6)';
        ctx.lineWidth   = sz * 0.22;
        ctx.stroke();
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, sz*1.65, sz*0.28, 0, Math.PI, Math.PI*2);
        ctx.strokeStyle = 'rgba(210,185,110,0.35)';
        ctx.lineWidth   = sz * 0.12;
        ctx.stroke();
        // Front ring
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, sz*2.1, sz*0.38, 0, 0, Math.PI);
        ctx.strokeStyle = 'rgba(210,185,110,0.75)';
        ctx.lineWidth   = sz * 0.22;
        ctx.stroke();
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, sz*1.65, sz*0.28, 0, 0, Math.PI);
        ctx.strokeStyle = 'rgba(210,185,110,0.4)';
        ctx.lineWidth   = sz * 0.12;
        ctx.stroke();
      }

      // Hover ring + label
      if (isH) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, sz + 6, 0, Math.PI*2);
        ctx.strokeStyle = 'rgba(100,180,255,0.9)';
        ctx.lineWidth   = 2;
        ctx.stroke();

        ctx.font        = `bold ${Math.max(12,sz*0.35)}px Inter,sans-serif`;
        ctx.fillStyle   = 'white';
        ctx.textAlign   = 'center';
        ctx.shadowColor = '#4a90e2';
        ctx.shadowBlur  = 16;
        ctx.fillText(p.name, p.x, p.y - sz - 16);
        ctx.shadowBlur  = 0;
      }
    });

    requestAnimationFrame(draw);
  }

  resize();
  draw();
  window.addEventListener('resize', resize);

  // ── Mouse events ──────────────────────────────────
  canvas.addEventListener('mousemove', (e) => {
    let found = -1;
    GALAXY_PLANETS.forEach((p, i) => {
      const dx = e.clientX - p.x;
      const dy = e.clientY - p.y;
      if (Math.sqrt(dx*dx + dy*dy) < p.size * 2.5) found = i;
    });

    hoveredIdx = found;

    if (found >= 0) {
      const p      = GALAXY_PLANETS[found];
      const sc     = p.score >= 70 ? '#4ade80' : p.score >= 40 ? '#f59e0b' : '#f87171';
      const stTxt  = p.score >= 70 ? '✅ Potentially Habitable'
                   : p.score >= 40 ? '⚠️ Borderline'
                   : '❌ Not Habitable';
      const stBg   = p.score >= 70 ? 'rgba(74,222,128,0.15)'
                   : p.score >= 40 ? 'rgba(245,158,11,0.15)'
                   : 'rgba(248,113,113,0.15)';
      const icons  = {
        'Earth':'🌍','Mars':'🔴','Jupiter':'🟠','Saturn':'🪐',
        'Venus':'🟡','Mercury':'⚫','Neptune':'🔵','Uranus':'🩵',
        'Kepler-452b':'🌍','TRAPPIST-1e':'🟢',
        'WASP-12b':'🔴','Proxima Cen b':'🌍'
      };

      document.getElementById('ttIcon').textContent    = icons[p.name] || '🪐';
      document.getElementById('ttName').textContent    = p.name;
      document.getElementById('ttScore').textContent   = p.score + '% Habitability';
      document.getElementById('ttScore').style.color   = sc;
      document.getElementById('ttStats').innerHTML     =
        `Temp: ${p.temp} K &nbsp;|&nbsp; Radius: ${p.radius} R⊕<br/><em>${p.desc}</em>`;
      document.getElementById('ttStatus').textContent  = stTxt;
      document.getElementById('ttStatus').style.background = stBg;
      document.getElementById('ttStatus').style.color      = sc;
      tooltip._idx = found;

      const tw = 240;
      const tx = e.clientX + 24 + tw > window.innerWidth
               ? e.clientX - tw - 12 : e.clientX + 24;
      const ty = Math.min(e.clientY - 10, window.innerHeight - 250);
      tooltip.style.left = tx + 'px';
      tooltip.style.top  = ty + 'px';
      tooltip.classList.add('show');
      canvas.style.cursor = 'pointer';
    } else {
      tooltip.classList.remove('show');
      canvas.style.cursor = 'crosshair';
    }
  });

  canvas.addEventListener('mouseleave', () => {
    hoveredIdx = -1;
    tooltip.classList.remove('show');
  });

  canvas.addEventListener('click', () => {
    if (hoveredIdx >= 0) {
      enterSite(GALAXY_PLANETS[hoveredIdx].sampleIndex);
    }
  });
}

// ═══════════════════════════════════════════════════
// ENTER SITE
// ═══════════════════════════════════════════════════
function enterSite(sampleIdx) {
  const intro = document.getElementById('galaxyIntro');
  const site  = document.getElementById('mainSite');
  intro.style.transition = 'opacity 0.8s ease';
  intro.style.opacity    = '0';
  setTimeout(() => {
    intro.style.display  = 'none';
    site.classList.add('visible');
    document.body.style.overflow = 'auto';
    AOS.init({ duration: 800, once: true, offset: 80 });
    checkAPIHealth();
    initMiniStars();
    initNavScroll();
    if (sampleIdx !== undefined) {
      setTimeout(() => {
        loadSample(sampleIdx);
        document.getElementById('predictor')
          ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 500);
    }
  }, 800);
}

// ═══════════════════════════════════════════════════
// MINI STARS
// ═══════════════════════════════════════════════════
function initMiniStars() {
  const canvas = document.getElementById('miniStars');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  function resize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);
  const stars = Array.from({ length: 120 }, () => ({
    x: Math.random(), y: Math.random(),
    r: Math.random() * 1.2 + 0.2,
    o: Math.random() * 0.4 + 0.1,
    t: Math.random() * Math.PI * 2
  }));
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stars.forEach(s => {
      s.t += 0.015;
      ctx.beginPath();
      ctx.arc(s.x*canvas.width, s.y*canvas.height, s.r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(255,255,255,${s.o*(0.6+0.4*Math.sin(s.t))})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
}

// ═══════════════════════════════════════════════════
// NAV SCROLL
// ═══════════════════════════════════════════════════
function initNavScroll() {
  window.addEventListener('scroll', () => {
    document.getElementById('mainNav')
      ?.classList.toggle('scrolled', window.scrollY > 50);
  });
}

// ═══════════════════════════════════════════════════
// API HEALTH
// ═══════════════════════════════════════════════════
async function checkAPIHealth() {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  if (!dot || !text) return;
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method:'GET', mode:'cors', cache:'no-cache'
    });
    if (res.ok) {
      dot.classList.add('online');
      text.textContent = 'API: Online';
    } else throw new Error();
  } catch {
    dot.classList.add('offline');
    text.textContent = 'API: Offline';
  }
}

// ═══════════════════════════════════════════════════
// SMOOTH SCROLL
// ═══════════════════════════════════════════════════
function smoothScroll(id) {
  document.getElementById(id)
    ?.scrollIntoView({ behavior:'smooth', block:'start' });
}

// ═══════════════════════════════════════════════════
// SAMPLE PLANETS
// ═══════════════════════════════════════════════════
const SAMPLE_PLANETS = [
  { planet_name:'Earth',             planet_radius:1.0,  orbital_period:365.25, equilibrium_temperature:255, semi_major_axis:1.0,    stellar_luminosity:1.0,      stellar_mass:1.0    },
  { planet_name:'Kepler-452b',       planet_radius:1.6,  orbital_period:384.8,  equilibrium_temperature:265, semi_major_axis:1.05,   stellar_luminosity:1.2,      stellar_mass:1.04   },
  { planet_name:'Kepler-442b',       planet_radius:1.34, orbital_period:112.3,  equilibrium_temperature:233, semi_major_axis:0.409,  stellar_luminosity:0.112,    stellar_mass:0.61   },
  { planet_name:'Kepler-62f',        planet_radius:1.41, orbital_period:267.3,  equilibrium_temperature:208, semi_major_axis:0.718,  stellar_luminosity:0.25,     stellar_mass:0.69   },
  { planet_name:'Kepler-22b',        planet_radius:2.4,  orbital_period:289.9,  equilibrium_temperature:262, semi_major_axis:0.849,  stellar_luminosity:0.79,     stellar_mass:0.97   },
  { planet_name:'Kepler-296e',       planet_radius:1.48, orbital_period:34.1,   equilibrium_temperature:243, semi_major_axis:0.165,  stellar_luminosity:0.02,     stellar_mass:0.497  },
  { planet_name:'Kepler-186f',       planet_radius:1.17, orbital_period:129.9,  equilibrium_temperature:188, semi_major_axis:0.432,  stellar_luminosity:0.04,     stellar_mass:0.544  },
  { planet_name:'TRAPPIST-1e',       planet_radius:0.91, orbital_period:6.1,    equilibrium_temperature:251, semi_major_axis:0.0293, stellar_luminosity:0.000553, stellar_mass:0.0898 },
  { planet_name:'TRAPPIST-1f',       planet_radius:1.04, orbital_period:9.2,    equilibrium_temperature:219, semi_major_axis:0.0385, stellar_luminosity:0.000553, stellar_mass:0.0898 },
  { planet_name:'TRAPPIST-1g',       planet_radius:1.13, orbital_period:12.4,   equilibrium_temperature:198, semi_major_axis:0.0469, stellar_luminosity:0.000553, stellar_mass:0.0898 },
  { planet_name:'Proxima Centauri b',planet_radius:1.08, orbital_period:11.2,   equilibrium_temperature:234, semi_major_axis:0.0485, stellar_luminosity:0.0017,   stellar_mass:0.1221 },
  { planet_name:'GJ 667Cc',          planet_radius:1.54, orbital_period:28.1,   equilibrium_temperature:277, semi_major_axis:0.125,  stellar_luminosity:0.013,    stellar_mass:0.33   },
  { planet_name:'HD 40307g',         planet_radius:1.89, orbital_period:197.8,  equilibrium_temperature:226, semi_major_axis:0.6,    stellar_luminosity:0.23,     stellar_mass:0.77   },
  { planet_name:'Tau Ceti e',        planet_radius:1.65, orbital_period:168.1,  equilibrium_temperature:271, semi_major_axis:0.538,  stellar_luminosity:0.52,     stellar_mass:0.783  },
  { planet_name:'Kepler-1229b',      planet_radius:1.4,  orbital_period:86.8,   equilibrium_temperature:213, semi_major_axis:0.298,  stellar_luminosity:0.056,    stellar_mass:0.54   },
  { planet_name:'Mars',              planet_radius:0.53, orbital_period:687,    equilibrium_temperature:210, semi_major_axis:1.52,   stellar_luminosity:1.0,      stellar_mass:1.0    },
  { planet_name:'Super-Earth K2-18b',planet_radius:2.27, orbital_period:32.9,   equilibrium_temperature:265, semi_major_axis:0.1429, stellar_luminosity:0.035,    stellar_mass:0.3593 },
  { planet_name:'Kepler-62e',        planet_radius:1.61, orbital_period:122.4,  equilibrium_temperature:270, semi_major_axis:0.427,  stellar_luminosity:0.25,     stellar_mass:0.69   },
  { planet_name:'Wolf 1061c',        planet_radius:1.66, orbital_period:17.9,   equilibrium_temperature:228, semi_major_axis:0.089,  stellar_luminosity:0.01,     stellar_mass:0.294  },
  { planet_name:'Gliese 163c',       planet_radius:1.8,  orbital_period:25.6,   equilibrium_temperature:277, semi_major_axis:0.125,  stellar_luminosity:0.022,    stellar_mass:0.4    },
  { planet_name:'Hot-Jupiter-X',     planet_radius:11.2, orbital_period:3.5,    equilibrium_temperature:800, semi_major_axis:0.05,   stellar_luminosity:2.5,      stellar_mass:1.1    },
  { planet_name:'Venus-like',        planet_radius:0.95, orbital_period:225,    equilibrium_temperature:737, semi_major_axis:0.72,   stellar_luminosity:1.0,      stellar_mass:1.0    },
  { planet_name:'Jupiter',           planet_radius:11.2, orbital_period:4333,   equilibrium_temperature:110, semi_major_axis:5.2,    stellar_luminosity:1.0,      stellar_mass:1.0    },
  { planet_name:'Mercury',           planet_radius:0.38, orbital_period:88,     equilibrium_temperature:440, semi_major_axis:0.39,   stellar_luminosity:1.0,      stellar_mass:1.0    },
  { planet_name:'55 Cancri e',       planet_radius:1.88, orbital_period:0.74,   equilibrium_temperature:2400,semi_major_axis:0.0154, stellar_luminosity:0.582,    stellar_mass:0.905  },
  { planet_name:'WASP-12b',          planet_radius:15.4, orbital_period:1.09,   equilibrium_temperature:2500,semi_major_axis:0.0229, stellar_luminosity:1.657,    stellar_mass:1.35   },
  { planet_name:'HD 189733b',        planet_radius:12.7, orbital_period:2.2,    equilibrium_temperature:1200,semi_major_axis:0.031,  stellar_luminosity:0.36,     stellar_mass:0.846  },
  { planet_name:'Kepler-7b',         planet_radius:14.6, orbital_period:4.9,    equilibrium_temperature:1540,semi_major_axis:0.062,  stellar_luminosity:3.9,      stellar_mass:1.36   },
];

// ═══════════════════════════════════════════════════
// LOAD SAMPLE
// ═══════════════════════════════════════════════════
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
  setTimeout(() => {
    document.getElementById('predictor')
      ?.scrollIntoView({ behavior:'smooth', block:'start' });
    const btn = document.getElementById('submitBtn');
    if (!btn) return;
    btn.style.boxShadow  = '0 0 35px rgba(26,106,255,0.8)';
    btn.style.background = 'linear-gradient(135deg,#00d4ff,#0040cc)';
    setTimeout(() => {
      btn.style.boxShadow  = '';
      btn.style.background = '';
    }, 3000);
  }, 200);
}

// ═══════════════════════════════════════════════════
// FORM SUBMIT
// ═══════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictForm');
  if (!form) return;
  form.addEventListener('submit', async function(e) {
    e.preventDefault();
    const fields = ['planet_radius','orbital_period',
      'equilibrium_temperature','semi_major_axis',
      'stellar_luminosity','stellar_mass'];
    let valid = true;
    fields.forEach(f => {
      const el = document.getElementById(f);
      if (!el||!el.value||isNaN(el.value)) {
        el?.classList.add('is-invalid'); valid = false;
      } else el?.classList.remove('is-invalid');
    });
    if (!valid) {
      showState('error');
      document.getElementById('errorMessage').textContent =
        'Please fill all required fields with valid numbers.';
      return;
    }
    const inputData = {};
    ['planet_name','planet_radius','orbital_period',
     'equilibrium_temperature','semi_major_axis',
     'stellar_luminosity','stellar_mass'].forEach(f => {
      const el = document.getElementById(f);
      inputData[f] = el ? (el.value||'Unknown') : 'Unknown';
    });
    showState('loading');
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(inputData)
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error||'Backend error');
      displayResults(result, inputData);
      addToHistory(inputData, result);
      showState('results');
    } catch(err) {
      showState('error');
      document.getElementById('errorMessage').textContent =
        err.message.includes('fetch')
          ? '❌ Cannot connect to backend. Make sure Flask is running.'
          : err.message;
    }
  });
});

// ═══════════════════════════════════════════════════
// DISPLAY RESULTS
// ═══════════════════════════════════════════════════
function displayResults(result, inputData) {
  const isH   = result.prediction === 1;
  const score = result.habitability_score;
  document.getElementById('resultIcon').textContent = isH ? '🌍' : '🔴';
  const st = document.getElementById('resultStatus');
  st.textContent = result.habitability_status;
  st.style.color = isH ? 'var(--green)' : 'var(--red)';
  const badge = document.getElementById('confidenceBadge');
  badge.textContent      = result.confidence;
  badge.style.background = isH ? 'rgba(74,222,128,0.15)' : 'rgba(248,113,113,0.15)';
  badge.style.color      = isH ? 'var(--green)' : 'var(--red)';
  setTimeout(() => {
    const bar = document.getElementById('scoreBar');
    if (!bar) return;
    bar.style.width      = score+'%';
    bar.style.background = isH
      ? 'linear-gradient(90deg,#4ade80,#22c55e)'
      : 'linear-gradient(90deg,#f87171,#ef4444)';
  }, 100);
  const sb = document.getElementById('scoreBig');
  if (sb) { sb.textContent = score+'%'; sb.style.color = isH?'var(--green)':'var(--red)'; }
  const sl = document.getElementById('scoreLabel');
  if (sl) sl.textContent = score+'%';
  const tb = document.getElementById('resultTableBody');
  if (tb) tb.innerHTML = `
    <tr><td><strong>Planet Radius</strong></td><td>${inputData.planet_radius} R⊕</td></tr>
    <tr><td><strong>Orbital Period</strong></td><td>${inputData.orbital_period} days</td></tr>
    <tr><td><strong>Temperature</strong></td><td>${inputData.equilibrium_temperature} K</td></tr>
    <tr><td><strong>Semi-Major Axis</strong></td><td>${inputData.semi_major_axis} AU</td></tr>
    <tr><td><strong>Stellar Luminosity</strong></td><td>${inputData.stellar_luminosity} L☉</td></tr>
    <tr><td><strong>Stellar Mass</strong></td><td>${inputData.stellar_mass} M☉</td></tr>`;
  const exp = document.getElementById('explanationText');
  if (exp) exp.textContent = isH
    ? '✅ This planet falls within parameters consistent with liquid water and atmospheric stability. Conditions are favorable for life.'
    : '❌ This planet\'s parameters fall outside the conventional habitable zone. Conditions make life as we know it unlikely.';
}

// ═══════════════════════════════════════════════════
// HISTORY
// ═══════════════════════════════════════════════════
function addToHistory(inputData, result) {
  predictionCount++;
  const isH  = result.prediction === 1;
  const time = new Date().toTimeString().slice(0,8);
  const nr   = document.getElementById('noHistoryRow');
  if (nr) nr.style.display = 'none';
  const tbody = document.getElementById('historyBody');
  if (!tbody) return;
  const rows = tbody.querySelectorAll('tr:not(#noHistoryRow)');
  if (rows.length >= 10) rows[rows.length-1].remove();
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${predictionCount}</td>
    <td>${inputData.planet_name}</td>
    <td>${inputData.planet_radius}</td>
    <td>${inputData.equilibrium_temperature}</td>
    <td>${inputData.orbital_period}</td>
    <td><strong>${result.habitability_score}%</strong></td>
    <td><span class="badge" style="background:${isH
      ?'rgba(74,222,128,0.15)':'rgba(248,113,113,0.15)'};
      color:${isH?'var(--green)':'var(--red)'}">
      ${isH?'✅ Habitable':'❌ Not Habitable'}</span></td>
    <td><small>${time}</small></td>`;
  tbody.prepend(row);
}

// ═══════════════════════════════════════════════════
// STATE MANAGER
// ═══════════════════════════════════════════════════
function showState(state) {
  ['placeholderState','loadingState','errorState','resultsState']
    .forEach(id => document.getElementById(id)?.classList.add('d-none'));
  const btn = document.getElementById('submitBtn');
  if (state === 'loading') {
    document.getElementById('loadingState')?.classList.remove('d-none');
    if (btn) { btn.disabled=true; btn.innerHTML='<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...'; }
  } else {
    if (btn) { btn.disabled=false; btn.innerHTML='<i class="fas fa-rocket me-2"></i>Predict Habitability'; }
    document.getElementById(state+'State')?.classList.remove('d-none');
  }
}
function resetToPlaceholder() { showState('placeholder'); }
function resetForm() {
  document.getElementById('predictForm')?.reset();
  ['planet_radius','orbital_period','equilibrium_temperature',
   'semi_major_axis','stellar_luminosity','stellar_mass']
    .forEach(id => document.getElementById(id)?.classList.remove('is-invalid'));
  showState('placeholder');
  const bar = document.getElementById('scoreBar');
  if (bar) bar.style.width = '0%';
}
function clearHistory() {
  const tbody = document.getElementById('historyBody');
  if (tbody) tbody.innerHTML = `<tr id="noHistoryRow">
    <td colspan="8" class="text-center py-4" style="color:var(--muted)">No predictions yet</td></tr>`;
  predictionCount = 0;
}

// ═══════════════════════════════════════════════════
// START
// ═══════════════════════════════════════════════════
document.body.style.overflow = 'hidden';
initGalaxy();