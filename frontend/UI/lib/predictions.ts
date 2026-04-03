// Habitability prediction engine - simulates ML analysis of exoplanet parameters
export interface HabitabilityParameters {
  radius: number; // In Earth radii (0.5-2.5)
  distance: number; // In AU from star (0.5-2.0)
  atmosphere: boolean;
  temperature: number; // In Kelvin (200-400)
  waterPresence: number; // Probability 0-100
}

export interface HabitabilityResult {
  score: number; // 0-100
  classification: string;
  factors: {
    size: { score: number; description: string };
    orbit: { score: number; description: string };
    atmosphere: { score: number; description: string };
    temperature: { score: number; description: string };
    water: { score: number; description: string };
  };
  recommendation: string;
}

export function calculateHabitability(params: HabitabilityParameters): HabitabilityResult {
  // Size factor: Optimal around 0.8-1.2 Earth radii
  const sizeScore = calculateSizeFactor(params.radius);
  
  // Orbital distance factor: Habitable zone roughly 0.8-1.5 AU
  const orbitScore = calculateOrbitFactor(params.distance);
  
  // Atmosphere factor: Essential for habitability
  const atmosphereScore = params.atmosphere ? 25 : 0;
  
  // Temperature factor: Ideal 250-310K (-23°C to 37°C)
  const temperatureScore = calculateTemperatureFactor(params.temperature);
  
  // Water presence factor
  const waterScore = Math.min(params.waterPresence, 100) * 0.25;
  
  // Calculate weighted total (normalized to 0-100)
  const totalScore = (sizeScore + orbitScore + atmosphereScore + temperatureScore + waterScore) / 1.25;
  const finalScore = Math.min(Math.round(totalScore), 100);
  
  // Classification
  const classification = getClassification(finalScore);
  
  // Generate detailed factors report
  const factors = {
    size: {
      score: sizeScore,
      description: getSizeDescription(params.radius)
    },
    orbit: {
      score: orbitScore,
      description: getOrbitDescription(params.distance)
    },
    atmosphere: {
      score: atmosphereScore,
      description: params.atmosphere ? 'Atmosphere detected' : 'No atmosphere detected'
    },
    temperature: {
      score: temperatureScore,
      description: getTemperatureDescription(params.temperature)
    },
    water: {
      score: waterScore,
      description: `Water presence probability: ${params.waterPresence.toFixed(1)}%`
    }
  };
  
  // Generate recommendation
  const recommendation = generateRecommendation(finalScore, classification, params);
  
  return {
    score: finalScore,
    classification,
    factors,
    recommendation
  };
}

function calculateSizeFactor(radius: number): number {
  // Optimal size is around 0.8-1.2 Earth radii (Super-Earths to Earth-like)
  const optimal = 1.0;
  const deviation = Math.abs(radius - optimal);
  
  if (deviation < 0.2) {
    return 25; // Near optimal
  } else if (deviation < 0.5) {
    return 20; // Good
  } else if (deviation < 1.0) {
    return 12; // Fair
  }
  return 5; // Poor
}

function calculateOrbitFactor(distance: number): number {
  // Habitable zone typically 0.8-1.5 AU
  if (distance >= 0.95 && distance <= 1.37) {
    return 25;
  } else if (distance >= 0.8 && distance <= 1.5) {
    return 20;
  } else if (distance >= 0.7 && distance <= 1.7) {
    return 12;
  }
  return 5;
}

function calculateTemperatureFactor(temp: number): number {
  // Liquid water range: 273K-373K (-0°C to 100°C)
  // Optimal for life: 250K-310K (-23°C to 37°C)
  const optimalMin = 250;
  const optimalMax = 310;
  const acceptableMin = 273;
  const acceptableMax = 373;
  
  if (temp >= optimalMin && temp <= optimalMax) {
    return 25;
  } else if (temp >= acceptableMin && temp <= acceptableMax) {
    return 20;
  } else if (temp >= 240 && temp <= 390) {
    return 12;
  }
  return 5;
}

function getSizeFactor(radius: number): number {
  return calculateSizeFactor(radius);
}

function getSizeDescription(radius: number): string {
  if (radius < 0.5) return 'Too small for significant atmosphere';
  if (radius < 0.8) return 'Small but potentially habitable';
  if (radius <= 1.2) return 'Optimal size range';
  if (radius < 2.0) return 'Larger Super-Earth, possibly habitable';
  return 'Too large, likely a mini-Neptune';
}

function getOrbitDescription(distance: number): string {
  if (distance < 0.8) return 'Too close to star - likely too hot';
  if (distance >= 0.95 && distance <= 1.37) return 'Optimal habitable zone';
  if (distance <= 1.5) return 'Within habitable zone';
  if (distance <= 2.0) return 'Edge of habitable zone';
  return 'Too far from star - likely too cold';
}

function getTemperatureDescription(temp: number): string {
  if (temp < 200) return 'Extremely cold';
  if (temp < 250) return 'Very cold';
  if (temp >= 250 && temp <= 310) return 'Optimal temperature range';
  if (temp <= 373) return 'Acceptable for liquid water';
  if (temp <= 450) return 'Very hot';
  return 'Extremely hot';
}

function getClassification(score: number): string {
  if (score >= 80) return 'Highly Habitable';
  if (score >= 60) return 'Potentially Habitable';
  if (score >= 40) return 'Marginally Habitable';
  if (score >= 20) return 'Unlikely Habitable';
  return 'Not Habitable';
}

function generateRecommendation(score: number, classification: string, params: HabitabilityParameters): string {
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
