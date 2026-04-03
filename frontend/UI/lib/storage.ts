// Safe localStorage wrapper (VERY IMPORTANT)
function isBrowser() {
  return typeof window !== "undefined";
}

// ---------------- TYPES ----------------
export interface User {
  id: string;
  email: string;
  name: string;
  createdAt: string;
}

export interface PredictionRecord {
  id: string;
  userId: string;
  planetName: string;
  
  parameters: {
  mass: number;
  density: number;
  starTemp: number;
  metallicity: number;
};
  habitabilityScore: number;
  classification: string;
  timestamp: string;
}

const USERS_KEY = 'exohabitat_users';
const CURRENT_USER_KEY = 'exohabitat_current_user';
const PREDICTIONS_KEY = 'exohabitat_predictions';

// ---------------- USER ----------------
export function getUserFromStorage(email: string, password: string): User | null {
  if (!isBrowser()) return null;

  const users = JSON.parse(localStorage.getItem(USERS_KEY) || '[]');
  const userRecord = users.find((u: any) => u.email === email && u.password === password);
  return userRecord?.user || null;
}

export function saveUserToStorage(email: string, password: string, name: string): User {
  if (!isBrowser()) return null as any;

  const users = JSON.parse(localStorage.getItem(USERS_KEY) || '[]');

  const user: User = {
    id: Date.now().toString(),
    email,
    name,
    createdAt: new Date().toISOString(),
  };

  users.push({ email, password, user });
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
  localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));

  return user;
}

export function setCurrentUser(user: User | null): void {
  if (!isBrowser()) return;

  if (user) {
    localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));
  } else {
    localStorage.removeItem(CURRENT_USER_KEY);
  }
}

export function getCurrentUser(): User | null {
  if (!isBrowser()) return null;

  const stored = localStorage.getItem(CURRENT_USER_KEY);
  return stored ? JSON.parse(stored) : null;
}

export function emailExists(email: string): boolean {
  if (!isBrowser()) return false;

  const users = JSON.parse(localStorage.getItem(USERS_KEY) || '[]');
  return users.some((u: any) => u.email === email);
}

// ---------------- PREDICTIONS ----------------
export function savePrediction(prediction: PredictionRecord): void {
  if (!isBrowser()) return;

  const predictions = JSON.parse(localStorage.getItem(PREDICTIONS_KEY) || '[]');
  predictions.push(prediction);
  localStorage.setItem(PREDICTIONS_KEY, JSON.stringify(predictions));
}

export function getPredictionsForUser(userId: string): PredictionRecord[] {
  if (!isBrowser()) return [];

  const predictions = JSON.parse(localStorage.getItem(PREDICTIONS_KEY) || '[]');
  return predictions
    .filter((p: any) => p.userId === userId)
    .sort((a: any, b: any) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
}

export function getAllPredictions(): PredictionRecord[] {
  if (!isBrowser()) return [];

  return JSON.parse(localStorage.getItem(PREDICTIONS_KEY) || '[]');
}

export function clearAllData(): void {
  if (!isBrowser()) return;

  localStorage.removeItem(USERS_KEY);
  localStorage.removeItem(CURRENT_USER_KEY);
  localStorage.removeItem(PREDICTIONS_KEY);
}

// ---------------- DEMO ----------------
export function initializeDemoAccount(): void {
  if (!isBrowser()) return;

  const users = JSON.parse(localStorage.getItem(USERS_KEY) || '[]');

  if (users.length === 0) {
    saveUserToStorage('demo@exohabitat.com', 'demo123', 'Demo Astronomer');
  }
}