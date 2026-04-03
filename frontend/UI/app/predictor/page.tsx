"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Navbar } from "@/components/navbar";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/components/toast-provider";
import { getCurrentUser, savePrediction, PredictionRecord } from "@/lib/storage";
import { CheckCircle } from "lucide-react";

export default function Predictor() {
  const router = useRouter();
  const { addToast } = useToast();

  const [user, setUser] = useState<any>(null);
  const [planetName, setPlanetName] = useState("");

  // ✅ CORRECT ML INPUTS
  const [mass, setMass] = useState(100);
  const [density, setDensity] = useState(2);
  const [starTemp, setStarTemp] = useState(5500);
  const [metallicity, setMetallicity] = useState(0);

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const currentUser = getCurrentUser();
    if (!currentUser) {
      router.push("/auth/signin");
      return;
    }
    setUser(currentUser);
    setLoading(false);
  }, [router]);

  // 🔥 Score Animation
  useEffect(() => {
    if (!result) return;

    let start = 0;
    const interval = setInterval(() => {
      start += 2;
      if (start >= result.score) {
        start = result.score;
        clearInterval(interval);
      }
      setAnimatedScore(start);
    }, 10);

    return () => clearInterval(interval);
  }, [result]);

  if (loading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        Loading...
      </div>
    );
  }

  // 🚀 API CALL
  const handlePredict = async () => {
    if (!planetName.trim()) {
      alert("Enter planet name");
      return;
    }

    setPredicting(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pl_bmasse: mass,
          pl_dens: density,
          st_teff: starTemp,
          st_met: metallicity,
        }),
      });

      const data = await response.json();

      setResult({
        score: Math.round(data.habitability_score * 100),
        classification:
          data.prediction === 1 ? "Habitable 🌍" : "Not Habitable ❌",
      });

      

    } catch (err) {
      console.error(err);
      alert("Backend error");
    }

    setPredicting(false);
  };

  // 💾 SAVE
  const handleSavePrediction = () => {
    if (!result) return;

    const prediction: PredictionRecord = {
      id: Date.now().toString(),
      userId: user.id,
      planetName,
      parameters: { mass, density, starTemp, metallicity },
      habitabilityScore: result.score,
      classification: result.classification,
      timestamp: new Date().toISOString(),
    };

    savePrediction(prediction);
    addToast("Saved!", "success", 2000);

    setPlanetName("");
    setResult(null);

    setTimeout(() => router.push("/dashboard"), 500);
  };

  const circumference = 2 * Math.PI * 60;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />

      <main className="pt-20 px-6 pb-12 max-w-6xl mx-auto">

        <h1 className="text-3xl font-bold mb-8 text-white drop-shadow-[0_0_10px_rgba(168,85,247,0.2)]">
          Exoplanet Habitability Predictor
        </h1>

        <div className="grid lg:grid-cols-2 gap-8">

          {/* INPUT PANEL */}
          <Card className="p-6 space-y-5 bg-white/5 backdrop-blur-xl border border-white/10 shadow-lg">
            <p className="text-sm mb-0.5">Planet Name</p>
            <input
              placeholder="eg. PLATO"
              value={planetName}
              onChange={(e) => setPlanetName(e.target.value)}
              className="w-full p-3 rounded-lg bg-black/40 border border-white/10"
            />

            <div>
              <p className="text-sm mb-1">Planet Mass: {mass}</p>
              <input
                type="range"
                min="0"
                max="1000"
                value={mass}
                onChange={(e) => setMass(parseFloat(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <p className="text-sm mb-1">Density: {density}</p>
              <input
                type="range"
                min="0"
                max="10"
                step="0.1"
                value={density}
                onChange={(e) => setDensity(parseFloat(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <p className="text-sm mb-1">Star Temperature: {starTemp}K</p>
              <input
                type="range"
                min="2000"
                max="8000"
                value={starTemp}
                onChange={(e) => setStarTemp(parseInt(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <p className="text-sm mb-1">Metallicity: {metallicity}</p>
              <input
                type="range"
                min="-1"
                max="1"
                step="0.1"
                value={metallicity}
                onChange={(e) => setMetallicity(parseFloat(e.target.value))}
                className="w-full accent-blue-500"
              />
            </div>

            <Button
              onClick={handlePredict}
              className="w-full py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:scale-105 transition"
            >
              🚀 Analyze Planet
            </Button>

            {predicting && (
              <p className="text-purple-400 animate-pulse text-center">
                🔭 Analyzing...
              </p>
            )}

          </Card>

          {/* RESULT PANEL */}
          <Card className="p-6 flex items-center justify-center bg-white/5 backdrop-blur-xl border border-white/10 shadow-lg">

            {result ? (
              <div className="text-center space-y-6">

                {/* 🔥 Animated Meter */}
                <div className="relative flex items-center justify-center">

                  <svg height="160" width="160">

                    <circle
                      stroke="rgba(255,255,255,0.1)"
                      fill="transparent"
                      strokeWidth="10"
                      r="60"
                      cx="80"
                      cy="80"
                    />

                    <circle
                      stroke="url(#gradient)"
                      fill="transparent"
                      strokeWidth="10"
                      strokeLinecap="round"
                      strokeDasharray={`${circumference}`}
                      strokeDashoffset={
                        circumference - (animatedScore / 100) * circumference
                      }
                      r="60"
                      cx="80"
                      cy="80"
                    />

                    <defs>
                      <linearGradient id="gradient">
                        <stop offset="0%" stopColor="#a855f7" />
                        <stop offset="100%" stopColor="#3b82f6" />
                      </linearGradient>
                    </defs>

                  </svg>

                  <div className="absolute text-center">
                    <div className="text-3xl font-bold text-white">
                      {animatedScore}
                    </div>
                    <div className="text-xs text-gray-400">Score</div>
                  </div>

                </div>

                <p className="text-lg font-semibold text-green-400">
                  {result.classification}
                </p>

                <div className="text-left mt-4 space-y-3">

                  <p className="text-sm font-semibold text-white">Analysis:</p>

                  <div className="space-y-2">

                    <p className="text-sm text-gray-300 bg-white/5 p-2 rounded-lg">
                      🌍 <b>Mass:</b>{" "}
                      {mass < 50
                        ? "Too small to retain atmosphere"
                        : mass <= 500
                        ? "Optimal mass for habitability"
                        : "High gravity may affect surface conditions"}
                    </p>

                    <p className="text-sm text-gray-300 bg-white/5 p-2 rounded-lg">
                      🪐 <b>Density:</b>{" "}
                      {density < 1
                        ? "Likely gaseous planet"
                        : density <= 5
                        ? "Supports solid surface"
                        : "Extremely dense composition"}
                    </p>

                    <p className="text-sm text-gray-300 bg-white/5 p-2 rounded-lg">
                      🌞 <b>Star Temperature:</b>{" "}
                      {starTemp >= 2500 && starTemp <= 6000
                        ? "Suitable for habitable zone"
                        : starTemp < 2500
                        ? "Too cold for stable energy"
                        : "Too hot, may cause radiation issues"}
                    </p>

                    <p className="text-sm text-gray-300 bg-white/5 p-2 rounded-lg">
                      ✨ <b>Metallicity:</b>{" "}
                      {metallicity > 0
                        ? "Rich in heavy elements, supports planet formation"
                        : metallicity === 0
                        ? "Neutral metallicity"
                        : "Low metallicity, fewer building materials"}
                    </p>

                  </div>

                </div>

                <Button onClick={handleSavePrediction}>
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Save
                </Button>

              </div>
            ) : (
              <p className="text-gray-400">Enter inputs to predict</p>
            )}

          </Card>

        </div>
      </main>
    </div>
  );
}