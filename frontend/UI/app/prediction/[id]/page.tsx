"use client";

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import Link from 'next/link';
import { Navbar } from '@/components/navbar';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { getCurrentUser, getAllPredictions, PredictionRecord } from '@/lib/storage';
import { ArrowLeft } from 'lucide-react';

export default function PredictionDetail() {
  const router = useRouter();
  const params = useParams();
  const predictionId = params.id as string;

  const [user, setUser] = useState<any>(null);
  const [prediction, setPrediction] = useState<PredictionRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);

    const currentUser = getCurrentUser();
    if (!currentUser) {
      router.push('/auth/signin');
      return;
    }

    setUser(currentUser);

    const allPredictions = getAllPredictions();
    const found = allPredictions.find(p => p.id === predictionId);

    if (found && found.userId === currentUser.id) {
      setPrediction(found);
    }

    setLoading(false);
  }, [router, predictionId]);

  // 🚨 Hydration Fix
  if (!mounted) return null;

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-muted-foreground animate-pulse">Loading...</div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="min-h-screen">
        <Navbar />
        <main className="pt-20 px-6">
          <Link href="/dashboard">
            <Button variant="outline" className="mb-4">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
          </Link>

          <Card className="p-8 text-center">
            Prediction not found
          </Card>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />

      <main className="pt-20 px-6 max-w-4xl mx-auto space-y-6">

        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-white">
            {prediction.planetName}
          </h1>
          <p className="text-muted-foreground">
            {new Date(prediction.timestamp).toLocaleDateString()} at{" "}
            {new Date(prediction.timestamp).toLocaleTimeString()}
          </p>
        </div>

        {/* Score Card */}
        <Card className="bg-gradient-to-br from-accent/20 to-accent/5 border border-accent/30 p-6 space-y-4">
          <div className="grid md:grid-cols-3 gap-6">

            <div>
              <p className="text-sm text-muted-foreground">Habitability Score</p>
              <p className="text-4xl font-bold text-accent">
                {prediction.habitabilityScore}
              </p>
            </div>

            <div>
              <p className="text-sm text-muted-foreground">Classification</p>
              <p className="text-xl font-semibold">
                {prediction.classification}
              </p>
            </div>

            <div>
              <p className="text-sm text-muted-foreground">Overall Score</p>
              <div className="w-full bg-muted rounded-full h-3 mt-2">
                <div
                  className="bg-accent h-3 rounded-full transition-all duration-1000"
                  style={{ width: `${prediction.habitabilityScore}%` }}
                />
              </div>
            </div>

          </div>
        </Card>

        {/* Parameters */}
        <Card className="p-6 space-y-4">
          <h2 className="text-xl font-semibold text-white">Planetary Parameters</h2>

          <div className="grid md:grid-cols-2 gap-4">

            <p>🌍 Mass: {prediction.parameters.mass}</p>
            <p>🪐 Density: {prediction.parameters.density}</p>
            <p>🌞 Star Temp: {prediction.parameters.starTemp}</p>
            <p>✨ Metallicity: {prediction.parameters.metallicity}</p>

          </div>
        </Card>

        {/* 🔥 FINAL ANALYSIS */}
        <Card className="p-6 space-y-4">
          <h2 className="text-xl font-semibold text-white">
            Analysis Summary
          </h2>

          <ul className="space-y-3">

            <li className="text-sm text-gray-300 bg-white/5 p-2 rounded-lg">
              🌍 <b>Mass:</b>{" "}
              {prediction.parameters.mass < 50
                ? "Too small to retain atmosphere"
                : prediction.parameters.mass <= 500
                ? "Optimal mass for habitability"
                : "High gravity may affect surface conditions"}
            </li>

            <li className="text-sm text-gray-300 bg-white/5 p-2 rounded-lg">
              🪐 <b>Density:</b>{" "}
              {prediction.parameters.density < 1
                ? "Likely gaseous planet"
                : prediction.parameters.density <= 5
                ? "Supports solid surface"
                : "Extremely dense composition"}
            </li>

            <li className="text-sm text-gray-300 bg-white/5 p-2 rounded-lg">
              🌞 <b>Star Temperature:</b>{" "}
              {prediction.parameters.starTemp >= 2500 &&
              prediction.parameters.starTemp <= 6000
                ? "Suitable for habitable zone"
                : prediction.parameters.starTemp < 2500
                ? "Too cold for stable energy"
                : "Too hot, may cause radiation issues"}
            </li>

            <li className="text-sm text-gray-300 bg-white/5 p-2 rounded-lg">
              ✨ <b>Metallicity:</b>{" "}
              {prediction.parameters.metallicity > 0
                ? "Rich in heavy elements, supports planet formation"
                : prediction.parameters.metallicity === 0
                ? "Neutral metallicity"
                : "Low metallicity, fewer building materials"}
            </li>

          </ul>
        </Card>

        {/* Recommendation */}
        <Card className="p-6 space-y-4">
          <h3 className="font-semibold">Recommendation</h3>
          <p className="text-muted-foreground">
            {prediction.habitabilityScore >= 80 &&
              'Excellent habitability potential. Prioritize further observation.'}
            {prediction.habitabilityScore >= 60 && prediction.habitabilityScore < 80 &&
              'Promising conditions. Further analysis recommended.'}
            {prediction.habitabilityScore >= 40 && prediction.habitabilityScore < 60 &&
              'Moderate potential. Requires deeper study.'}
            {prediction.habitabilityScore < 40 &&
              'Low habitability likelihood under known conditions.'}
          </p>
        </Card>

      </main>
    </div>
  );
}