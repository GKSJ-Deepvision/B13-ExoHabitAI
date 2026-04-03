'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Navbar } from '@/components/navbar';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { AnimatedCounter } from '@/components/animated-counter';
import { SkeletonLoader } from '@/components/skeleton-loader';
import { useToast } from '@/components/toast-provider';
import { getCurrentUser, getPredictionsForUser, PredictionRecord } from '@/lib/storage';
import { BarChart3, Plus, Trash2 } from 'lucide-react';

export default function Dashboard() {
  const router = useRouter();
  const { addToast } = useToast();
  const [user, setUser] = useState<any>(null);
  const [predictions, setPredictions] = useState<PredictionRecord[]>([]);
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
    const userPredictions = getPredictionsForUser(currentUser.id);
    setPredictions(userPredictions);
    setLoading(false);
  }, [router]);

  if (loading || !user) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-muted-foreground animate-pulse">Loading...</div>
      </div>
    );
  }

  const deletePrediction = (id: string) => {
    const allPredictions = JSON.parse(localStorage.getItem('exohabitat_predictions') || '[]');
    const filtered = allPredictions.filter((p: PredictionRecord) => p.id !== id);
    localStorage.setItem('exohabitat_predictions', JSON.stringify(filtered));
    setPredictions(predictions.filter(p => p.id !== id));
    addToast('Prediction deleted successfully', 'info', 2000);
  };

  const highlyHabitable = predictions.filter(p => p.habitabilityScore >= 80).length;
  const potentiallyHabitable = predictions.filter(p => p.habitabilityScore >= 60 && p.habitabilityScore < 80).length;
  

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      
      <main className="pt-20 px-4 sm:px-6 lg:px-8 pb-12">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="space-y-6 mb-8 animate-fade-in-up">
            <div>
              <h1 className="text-3xl font-bold text-white drop-shadow-[0_0_10px_rgba(168,85,247,0.2)]">Dashboard</h1>
              <p className="text-muted-foreground">Welcome back, {user.name}</p>
            </div>

            <div className="grid md:grid-cols-4 gap-4">
              <Card className="bg-card border border-border p-6 space-y-2 card-hover-lift">
                <p className="text-muted-foreground text-sm">Total Predictions</p>
                <p className="text-3xl font-bold text-accent">
                  <AnimatedCounter value={predictions.length} />
                </p>
              </Card>
              <Card className="bg-card border border-border p-6 space-y-2 card-hover-lift">
                <p className="text-muted-foreground text-sm">Highly Habitable</p>
                <p className="text-3xl font-bold text-accent">
                  <AnimatedCounter value={highlyHabitable} />
                </p>
              </Card>
              <Card className="bg-card border border-border p-6 space-y-2 card-hover-lift">
                <p className="text-muted-foreground text-sm">Potentially Habitable</p>
                <p className="text-3xl font-bold text-accent">
                  <AnimatedCounter value={potentiallyHabitable} />
                </p>
              </Card>
              <Card className="bg-card border border-border p-6 space-y-2 card-hover-lift">
                <p className="text-muted-foreground text-sm">Account Created</p>
                <p className="text-sm text-muted-foreground">
                  {new Date(user.createdAt).toLocaleDateString()}
                </p>
              </Card>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex gap-4 mb-8 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
            <Link href="/predictor">
              <Button className="gap-2 button-hover-glow">
                <Plus className="w-4 h-4" />
                New Prediction
              </Button>
            </Link>
            <Link href="/explore">
              <Button variant="outline" className="button-hover-glow">
                Explore Database
              </Button>
            </Link>
          </div>

          {/* Recent Predictions */}
          <div className="space-y-6 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
            <div>
              <h2 className="text-xl font-bold mb-4">Recent Predictions</h2>
              
              
              {predictions.length === 0 ? (
                <Card className="bg-card border border-border p-8 text-center space-y-4 card-hover-lift">
                  <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto opacity-50 animate-pulse-gentle" />
                  <p className="text-muted-foreground">No predictions yet</p>
                  <Link href="/predictor">
                    <Button className="button-hover-glow">Create Your First Prediction</Button>
                  </Link>
                </Card>
              ) : (
                <div className="space-y-3">
                  {predictions.slice(0, 10).map((prediction, index) => (
                    <div
                      key={prediction.id}
                      className="bg-card border border-border rounded-lg p-4 flex justify-between items-center card-hover-lift animate-fade-in-up"
                      style={{ animationDelay: `${300 + index * 50}ms` }}
                    >
                      <div className="flex-1">
                        <h3 className="font-semibold">{prediction.planetName}</h3>
                        <div className="flex gap-4 mt-2 text-sm text-muted-foreground">
                          <span>Score: {prediction.habitabilityScore}</span>
                          <span>{prediction.classification}</span>
                          <span>
                            {new Date(prediction.timestamp).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Link href={`/prediction/${prediction.id}`}>
                          <Button variant="outline" size="sm" className="button-hover-glow">
                            View
                          </Button>
                        </Link>
                        <button
                          onClick={() => deletePrediction(prediction.id)}
                          className="p-2 rounded-lg hover:bg-destructive/10 text-destructive transition-all duration-200 hover:scale-110"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
