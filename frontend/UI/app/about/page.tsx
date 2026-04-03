'use client';

import Link from 'next/link';
import { Navbar } from '@/components/navbar';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { BookOpen, Code2, GitBranch } from 'lucide-react';

export default function About() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      
      <main className="pt-20 px-4 sm:px-6 lg:px-8 pb-12">
        <div className="max-w-4xl mx-auto space-y-12">
          {/* Header */}
          <div className="text-center space-y-4">
            <h1 className="text-3xl font-bold text-white drop-shadow-[0_0_10px_rgba(168,85,247,0.2)]">About ExoHabitAI</h1>
            <p className="text-lg text-muted-foreground">
              Exploring the cosmos through data-driven analysis
            </p>
          </div>

          {/* Mission */}
          <Card className="bg-card border border-border p-8 space-y-4">
            <h2 className="text-2xl font-bold">Our Mission</h2>
            <p className="text-muted-foreground leading-relaxed">
              ExoHabitAI brings cutting-edge artificial intelligence to exoplanet research. We believe that understanding 
              which distant worlds might support life is one of humanity&apos;s most profound scientific pursuits. By combining 
              advanced machine learning algorithms with comprehensive exoplanet data, we make habitability analysis 
              accessible to researchers, students, and space enthusiasts worldwide.
            </p>
          </Card>

          {/* How It Works */}
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">How It Works</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <Card className="bg-card border border-border p-6 space-y-4">
                <div className="w-12 h-12 bg-accent/20 rounded-lg flex items-center justify-center">
                  <Code2 className="w-6 h-6 text-accent" />
                </div>
                <h3 className="font-semibold text-lg">Parameter Analysis</h3>
                <p className="text-muted-foreground">
                  Our system evaluates key exoplanet characteristics including size, orbital distance, atmospheric composition, 
                  temperature, and water presence.
                </p>
              </Card>

              <Card className="bg-card border border-border p-6 space-y-4">
                <div className="w-12 h-12 bg-accent/20 rounded-lg flex items-center justify-center">
                  <GitBranch className="w-6 h-6 text-accent" />
                </div>
                <h3 className="font-semibold text-lg">ML Algorithms</h3>
                <p className="text-muted-foreground">
                  Machine learning models synthesize these parameters to predict habitability scores, accounting for complex 
                  interactions between planetary conditions.
                </p>
              </Card>

              <Card className="bg-card border border-border p-6 space-y-4">
                <div className="w-12 h-12 bg-accent/20 rounded-lg flex items-center justify-center">
                  <BookOpen className="w-6 h-6 text-accent" />
                </div>
                <h3 className="font-semibold text-lg">Detailed Reports</h3>
                <p className="text-muted-foreground">
                  Users receive comprehensive reports breaking down each factor&apos;s contribution to the final habitability 
                  assessment.
                </p>
              </Card>
            </div>
          </div>

          {/* Methodology */}
          <Card className="bg-card border border-border p-8 space-y-6">
            <h2 className="text-2xl font-bold">Our Methodology</h2>
            
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-lg mb-2">Habitability Zone</h3>
                <p className="text-muted-foreground">
                  We calculate the habitable zone around each star, typically between 0.8-1.5 AU, where liquid water could exist 
                  on a planet&apos;s surface. Planets within this zone receive priority consideration.
                </p>
              </div>

              <div>
                <h3 className="font-semibold text-lg mb-2">Planetary Size</h3>
                <p className="text-muted-foreground">
                  Super-Earths and Earth-sized planets (0.8-1.2 Earth radii) are considered ideal. These sizes can support dense 
                  atmospheres while maintaining a stable climate. Larger planets may become mini-Neptunes unsuitable for terrestrial life.
                </p>
              </div>

              <div>
                <h3 className="font-semibold text-lg mb-2">Atmospheric Composition</h3>
                <p className="text-muted-foreground">
                  An atmosphere is crucial for temperature regulation and protection from stellar radiation. Our model heavily 
                  weights the presence of a substantial atmosphere in the habitability calculation.
                </p>
              </div>

              <div>
                <h3 className="font-semibold text-lg mb-2">Temperature Range</h3>
                <p className="text-muted-foreground">
                  Optimal temperatures for life (Earth-like: 250-310K) support liquid water and biological processes. Extreme 
                  temperatures indicate a hostile environment for known life forms.
                </p>
              </div>

              <div>
                <h3 className="font-semibold text-lg mb-2">Water Presence</h3>
                <p className="text-muted-foreground">
                  Water is essential for life as we understand it. We estimate the probability of water presence based on 
                  temperature, atmospheric properties, and planetary composition data.
                </p>
              </div>
            </div>
          </Card>

          {/* Limitations */}
          <Card className="bg-muted/50 border border-border p-8 space-y-4">
            <h2 className="text-2xl font-bold">Limitations & Disclaimers</h2>
            <ul className="space-y-2 text-muted-foreground list-disc list-inside">
              <li>Our predictions are based on current scientific understanding and may change as research evolves</li>
              <li>Habitability scoring is a simplification of extremely complex exoplanet science</li>
              <li>Detection methods have observational biases affecting the exoplanet database</li>
              <li>Life could theoretically exist under conditions we haven&apos;t considered or detected</li>
              <li>This tool is for educational and research purposes, not definitive scientific assessment</li>
            </ul>
          </Card>

          {/* CTA */}
          <div className="text-center space-y-4">
            <p className="text-lg">Ready to explore the universe?</p>
            <Link href="/predictor">
              <Button size="lg">Start Analyzing</Button>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
