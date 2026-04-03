'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Navbar } from '@/components/navbar';
import { ArrowRight, Zap, Globe, Microscope } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      
      <main className="pt-20">
        {/* Hero Section */}
        <section className="relative overflow-hidden py-20 px-4 sm:px-6 lg:px-8">
          <div className="absolute inset-0 bg-gradient-to-br from-accent/10 via-transparent to-transparent pointer-events-none" />
          
          <div className="relative max-w-7xl mx-auto">
            <div className="text-center space-y-6 mb-16 animate-fade-in-up">
              <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight text-balance">
                Discover Habitable <span className="text-accent animate-glow-pulse">Exoplanets</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto text-balance">
                Analyze distant worlds with ExoHabitAI. Use advanced algorithms to predict which exoplanets might support life.
              </p>
              <div className="flex gap-4 justify-center flex-wrap">
                <Link href="/auth/signup">
                  <Button size="lg" className="gap-2 button-hover-glow">
                    Get Started <ArrowRight className="w-4 h-4" />
                  </Button>
                </Link>
                <Link href="/explore">
                  <Button variant="outline" size="lg" className="button-hover-glow">
                    Browse Exoplanets
                  </Button>
                </Link>
              </div>
            </div>

            {/* Feature Cards */}
            <div className="grid md:grid-cols-3 gap-6 mt-16">
              {[
                {
                  icon: Zap,
                  title: 'AI Analysis',
                  desc: 'Predict habitability scores using advanced machine learning algorithms trained on exoplanet data.',
                },
                {
                  icon: Globe,
                  title: 'Explore Worlds',
                  desc: 'Browse a catalog of exoplanets and discover their characteristics and habitability potential.',
                },
                {
                  icon: Microscope,
                  title: 'Deep Analysis',
                  desc: 'Get detailed reports on atmospheric composition, temperature, orbital mechanics, and water presence.',
                },
              ].map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <div
                    key={feature.title}
                    className="bg-card border border-border rounded-lg p-8 space-y-4 card-hover-lift animate-fade-in-up"
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div className="w-12 h-12 bg-accent/20 rounded-lg flex items-center justify-center animate-glow-pulse">
                      <Icon className="w-6 h-6 text-accent" />
                    </div>
                    <h3 className="text-lg font-semibold">{feature.title}</h3>
                    <p className="text-muted-foreground">{feature.desc}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 px-4 sm:px-6 lg:px-8 bg-card border-t border-border animate-fade-in-up" style={{ animationDelay: '300ms' }}>
          <div className="max-w-4xl mx-auto text-center space-y-6">
            <h2 className="text-3xl sm:text-4xl font-bold text-balance">
              Ready to Explore the Universe?
            </h2>
            <p className="text-lg text-muted-foreground">
              Create an account to start analyzing exoplanets and building your personal catalog of discoveries.
            </p>
            <Link href="/auth/signup">
              <Button size="lg" className="gap-2 button-hover-glow">
                Sign Up Now <ArrowRight className="w-4 h-4" />
              </Button>
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}
