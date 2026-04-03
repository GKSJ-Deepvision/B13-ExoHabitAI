'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Navbar } from '@/components/navbar';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/toast-provider';
import { Search, Heart } from 'lucide-react';

interface ExoplanetData {
  id: string;
  name: string;
  star: string;
  radius: number;
  distance: number;
  temperature: number;
  atmosphere: boolean;
  discovered: number;
  habitability: number;
}

const EXOPLANETS: ExoplanetData[] = [
  {
    id: '1',
    name: 'TRAPPIST-1e',
    star: 'TRAPPIST-1',
    radius: 0.92,
    distance: 0.02925,
    temperature: 246,
    atmosphere: true,
    discovered: 2017,
    habitability: 86,
  },
  {
    id: '2',
    name: 'Kepler-452b',
    star: 'Kepler-452',
    radius: 1.61,
    distance: 1.05,
    temperature: 265,
    atmosphere: true,
    discovered: 2015,
    habitability: 78,
  },
  {
    id: '3',
    name: 'Proxima Centauri b',
    star: 'Proxima Centauri',
    radius: 1.27,
    distance: 0.0485,
    temperature: 234,
    atmosphere: false,
    discovered: 2016,
    habitability: 62,
  },
  {
    id: '4',
    name: 'WASP-47e',
    star: 'WASP-47',
    radius: 1.83,
    distance: 0.0411,
    temperature: 712,
    atmosphere: true,
    discovered: 2015,
    habitability: 28,
  },
  {
    id: '5',
    name: 'LHS 1140 b',
    star: 'LHS 1140',
    radius: 1.4,
    distance: 0.0369,
    temperature: 258,
    atmosphere: true,
    discovered: 2018,
    habitability: 71,
  },
  {
    id: '6',
    name: 'K2-18b',
    star: 'K2-18',
    radius: 2.49,
    distance: 0.1495,
    temperature: 269,
    atmosphere: true,
    discovered: 2015,
    habitability: 65,
  },
  {
    id: '7',
    name: 'Gliese 667Cc',
    star: 'Gliese 667C',
    radius: 1.5,
    distance: 0.1254,
    temperature: 232,
    atmosphere: true,
    discovered: 2011,
    habitability: 64,
  },
  {
    id: '8',
    name: 'Kepler-186f',
    star: 'Kepler-186',
    radius: 1.11,
    distance: 0.432,
    temperature: 188,
    atmosphere: false,
    discovered: 2014,
    habitability: 51,
  },
];

export default function Explore() {
  const { addToast } = useToast();
  const [searchTerm, setSearchTerm] = useState('');
  const [filterHabitability, setFilterHabitability] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [favorites, setFavorites] = useState<Set<string>>(new Set());

  const toggleFavorite = (planetId: string, planetName: string) => {
    const newFavorites = new Set(favorites);
    if (newFavorites.has(planetId)) {
      newFavorites.delete(planetId);
      addToast(`Removed ${planetName} from favorites`, 'info', 1500);
    } else {
      newFavorites.add(planetId);
      addToast(`Added ${planetName} to favorites`, 'success', 1500);
    }
    setFavorites(newFavorites);
  };

  const filteredPlanets = EXOPLANETS.filter((planet) => {
    const matchesSearch =
      planet.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      planet.star.toLowerCase().includes(searchTerm.toLowerCase());

    let matchesHabitability = true;
    if (filterHabitability === 'high') matchesHabitability = planet.habitability >= 70;
    if (filterHabitability === 'medium') matchesHabitability = planet.habitability >= 40 && planet.habitability < 70;
    if (filterHabitability === 'low') matchesHabitability = planet.habitability < 40;

    return matchesSearch && matchesHabitability;
  });

  const getHabitabilityColor = (score: number) => {
    if (score >= 70) return 'text-green-500';
    if (score >= 40) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getClassification = (score: number) => {
    if (score >= 70) return 'Highly Habitable';
    if (score >= 40) return 'Potentially Habitable';
    return 'Unlikely Habitable';
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />

      <main className="pt-20 px-4 sm:px-6 lg:px-8 pb-12">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8 animate-fade-in-up">
            <h1 className="text-3xl font-bold text-white drop-shadow-[0_0_10px_rgba(168,85,247,0.2)]">Exoplanet Database</h1>
            <p className="text-muted-foreground mt-2">
              Browse known exoplanets and their habitability potential
            </p>
          </div>

          {/* Search and Filter */}
          <div className="mb-8 space-y-4 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
            <div className="relative">
              <Search className="absolute left-3 top-3 w-5 h-5 text-muted-foreground" />
              <input
                placeholder="Search by planet name or star..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 px-3 py-2 rounded-lg bg-background border border-border input-glow text-foreground placeholder:text-muted-foreground transition-all duration-200"
              />
            </div>

            <div className="flex gap-2 flex-wrap">
              {(['all', 'high', 'medium', 'low'] as const).map((filter, index) => (
                <button
                  key={filter}
                  onClick={() => setFilterHabitability(filter)}
                  className="px-4 py-2 rounded-lg border transition button-hover-glow animate-fade-in-up"
                  style={{ animationDelay: `${150 + index * 50}ms` }}
                >
                  <span
                    className={`${
                      filterHabitability === filter
                        ? 'bg-accent text-accent-foreground border-accent block px-2 py-1 rounded'
                        : 'bg-card border-border hover:border-accent'
                    }`}
                  >
                    {filter === 'all' && 'All Planets'}
                    {filter === 'high' && 'Highly Habitable'}
                    {filter === 'medium' && 'Potentially Habitable'}
                    {filter === 'low' && 'Unlikely Habitable'}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Results */}
          {filteredPlanets.length === 0 ? (
            <Card className="bg-card border border-border p-8 text-center animate-fade-in-up">
              <p className="text-muted-foreground">No exoplanets found matching your search.</p>
            </Card>
          ) : (
            <div className="grid md:grid-cols-2 gap-6">
              {filteredPlanets.map((planet, index) => (
                <Card
                  key={planet.id}
                  className="bg-card border border-border p-6 card-hover-lift animate-fade-in-up"
                  style={{ animationDelay: `${300 + index * 50}ms` }}
                >
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-xl font-bold">{planet.name}</h3>
                      <p className="text-sm text-muted-foreground">around {planet.star}</p>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Radius</p>
                        <p className="font-semibold">{planet.radius.toFixed(2)} R⊕</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Distance</p>
                        <p className="font-semibold">{planet.distance.toFixed(4)} AU</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Temperature</p>
                        <p className="font-semibold">{planet.temperature}K</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Discovered</p>
                        <p className="font-semibold">{planet.discovered}</p>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <p className="text-sm font-medium">Habitability</p>
                        <p className={`font-bold ${getHabitabilityColor(planet.habitability)}`}>
                          {planet.habitability}
                        </p>
                      </div>
                      <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-accent h-2 rounded-full transition-all duration-700 ease-out"
                          style={{ width: `${planet.habitability}%` }}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {getClassification(planet.habitability)}
                      </p>
                    </div>

                    <div className="flex justify-between items-start gap-2 pt-2">
                      <div className="flex-1 text-xs">
                        {planet.atmosphere ? (
                          <span className="inline-block bg-accent/20 text-accent px-2 py-1 rounded animate-pulse-gentle">
                            Has Atmosphere
                          </span>
                        ) : (
                          <span className="inline-block bg-muted text-muted-foreground px-2 py-1 rounded opacity-70">
                            No Atmosphere
                          </span>
                        )}
                      </div>

                      <button
                        onClick={() => toggleFavorite(planet.id, planet.name)}
                        className="p-2 rounded-lg hover:bg-accent/20 transition-all duration-200 icon-hover-scale"
                        aria-label="Toggle favorite"
                      >
                        <Heart
                          className={`w-5 h-5 transition-all duration-200 ${
                            favorites.has(planet.id)
                              ? 'fill-accent text-accent'
                              : 'text-muted-foreground'
                          }`}
                        />
                      </button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}

          {/* CTA */}
          <div className="mt-12 text-center animate-fade-in-up" style={{ animationDelay: '400ms' }}>
            <p className="text-muted-foreground mb-4">Want to analyze these planets further?</p>
            <Link href="/predictor">
              <Button className="button-hover-glow">Use the Predictor Tool</Button>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}