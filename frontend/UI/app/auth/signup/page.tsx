'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Navbar } from '@/components/navbar';
import { saveUserToStorage, emailExists } from '@/lib/storage';

export default function SignUp() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Validation
    if (!email || !password || !name) {
      setError('Please fill in all fields');
      setLoading(false);
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      setLoading(false);
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    if (emailExists(email)) {
      setError('Email already registered');
      setLoading(false);
      return;
    }

    try {
      saveUserToStorage(email, password, name);
      router.push('/dashboard');
    } catch (err) {
      setError('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      
      <main className="pt-20 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-md animate-scale-in">
          <div className="bg-card border border-border rounded-lg p-8 space-y-6 card-hover-lift">
            <div className="space-y-2 text-center">
              <h1 className="text-2xl font-bold">Create Account</h1>
              <p className="text-muted-foreground">
                Join ExoHabitAI to start analyzing exoplanets
              </p>
            </div>

            {error && (
              <div className="bg-destructive/10 border border-destructive text-destructive rounded-lg p-3 text-sm animate-scale-in">
                {error}
              </div>
            )}

            <form onSubmit={handleSignUp} className="space-y-4">
              <div className="space-y-2 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
                <label htmlFor="name" className="text-sm font-medium">
                  Full Name
                </label>
                <input
                  id="name"
                  type="text"
                  placeholder="Dr. Astro Scientist"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={loading}
                  className="w-full px-3 py-2 rounded-lg bg-background border border-border input-glow text-foreground placeholder:text-muted-foreground transition-all duration-200 disabled:opacity-50"
                />
              </div>

              <div className="space-y-2 animate-fade-in-up" style={{ animationDelay: '150ms' }}>
                <label htmlFor="email" className="text-sm font-medium">
                  Email
                </label>
                <input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={loading}
                  className="w-full px-3 py-2 rounded-lg bg-background border border-border input-glow text-foreground placeholder:text-muted-foreground transition-all duration-200 disabled:opacity-50"
                />
              </div>

              <div className="space-y-2 animate-fade-in-up" style={{ animationDelay: '200ms' }}>
                <label htmlFor="password" className="text-sm font-medium">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={loading}
                  className="w-full px-3 py-2 rounded-lg bg-background border border-border input-glow text-foreground placeholder:text-muted-foreground transition-all duration-200 disabled:opacity-50"
                />
              </div>

              <div className="space-y-2 animate-fade-in-up" style={{ animationDelay: '250ms' }}>
                <label htmlFor="confirm-password" className="text-sm font-medium">
                  Confirm Password
                </label>
                <input
                  id="confirm-password"
                  type="password"
                  placeholder="••••••••"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  disabled={loading}
                  className="w-full px-3 py-2 rounded-lg bg-background border border-border input-glow text-foreground placeholder:text-muted-foreground transition-all duration-200 disabled:opacity-50"
                />
              </div>

              <Button
                type="submit"
                className="w-full button-hover-glow animate-fade-in-up"
                style={{ animationDelay: '300ms' }}
                disabled={loading}
              >
                {loading ? 'Creating Account...' : 'Sign Up'}
              </Button>
            </form>

            <div className="text-center text-sm animate-fade-in-up" style={{ animationDelay: '350ms' }}>
              Already have an account?{' '}
              <Link href="/auth/signin" className="text-accent hover:underline transition-colors">
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
