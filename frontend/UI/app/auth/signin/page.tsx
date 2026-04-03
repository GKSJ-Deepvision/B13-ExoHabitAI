'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Navbar } from '@/components/navbar';
import { getUserFromStorage, setCurrentUser } from '@/lib/storage';

export default function SignIn() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (!email || !password) {
      setError('Please enter email and password');
      setLoading(false);
      return;
    }

    try {
      const user = getUserFromStorage(email, password);
      if (user) {
        setCurrentUser(user);
        router.push('/dashboard');
      } else {
        setError('Invalid email or password');
      }
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
              <h1 className="text-2xl font-bold">Sign In</h1>
              <p className="text-muted-foreground">
                Welcome back to ExoHabitAI
              </p>
            </div>

            {error && (
              <div className="bg-destructive/10 border border-destructive text-destructive rounded-lg p-3 text-sm animate-scale-in">
                {error}
              </div>
            )}

            <form onSubmit={handleSignIn} className="space-y-4">
              <div className="space-y-2 animate-fade-in-up" style={{ animationDelay: '100ms' }}>
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

              <div className="space-y-2 animate-fade-in-up" style={{ animationDelay: '150ms' }}>
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

              <Button
                type="submit"
                className="w-full button-hover-glow animate-fade-in-up"
                style={{ animationDelay: '200ms' }}
                disabled={loading}
              >
                {loading ? 'Signing In...' : 'Sign In'}
              </Button>
            </form>

            <div className="text-center text-sm animate-fade-in-up" style={{ animationDelay: '250ms' }}>
              Don&apos;t have an account?{' '}
              <Link href="/auth/signup" className="text-accent hover:underline transition-colors">
                Sign Up
              </Link>
            </div>

            {/* Demo credentials hint */}
            <div className="bg-muted/50 border border-border rounded p-3 text-xs text-muted-foreground space-y-1 animate-fade-in-up" style={{ animationDelay: '300ms' }}>
              <p className="font-medium">Demo Account:</p>
              <p>Email: demo@exohabitat.com</p>
              <p>Password: demo123</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
