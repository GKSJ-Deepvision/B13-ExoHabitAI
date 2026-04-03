'use client';


import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { useTheme } from '@/components/theme-provider';
import { Moon, Sun, Menu, X } from 'lucide-react';
import { getCurrentUser, setCurrentUser } from '@/lib/storage';
import { useState, useEffect } from 'react';
import { usePathname } from "next/navigation";

export function Navbar() {
  const { isDark, toggleTheme } = useTheme();
  const [isOpen, setIsOpen] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [mounted, setMounted] = useState(false);
  const pathname = usePathname();
  const isActive = (path: string) => pathname === path;

  useEffect(() => {
    setMounted(true);

    const currentUser = getCurrentUser();
    if (currentUser) {
      setUser(currentUser);
    }
  }, []);
  
  if (!mounted) return null;

  const handleLogout = () => {
    setCurrentUser(null);
    window.location.href = '/';

    
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-sm border-b border-border animate-slide-in-top">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 font-bold text-xl hover:opacity-80 transition-opacity">
            <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center text-accent-foreground text-sm font-bold animate-glow-pulse">
              E
            </div>
            <span>ExoHabitAI</span>
          </Link>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center gap-8">
            <Link
              href="/"
              className={`transition-colors duration-200 ${
                isActive("/")
                  ? "text-foreground animate-glow-pulse"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Home
            </Link>
            {user && (
              <>
                <Link
                  href="/dashboard"
                  className={`transition-colors duration-200 ${
                    isActive("/dashboard")
                      ? "text-foreground animate-glow-pulse"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Dashboard
                </Link>
                <Link
                  href="/predictor"
                  className={`transition-colors duration-200 ${
                    isActive("/predictor")
                      ? "text-foreground animate-glow-pulse"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Predictor
                </Link>
                <Link
                  href="/explore"
                  className={`transition-colors duration-200 ${
                    isActive("/explore")
                      ? "text-foreground animate-glow-pulse"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Explore
                </Link>
              </>
            )}
            <Link href="/about" className="text-muted-foreground hover:text-foreground transition-colors duration-200">
              About
            </Link>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-4">
            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg hover:bg-card border border-border transition-all duration-200 hover:shadow-lg hover:shadow-accent/20"
              aria-label="Toggle theme"
            >
              {isDark ? (
                <Sun className="w-4 h-4 text-accent animate-fade-in" />
              ) : (
                <Moon className="w-4 h-4 text-accent animate-fade-in" />
              )}
            </button>

            {/* Auth Buttons */}
            <div className="hidden md:flex items-center gap-2">
              {user ? (
                <>
                  <span className="text-sm text-muted-foreground">{user.email}</span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleLogout}
                    className="button-hover-glow"
                  >
                    Logout
                  </Button>
                </>
              ) : (
                <>
                  <Link href="/auth/signin">
                    <Button variant="outline" size="sm" className="button-hover-glow">
                      Sign In
                    </Button>
                  </Link>
                  <Link href="/auth/signup">
                    <Button size="sm" className="button-hover-glow">
                      Sign Up
                    </Button>
                  </Link>
                </>
              )}
            </div>

            {/* Mobile Menu Toggle */}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-card border border-border transition-all duration-200"
            >
              {isOpen ? (
                <X className="w-5 h-5 animate-scale-in" />
              ) : (
                <Menu className="w-5 h-5 animate-scale-in" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="md:hidden border-t border-border py-4 space-y-4 animate-scale-in">
            <Link
              href="/"
              className="block text-muted-foreground hover:text-foreground transition-colors duration-200"
              onClick={() => setIsOpen(false)}
            >
              Home
            </Link>
            {user && (
              <>
                <Link
                  href="/dashboard"
                  className="block text-muted-foreground hover:text-foreground transition-colors duration-200"
                  onClick={() => setIsOpen(false)}
                >
                  Dashboard
                </Link>
                <Link
                  href="/predictor"
                  className="block text-muted-foreground hover:text-foreground transition-colors duration-200"
                  onClick={() => setIsOpen(false)}
                >
                  Predictor
                </Link>
                <Link
                  href="/explore"
                  className="block text-muted-foreground hover:text-foreground transition-colors duration-200"
                  onClick={() => setIsOpen(false)}
                >
                  Explore
                </Link>
              </>
            )}
            <Link
              href="/about"
              className="block text-muted-foreground hover:text-foreground transition-colors duration-200"
              onClick={() => setIsOpen(false)}
            >
              About
            </Link>

            <div className="pt-4 border-t border-border space-y-2">
              {user ? (
                <>
                  <div className="text-sm text-muted-foreground px-4">{user.email}</div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      handleLogout();
                      setIsOpen(false);
                    }}
                    className="w-full button-hover-glow"
                  >
                    Logout
                  </Button>
                </>
              ) : (
                <>
                  <Link href="/auth/signin" onClick={() => setIsOpen(false)}>
                    <Button variant="outline" size="sm" className="w-full button-hover-glow">
                      Sign In
                    </Button>
                  </Link>
                  <Link href="/auth/signup" onClick={() => setIsOpen(false)}>
                    <Button size="sm" className="w-full button-hover-glow">
                      Sign Up
                    </Button>
                  </Link>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
