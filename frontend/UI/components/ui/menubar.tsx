"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

export function Navbar() {
  const [mounted, setMounted] = useState(false);
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    setMounted(true);

    try {
      const storedUser = localStorage.getItem("user");
      if (storedUser) {
        setUser(JSON.parse(storedUser));
      }
    } catch (err) {
      console.error(err);
    }
  }, []);

  // 🚨 CRITICAL FIX (this stops hydration error)
  if (!mounted) return null;

  return (
    <nav className="fixed top-0 w-full z-50 bg-black/40 backdrop-blur-md border-b border-white/10">
      <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
        
        {/* Logo */}
        <Link href="/" className="text-xl font-bold text-white">
          ExoHabitAI
        </Link>

        {/* Links */}
        <div className="flex gap-6 text-sm">
          {user ? (
            <>
              <Link href="/dashboard" className="hover:text-white">
                Dashboard
              </Link>

              <Link href="/predictor" className="hover:text-white">
                Predictor
              </Link>

              <button
                onClick={() => {
                  localStorage.removeItem("user");
                  window.location.reload();
                }}
                className="text-red-400 hover:text-red-300"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link href="/about" className="hover:text-white">
                About
              </Link>

              <Link href="/auth/signin" className="hover:text-white">
                Login
              </Link>

              <Link
                href="/auth/signup"
                className="px-4 py-2 bg-white text-black rounded-lg"
              >
                Sign Up
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}