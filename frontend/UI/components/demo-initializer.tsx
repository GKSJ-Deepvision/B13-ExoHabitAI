'use client';

import { useEffect } from 'react';
import { initializeDemoAccount } from '@/lib/storage';

export function DemoInitializer() {
  useEffect(() => {
    initializeDemoAccount();
  }, []);

  return null;
}
