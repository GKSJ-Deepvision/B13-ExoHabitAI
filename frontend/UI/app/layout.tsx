import type { Metadata, Viewport } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import { ThemeProvider } from '@/components/theme-provider'
import { DemoInitializer } from '@/components/demo-initializer'
import { ToastProvider } from '@/components/toast-provider'
import { FloatingParticles } from '@/components/floating-particles'
import './globals.css'

const geist = Geist({ subsets: ["latin"], variable: '--font-geist-sans' });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: '--font-geist-mono' });

export const metadata: Metadata = {
  title: 'ExoHabitAI - Exoplanet Habitability Analysis',
  description: 'Discover the habitability potential of distant exoplanets using advanced AI analysis. Explore data-driven predictions about which planets might support life.',
  generator: 'v0.app',
  icons: {
    icon: [
      {
        url: '/icon-light-32x32.png',
        media: '(prefers-color-scheme: light)',
      },
      {
        url: '/icon-dark-32x32.png',
        media: '(prefers-color-scheme: dark)',
      },
      {
        url: '/icon.svg',
        type: 'image/svg+xml',
      },
    ],
    apple: '/apple-icon.png',
  },
}

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0f0f12' }
  ],
  userScalable: false,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body
        suppressHydrationWarning
        className={`${geist.variable} ${geistMono.variable} font-sans antialiased bg-background text-foreground dark`}
      >

{/* </body>
      <body className={`${geist.variable} ${geistMono.variable} font-sans antialiased bg-background text-foreground dark`}> */}
        <ThemeProvider>
          <ToastProvider>
            <FloatingParticles />
            <DemoInitializer />
            {children}
          </ToastProvider>
        </ThemeProvider>
        <Analytics />
      </body>
    </html>
  )
}
