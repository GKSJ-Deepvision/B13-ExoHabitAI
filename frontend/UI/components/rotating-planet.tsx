'use client';

import { useEffect, useRef } from 'react';

interface RotatingPlanetProps {
  color?: string;
  size?: number;
  speed?: number;
}

export function RotatingPlanet({
  color = 'rgba(101, 84, 255, 0.6)',
  size = 100,
  speed = 20,
}: RotatingPlanetProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = size;
    canvas.height = size;

    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size / 2 - 5;

    let rotation = 0;
    let animationId: number;

    const animate = () => {
      // Clear canvas
      ctx.fillStyle = 'transparent';
      ctx.clearRect(0, 0, size, size);

      // Draw planet
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw planet details (stripes)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      for (let i = 0; i < 3; i++) {
        const offset = (i - 1) * 10 + (rotation / 10) % 10;
        ctx.beginPath();
        ctx.ellipse(centerX, centerY + offset, radius * 0.9, radius * 0.3, 0, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Draw rotation ring
      ctx.strokeStyle = `rgba(101, 84, 255, ${0.3 + Math.sin(rotation / 100) * 0.2})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius + 8, 0, Math.PI * 2);
      ctx.stroke();

      // Draw rotating point
      const pointAngle = (rotation / speed) * (Math.PI / 180);
      const pointX = centerX + (radius + 12) * Math.cos(pointAngle);
      const pointY = centerY + (radius + 12) * Math.sin(pointAngle);
      ctx.fillStyle = 'rgba(101, 84, 255, 0.8)';
      ctx.beginPath();
      ctx.arc(pointX, pointY, 3, 0, Math.PI * 2);
      ctx.fill();

      rotation += 0.5;
      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [size, color, speed]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        display: 'block',
        margin: '0 auto',
      }}
    />
  );
}
