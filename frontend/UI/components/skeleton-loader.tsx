'use client';

interface SkeletonLoaderProps {
  count?: number;
  height?: string;
  className?: string;
}

export function SkeletonLoader({
  count = 1,
  height = 'h-12',
  className = '',
}: SkeletonLoaderProps) {
  return (
    <div className={`space-y-3 ${className}`}>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className={`skeleton ${height} rounded-lg w-full`}
        />
      ))}
    </div>
  );
}
