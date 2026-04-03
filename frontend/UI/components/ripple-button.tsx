'use client';

import React, { useRef, useEffect } from 'react';

interface RippleButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  rippleColor?: string;
}

export function RippleButton({
  children,
  rippleColor = 'rgba(101, 84, 255, 0.5)',
  ...props
}: RippleButtonProps) {
  const buttonRef = useRef<HTMLButtonElement>(null);

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    const button = buttonRef.current;
    if (!button) return;

    const rect = button.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const ripple = document.createElement('span');
    ripple.style.position = 'absolute';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.style.width = '5px';
    ripple.style.height = '5px';
    ripple.style.backgroundColor = rippleColor;
    ripple.style.borderRadius = '100%';
    ripple.style.pointerEvents = 'none';
    ripple.style.opacity = '1';
    ripple.style.transform = 'scale(1)';
    ripple.style.animation = `ripple-animation 0.6s ease-out forwards`;

    button.appendChild(ripple);

    setTimeout(() => ripple.remove(), 600);

    if (props.onClick) {
      props.onClick(e);
    }
  };

  return (
    <button
      ref={buttonRef}
      {...props}
      onClick={handleClick}
      style={{
        position: 'relative',
        overflow: 'hidden',
        ...props.style,
      }}
    >
      {children}
    </button>
  );
}
