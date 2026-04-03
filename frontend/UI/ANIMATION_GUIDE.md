# Animation Quick Reference Guide

## Using Animations in ExoHabitAI

### Commonly Used Animation Classes

#### Page/Container Animations
```tsx
// Fade in with upward slide
<div className="animate-fade-in-up">...</div>

// Scale in (perfect for modals/cards on trigger)
<div className="animate-scale-in">...</div>

// Fade in only
<div className="animate-fade-in">...</div>

// Navbar slide in from top
<nav className="animate-slide-in-top">...</nav>
```

#### Button Animations
```tsx
// Automatic hover/click effects
<Button className="button-hover-glow">
  Click me
</Button>

// Effects: hover scales to 1.05, active scales to 0.95
// Smooth 200ms transitions with shadow effects
```

#### Card/Container Animations
```tsx
// Lift on hover effect
<Card className="card-hover-lift">
  Content
</Card>

// Effects: hover lifts 4px up (-translate-y-1) with shadow
// Smooth 200ms transitions
```

#### Input Field Animations
```tsx
// Focus glow effect
<input className="input-glow" />

// Effects: focus state gets accent border + shadow-lg/20 opacity
```

#### Icon Animations
```tsx
// Glowing pulse effect
<div className="animate-glow-pulse">
  <Icon />
</div>

// Perfect for accent icons, logo elements
```

#### Loading States
```tsx
// Shimmer skeleton loader
<SkeletonLoader count={3} height="h-12" />

// Or manual skeleton with shimmer class
<div className="skeleton h-12 w-full rounded-lg" />
```

### Staggering Animations

```tsx
{items.map((item, index) => (
  <div
    key={item.id}
    className="animate-fade-in-up"
    style={{ animationDelay: `${index * 50}ms` }}
  >
    {item.content}
  </div>
))}
```

### Animated Counters

```tsx
import { AnimatedCounter } from '@/components/animated-counter';

// Basic usage
<AnimatedCounter value={42} />

// With options
<AnimatedCounter 
  value={100} 
  duration={1500}
  decimals={2}
  suffix="%" 
/>
```

### Animation Durations

- **Instant**: No delay (0ms)
- **Fast**: 200ms (button hovers, color transitions)
- **Medium**: 300-400ms (page/element transitions)
- **Slow**: 700-1000ms (progress bars, fills)

### Standard Delays (for staggering)

```
First item:    0ms
Second item:   50ms
Third item:    100ms
Fourth item:   150ms
...and so on
```

### Progress Bar Animation

```tsx
<div className="w-full bg-muted rounded-full h-2 overflow-hidden">
  <div
    className="bg-accent h-2 rounded-full transition-all duration-1000"
    style={{ width: `${percentage}%` }}
  />
</div>
```

### Combining Animations

```tsx
// Scale in on load, then hover lift
<Card className="animate-scale-in card-hover-lift">
  Content
</Card>

// Fade in, then button hover on click
<Button className="animate-fade-in button-hover-glow">
  Action
</Button>
```

## Performance Tips

1. **Use transform and opacity** for smoothest animations
2. **Keep durations short** (200-500ms) for responsiveness
3. **Stagger animations** by 50ms increments to prevent repaints
4. **Avoid animating** layout properties (width, height, etc.)
5. **GPU acceleration** happens automatically with transform/opacity

## Animation Properties

All animations use:
- ✅ Hardware-accelerated transforms
- ✅ Cubic-bezier easing for smoothness
- ✅ Will-change hints (handled by CSS)
- ✅ Optimal performance on mobile and desktop

## CSS Variables Available

```css
--duration-200: 200ms
--duration-300: 300ms
--duration-500: 500ms
```

## Responsive Behavior

All animations work seamlessly on:
- ✅ Mobile (320px+)
- ✅ Tablet (768px+)
- ✅ Desktop (1024px+)
- ✅ Touch devices (animations don't cause lag)

## Browser Support

- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Mobile browsers (iOS Safari, Chrome Android)

No vendor prefixes needed - all animations use standard CSS!
