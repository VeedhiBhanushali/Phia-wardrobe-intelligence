'use client';

import { useState, useRef, useCallback, type ReactNode } from 'react';

interface CarouselProps {
  children: ReactNode[];
  showDots?: boolean;
}

export function Carousel({ children, showDots = true }: CarouselProps) {
  const [activeIndex, setActiveIndex] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  const handleScroll = useCallback(() => {
    const container = scrollRef.current;
    if (!container) return;
    const scrollLeft = container.scrollLeft;
    const childWidth = container.offsetWidth;
    const index = Math.round(scrollLeft / childWidth);
    setActiveIndex(index);
  }, []);

  return (
    <div className="w-full">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex overflow-x-auto snap-x snap-mandatory hide-scrollbar"
      >
        {children.map((child, i) => (
          <div key={i} className="w-full flex-shrink-0 snap-start">
            {child}
          </div>
        ))}
      </div>

      {showDots && children.length > 1 && (
        <div className="flex items-center justify-center gap-1.5 mt-3">
          {children.map((_, i) => (
            <span
              key={i}
              className={`w-1.5 h-1.5 rounded-full transition-colors ${
                i === activeIndex ? 'bg-phia-black' : 'bg-phia-gray-300'
              }`}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default Carousel;
