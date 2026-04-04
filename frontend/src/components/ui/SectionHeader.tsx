'use client';

import { ChevronRight } from 'lucide-react';

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  onSeeAll?: () => void;
}

export function SectionHeader({ title, subtitle, onSeeAll }: SectionHeaderProps) {
  return (
    <div className="flex items-start justify-between">
      <div>
        <h2 className="font-serif text-xl text-phia-black">{title}</h2>
        {subtitle && (
          <p className="text-sm text-phia-gray-400 mt-0.5">{subtitle}</p>
        )}
      </div>
      {onSeeAll && (
        <button
          onClick={onSeeAll}
          className="flex items-center gap-0.5 text-sm text-phia-gray-400 mt-1"
        >
          <ChevronRight size={18} />
        </button>
      )}
    </div>
  );
}

export default SectionHeader;
