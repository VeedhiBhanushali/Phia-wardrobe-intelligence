'use client';

import { Search } from 'lucide-react';

interface SearchBarProps {
  value?: string;
  onChange?: (value: string) => void;
  onFocus?: () => void;
  placeholder?: string;
}

export function SearchBar({
  value,
  onChange,
  onFocus,
  placeholder = 'Search',
}: SearchBarProps) {
  return (
    <div className="relative w-full">
      <Search
        size={18}
        className="absolute left-4 top-1/2 -translate-y-1/2 text-phia-gray-400"
        strokeWidth={1.5}
      />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        onFocus={onFocus}
        placeholder={placeholder}
        className="w-full rounded-full border border-phia-gray-200 bg-phia-white py-2.5 pl-11 pr-4 text-sm text-phia-black placeholder:text-phia-gray-400 outline-none focus:border-phia-gray-300 transition-colors"
      />
    </div>
  );
}

export default SearchBar;
