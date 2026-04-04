'use client';

import { Bell, Search } from 'lucide-react';

interface PhiaHeaderProps {
  onNotificationPress?: () => void;
  onSearchPress?: () => void;
}

export function PhiaHeader({ onNotificationPress, onSearchPress }: PhiaHeaderProps) {
  return (
    <header className="fixed top-0 left-1/2 -translate-x-1/2 w-full max-w-[430px] bg-phia-white z-50">
      <div className="flex items-center justify-between px-4 h-14">
        <span className="font-serif font-bold italic text-2xl text-phia-black">
          phia
        </span>
        <div className="flex items-center gap-4">
          <button onClick={onNotificationPress} className="text-phia-black">
            <Bell size={22} strokeWidth={1.5} />
          </button>
          <button onClick={onSearchPress} className="text-phia-black">
            <Search size={22} strokeWidth={1.5} />
          </button>
        </div>
      </div>
    </header>
  );
}

export default PhiaHeader;
