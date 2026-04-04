'use client';

import { Home, Search, Bookmark, User } from 'lucide-react';
import { motion } from 'framer-motion';

type Tab = 'home' | 'search' | 'saved' | 'profile';

interface BottomNavProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

const tabs: { id: Tab; icon: typeof Home }[] = [
  { id: 'home', icon: Home },
  { id: 'search', icon: Search },
  { id: 'saved', icon: Bookmark },
  { id: 'profile', icon: User },
];

export function BottomNav({ activeTab, onTabChange }: BottomNavProps) {
  return (
    <nav className="fixed bottom-0 left-1/2 -translate-x-1/2 w-full max-w-[430px] bg-phia-white border-t border-phia-gray-200 z-50 pb-[env(safe-area-inset-bottom)]">
      <div className="flex items-center justify-around h-14">
        {tabs.map(({ id, icon: Icon }) => {
          const isActive = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className="relative flex items-center justify-center w-12 h-12"
            >
              <Icon
                size={24}
                className={isActive ? 'text-phia-black' : 'text-phia-gray-400'}
                strokeWidth={isActive ? 2.5 : 1.5}
                fill={isActive ? 'currentColor' : 'none'}
              />
              {isActive && (
                <motion.div
                  layoutId="bottomNavIndicator"
                  className="absolute -top-px left-2 right-2 h-0.5 bg-phia-black rounded-full"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
}

export default BottomNav;
