'use client';

import { motion } from 'framer-motion';

interface Tab {
  id: string;
  label: string;
}

const tabs: Tab[] = [
  { id: 'explore', label: 'Explore' },
  { id: 'foryou', label: 'For you' },
  { id: 'trending', label: 'Trending' },
  { id: 'wardrobe', label: 'Wardrobe' },
];

interface HomeTabBarProps {
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export function HomeTabBar({ activeTab, onTabChange }: HomeTabBarProps) {
  return (
    <div className="overflow-x-auto hide-scrollbar">
      <div className="flex items-center gap-6 px-4 py-3 min-w-max">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`relative pb-2 text-sm transition-colors ${
                isActive
                  ? 'font-semibold text-phia-black'
                  : 'text-phia-gray-400'
              }`}
            >
              {tab.label}
              {isActive && (
                <motion.div
                  layoutId="homeTabUnderline"
                  className="absolute bottom-0 left-0 right-0 h-[2px] bg-phia-black rounded-full"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default HomeTabBar;
