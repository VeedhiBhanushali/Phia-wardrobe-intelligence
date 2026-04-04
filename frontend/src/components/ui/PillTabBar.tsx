'use client';

import { motion } from 'framer-motion';

interface Tab {
  id: string;
  label: string;
}

interface PillTabBarProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export function PillTabBar({ tabs, activeTab, onTabChange }: PillTabBarProps) {
  return (
    <div className="overflow-x-auto hide-scrollbar">
      <div className="flex items-center gap-2 px-4 min-w-max">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <motion.button
              key={tab.id}
              whileTap={{ scale: 0.95 }}
              onClick={() => onTabChange(tab.id)}
              className={`rounded-full px-4 py-1.5 text-sm transition-colors ${
                isActive
                  ? 'bg-phia-black text-white'
                  : 'bg-transparent border border-phia-gray-200 text-phia-gray-400'
              }`}
            >
              {tab.label}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}

export default PillTabBar;
