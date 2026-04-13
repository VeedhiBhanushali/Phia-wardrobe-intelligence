"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";

const DEMO_PROFILES = [
  { id: "minimalist", label: "Minimalist" },
  { id: "streetwear", label: "Streetwear" },
  { id: "smart_casual", label: "Smart Casual" },
  { id: "cold_start", label: "Cold Start" },
];

interface DemoProfileSelectorProps {
  onSelect: (profileId: string) => void;
}

export function DemoProfileSelector({ onSelect }: DemoProfileSelectorProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[10px] text-phia-gray-400 hover:text-phia-black transition-colors"
      >
        Demo
        <ChevronDown size={10} />
      </button>
      {open && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setOpen(false)}
          />
          <div className="absolute right-0 top-full mt-1 z-50 bg-white rounded-lg shadow-lg border border-phia-gray-200 py-1 min-w-[140px]">
            {DEMO_PROFILES.map((profile) => (
              <button
                key={profile.id}
                onClick={() => {
                  onSelect(profile.id);
                  setOpen(false);
                }}
                className="block w-full text-left px-3 py-1.5 text-xs text-phia-black hover:bg-phia-gray-50 transition-colors"
              >
                {profile.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default DemoProfileSelector;
