'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

interface Attribute {
  label: string;
  confidence: number;
  scores?: Record<string, number>;
}

interface AestheticProfileCardProps {
  attributes: Record<string, Attribute>;
  trendFingerprint?: Record<string, number>;
}

const DISPLAY_ORDER = ['silhouette', 'color_story', 'formality', 'occasion'];

const DISPLAY_LABELS: Record<string, string> = {
  silhouette: 'Silhouette',
  color_story: 'Color Story',
  formality: 'Formality',
  occasion: 'Occasion',
};

export function AestheticProfileCard({
  attributes,
  trendFingerprint,
}: AestheticProfileCardProps) {
  const [expanded, setExpanded] = useState(false);

  const topTrends = trendFingerprint
    ? Object.entries(trendFingerprint).slice(0, 3)
    : [];

  return (
    <div className="space-y-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full"
      >
        <h3 className="font-serif text-lg text-phia-black">Your aesthetic</h3>
        {expanded ? (
          <ChevronUp size={18} className="text-phia-gray-500" />
        ) : (
          <ChevronDown size={18} className="text-phia-gray-500" />
        )}
      </button>

      <div className="flex flex-wrap gap-2">
        {topTrends.length > 0
          ? topTrends.map(([name]) => (
              <span
                key={name}
                className="rounded-full bg-phia-black text-white text-xs px-3 py-1 font-medium"
              >
                {name}
              </span>
            ))
          : DISPLAY_ORDER.filter((key) => attributes[key]).map((key) => (
              <span
                key={key}
                className="rounded-full bg-phia-gray-100 text-phia-gray-900 text-xs px-3 py-1 font-medium"
              >
                {attributes[key].label}
              </span>
            ))}
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="space-y-2 pt-2">
              {topTrends.length > 0 && (
                <>
                  <p className="text-[10px] uppercase tracking-[0.15em] text-phia-gray-400 mb-1">
                    Style DNA
                  </p>
                  {Object.entries(trendFingerprint!).slice(0, 6).map(([name, score]) => {
                    const pct = Math.round(score * 100);
                    return (
                      <div key={name} className="flex items-center justify-between">
                        <p className="text-sm font-medium text-phia-black">{name}</p>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 rounded-full bg-phia-gray-200 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-phia-black"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                          <span className="text-[11px] text-phia-gray-500 tabular-nums w-8 text-right">
                            {pct}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                  <div className="h-px bg-phia-gray-200 my-2" />
                </>
              )}
              {DISPLAY_ORDER.filter((key) => attributes[key]).map((key) => {
                const attr = attributes[key];
                const pct = Math.round(attr.confidence * 100);
                return (
                  <div key={key} className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-phia-gray-500">
                        {DISPLAY_LABELS[key] || key}
                      </p>
                      <p className="text-sm font-medium text-phia-black">
                        {attr.label}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 rounded-full bg-phia-gray-200 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-phia-black"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-[11px] text-phia-gray-500 tabular-nums w-8 text-right">
                        {pct}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default AestheticProfileCard;
