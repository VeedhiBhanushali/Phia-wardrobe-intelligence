'use client';

import { Bookmark, BookmarkCheck, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { resolveImageUrl } from '@/lib/api';

interface GapItem {
  item_id?: string;
  title: string;
  brand: string;
  price: number;
  image_url: string;
  slot: string;
}

interface GapRecommendationCardProps {
  item: GapItem;
  unlockCount: number;
  tasteFit: number;
  explanation: string;
  onSave: () => void;
  onBrowseSimilar: () => void;
  isSaved: boolean;
  onDismiss?: () => void;
}

export function GapRecommendationCard({
  item,
  unlockCount,
  tasteFit,
  explanation,
  onSave,
  onBrowseSimilar,
  isSaved,
  onDismiss,
}: GapRecommendationCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="bg-phia-gray-50 rounded-2xl p-4"
    >
      <div className="overflow-hidden rounded-2xl">
        <img
          src={resolveImageUrl(item.image_url)}
          alt={item.title}
          className="w-full aspect-[4/5] object-cover"
          loading="lazy"
        />
      </div>

      <div className="mt-4 space-y-3">
        <p className="font-serif italic text-phia-gray-700 text-sm leading-relaxed">
          {explanation}
        </p>

        <div>
          <h3 className="text-base font-medium text-phia-black">{item.title}</h3>
          <p className="text-[11px] uppercase tracking-wider text-phia-gray-500 mt-0.5">
            {item.brand}
          </p>
          <p className="text-sm font-medium text-phia-black mt-1">
            ${(item.price ?? 0).toLocaleString()}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-1">
          <button
            onClick={onSave}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-full border text-sm font-medium transition-colors ${
              isSaved
                ? 'border-phia-black bg-phia-black text-phia-white'
                : 'border-phia-gray-300 text-phia-black hover:border-phia-black'
            }`}
          >
            {isSaved ? <BookmarkCheck size={16} /> : <Bookmark size={16} />}
            {isSaved ? 'Saved' : 'Save'}
          </button>

          <button
            onClick={onBrowseSimilar}
            className="flex items-center gap-1 text-sm text-phia-gray-600 hover:text-phia-black transition-colors"
          >
            Browse similar
            <ArrowRight size={14} />
          </button>

          {onDismiss && (
            <button
              type="button"
              onClick={onDismiss}
              className="text-sm text-phia-gray-400 hover:text-phia-black transition-colors"
            >
              Not for me
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export default GapRecommendationCard;
