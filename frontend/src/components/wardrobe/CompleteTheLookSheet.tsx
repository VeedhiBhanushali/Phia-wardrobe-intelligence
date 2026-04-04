'use client';

import { X } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { resolveImageUrl } from '@/lib/api';

interface PairItem {
  item_id?: string;
  id?: string;
  title: string;
  image_url: string;
  pair_score?: number;
  slot?: string;
}

interface AnchorItem {
  item_id?: string;
  id?: string;
  title: string;
  image_url: string;
}

interface RecommendationItem {
  title: string;
  brand: string;
  price: number;
  image_url: string;
}

interface CompleteTheLookSheetProps {
  isOpen: boolean;
  onClose: () => void;
  anchorItem: AnchorItem | null;
  pairs: PairItem[];
  recommendation: RecommendationItem | null;
}

export function CompleteTheLookSheet({
  isOpen,
  onClose,
  anchorItem,
  pairs,
  recommendation,
}: CompleteTheLookSheetProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/40 z-40"
            onClick={onClose}
          />

          <motion.div
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 300 }}
            className="fixed bottom-0 left-0 right-0 z-50 bg-phia-white rounded-t-3xl max-h-[85vh] overflow-y-auto pb-[env(safe-area-inset-bottom)]"
          >
            <div className="sticky top-0 bg-phia-white rounded-t-3xl z-10">
              <div className="flex justify-center pt-3 pb-1">
                <div className="w-8 h-1 rounded-full bg-phia-gray-300" />
              </div>
              <div className="flex items-center justify-between px-5 pb-4">
                <h2 className="font-serif text-lg text-phia-black">
                  Complete the look
                </h2>
                <button
                  onClick={onClose}
                  className="p-1.5 rounded-full hover:bg-phia-gray-100 transition-colors"
                >
                  <X size={20} className="text-phia-gray-600" />
                </button>
              </div>
            </div>

            <div className="px-5 space-y-6 pb-6">
              {anchorItem && (
              <div className="flex items-center gap-3">
                <img
                  src={resolveImageUrl(anchorItem.image_url)}
                  alt={anchorItem.title}
                  className="w-14 h-14 rounded-xl object-cover"
                  loading="lazy"
                />
                <p className="text-sm font-medium text-phia-black">
                  {anchorItem.title}
                </p>
              </div>
              )}

              {pairs.length > 0 && (
                <div>
                  <h3 className="text-xs uppercase tracking-wider text-phia-gray-500 mb-3">
                    Pairs with
                  </h3>
                  <div className="flex gap-3 overflow-x-auto scrollbar-hide -mx-5 px-5">
                    {pairs.map((pair) => (
                      <div key={pair.item_id || pair.id} className="flex-shrink-0 w-28">
                        <div className="relative">
                          <img
                            src={resolveImageUrl(pair.image_url)}
                            alt={pair.title}
                            className="w-28 h-36 rounded-xl object-cover"
                            loading="lazy"
                          />
                          <span className="absolute top-1.5 right-1.5 rounded-full bg-phia-green/90 text-white text-[10px] font-medium px-1.5 py-0.5">
                            {Math.round((pair.pair_score ?? 0) * 100)}%
                          </span>
                        </div>
                        <p className="text-xs text-phia-gray-700 mt-1.5 line-clamp-2">
                          {pair.title}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {recommendation && (
              <div>
                <h3 className="text-xs uppercase tracking-wider text-phia-gray-500 mb-3">
                  Recommended addition
                </h3>
                <div className="flex items-center gap-3 bg-phia-gray-50 rounded-xl p-3">
                  <img
                    src={resolveImageUrl(recommendation.image_url)}
                    alt={recommendation.title}
                    className="w-20 h-24 rounded-lg object-cover"
                    loading="lazy"
                  />
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-phia-black truncate">
                      {recommendation.title}
                    </p>
                    <p className="text-[11px] uppercase tracking-wider text-phia-gray-500 mt-0.5">
                      {recommendation.brand}
                    </p>
                    <p className="text-sm font-medium text-phia-black mt-1">
                      ${(recommendation.price ?? 0).toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

export default CompleteTheLookSheet;
