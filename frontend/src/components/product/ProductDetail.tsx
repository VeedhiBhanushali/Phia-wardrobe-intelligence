"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, Bookmark } from "lucide-react";
import { ScoreDisplay } from "@/components/wardrobe/ScoreDisplay";
import { ProductDetailSkeleton } from "@/components/ui/Skeleton";
import { InlineError } from "@/components/ui/ErrorBoundary";
import { api, resolveImageUrl } from "@/lib/api";
import { useEventLog } from "@/lib/useEventLog";
import type { TasteProfile, WardrobeItem } from "@/lib/store";

interface ProductDetailProps {
  itemId: string;
  tasteProfile: TasteProfile | null;
  wardrobeItems: WardrobeItem[];
  isSaved: boolean;
  onBack: () => void;
  onSave: (item: WardrobeItem) => void;
  onRemove: (itemId: string) => void;
}

interface EvalResult {
  taste_fit: number;
  unlock_count: number;
  pairs_with: WardrobeItem[];
  explanation: string;
  confidence: number;
}

export function ProductDetail({
  itemId,
  tasteProfile,
  wardrobeItems,
  isSaved,
  onBack,
  onSave,
  onRemove,
}: ProductDetailProps) {
  const [item, setItem] = useState<WardrobeItem | null>(null);
  const [evaluation, setEvaluation] = useState<EvalResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { log } = useEventLog(tasteProfile?.user_id || "");

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const catalogItem = (await api.catalog.getItem(itemId)) as WardrobeItem;
        setItem(catalogItem);

        log("click", "product_detail", itemId);

        if (tasteProfile) {
          const evalResult = (await api.recommendations.evaluateItem({
            item_id: itemId,
            user_id: tasteProfile.user_id,
            wardrobe_item_ids: wardrobeItems.map((i) => i.item_id),
            taste_vector: tasteProfile.taste_vector,
          })) as EvalResult;
          setEvaluation(evalResult);

          log("impression", "product_detail", itemId, {
            score: evalResult.confidence,
            unlock_count: evalResult.unlock_count,
            taste_score: evalResult.taste_fit,
          });
        }
      } catch {
        setError("Couldn't load product details");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [itemId, tasteProfile, wardrobeItems, log]);

  const handleBookmark = () => {
    if (!item) return;
    if (isSaved) {
      onRemove(item.item_id);
    } else {
      log("save", "product_detail", item.item_id, {
        taste_score: evaluation?.taste_fit,
        unlock_count: evaluation?.unlock_count,
      });
      onSave(item);
    }
  };

  if (loading) {
    return (
      <div>
        <button
          onClick={onBack}
          className="absolute top-3 left-3 z-10 w-9 h-9 rounded-full bg-white/90 flex items-center justify-center shadow-sm"
        >
          <ArrowLeft size={18} className="text-phia-black" />
        </button>
        <ProductDetailSkeleton />
      </div>
    );
  }

  if (error || !item) {
    return (
      <div className="pt-14 px-4">
        <button
          onClick={onBack}
          className="mb-4 w-9 h-9 rounded-full bg-phia-gray-50 flex items-center justify-center"
        >
          <ArrowLeft size={18} className="text-phia-black" />
        </button>
        <InlineError
          message={error || "Product not found"}
          onRetry={() => window.location.reload()}
        />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.2 }}
    >
      <div className="relative">
        <button
          onClick={onBack}
          className="absolute top-3 left-3 z-10 w-9 h-9 rounded-full bg-white/90 flex items-center justify-center shadow-sm"
        >
          <ArrowLeft size={18} className="text-phia-black" />
        </button>

        <div className="aspect-[3/4] bg-phia-gray-100 rounded-b-3xl overflow-hidden">
          <img
            src={resolveImageUrl(item.image_url)}
            alt={item.title}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        </div>
      </div>

      <div className="px-4 py-4">
        <p className="text-[10px] uppercase tracking-[0.15em] text-phia-gray-400 mb-1">
          {item.brand}
        </p>
        <h1 className="text-lg font-medium text-phia-black mb-1">
          {item.title}
        </h1>
        <p className="text-lg font-semibold text-phia-black">
          ${item.price}
        </p>

        {evaluation && (
          <div className="mt-4">
            <ScoreDisplay
              tasteFit={evaluation.taste_fit}
              unlockCount={evaluation.unlock_count}
              bestPrice={item.price}
            />

            {evaluation.pairs_with.length > 0 && (
              <div className="mt-4">
                <p className="text-xs font-medium text-phia-gray-900 mb-2">
                  Pairs with
                </p>
                <div className="flex gap-2 overflow-x-auto hide-scrollbar pb-1">
                  {evaluation.pairs_with.map((pair) => (
                    <div
                      key={pair.item_id}
                      className="w-16 shrink-0"
                    >
                      <div className="aspect-square rounded-lg bg-phia-gray-100 overflow-hidden">
                        <img
                          src={resolveImageUrl(pair.image_url)}
                          alt={pair.title}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                      </div>
                      <p className="text-[9px] text-phia-gray-400 mt-0.5 truncate">
                        {pair.title}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-3 rounded-xl bg-phia-gray-50 px-4 py-3">
              <p className="text-sm text-phia-gray-600 italic font-serif">
                &ldquo;{evaluation.explanation}&rdquo;
              </p>
            </div>
          </div>
        )}

        <div className="flex gap-3 mt-6">
          <button
            onClick={handleBookmark}
            className={`flex-1 rounded-full py-3.5 text-sm font-medium flex items-center justify-center gap-2 transition-colors ${
              isSaved
                ? "bg-phia-blue text-white"
                : "bg-phia-black text-white"
            }`}
          >
            <Bookmark
              size={16}
              fill={isSaved ? "currentColor" : "none"}
            />
            {isSaved ? "Saved" : "Save to wardrobe"}
          </button>
        </div>
      </div>
    </motion.div>
  );
}

export default ProductDetail;
