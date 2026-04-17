"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, Bookmark, MessageCircle } from "lucide-react";
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
  intentVector?: number[] | null;
  intentConfidence?: number;
  onAskPhia?: (itemId: string) => void;
  onViewItem?: (view: { item_id: string; embedding: number[] }) => void;
}

interface EvalResult {
  taste_fit: number;
  intent_match: number | null;
  purchase_confidence: string;
  unlock_count: number;
  pairs_with: WardrobeItem[];
  explanation: string;
  best_price: number;
}

const CONFIDENCE_COLORS: Record<string, string> = {
  HIGH: "text-green-600 bg-green-50",
  MEDIUM: "text-amber-600 bg-amber-50",
  LOW: "text-phia-gray-500 bg-phia-gray-50",
};

export function ProductDetail({
  itemId,
  tasteProfile,
  wardrobeItems,
  isSaved,
  onBack,
  onSave,
  onRemove,
  intentVector,
  intentConfidence,
  onAskPhia,
  onViewItem,
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
        const catalogItem = (await api.catalog.getItem(itemId, true)) as WardrobeItem & { embedding?: number[] };
        const embedding = catalogItem.embedding;
        const cleanItem = { ...catalogItem } as Record<string, unknown>;
        delete cleanItem.embedding;
        setItem(cleanItem as WardrobeItem);

        if (embedding && onViewItem) {
          onViewItem({ item_id: itemId, embedding });
        }

        log("click", "product_detail", itemId);

        if (tasteProfile) {
          try {
            const evalResult = (await api.recommendations.evaluateItemV2({
              item_id: itemId,
              user_id: tasteProfile.user_id,
              wardrobe_item_ids: wardrobeItems.map((i) => i.item_id),
              taste_vector: tasteProfile.taste_vector,
              intent_vector: intentVector,
              intent_confidence: intentConfidence ?? 0,
            })) as EvalResult;
            setEvaluation(evalResult);

            log("impression", "product_detail", itemId, {
              score: evalResult.taste_fit,
              unlock_count: evalResult.unlock_count,
              taste_score: evalResult.taste_fit,
            });
          } catch {
            // Fall back to v1 endpoint
            const evalResult = (await api.recommendations.evaluateItem({
              item_id: itemId,
              user_id: tasteProfile.user_id,
              wardrobe_item_ids: wardrobeItems.map((i) => i.item_id),
              taste_vector: tasteProfile.taste_vector,
            })) as {
              taste_fit: number;
              unlock_count: number;
              pairs_with: WardrobeItem[];
              explanation: string;
              confidence: number;
            };
            setEvaluation({
              ...evalResult,
              intent_match: null,
              purchase_confidence:
                evalResult.confidence >= 0.5
                  ? "HIGH"
                  : evalResult.confidence >= 0.3
                  ? "MEDIUM"
                  : "LOW",
              best_price: catalogItem.price,
            });
          }
        }
      } catch {
        setError("Couldn't load product details");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [itemId, tasteProfile, wardrobeItems, intentVector, intentConfidence, log, onViewItem]);

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
            {/* Score cards grid */}
            <div className="grid grid-cols-3 gap-2 mb-3">
              {/* Taste Fit */}
              <div className="rounded-xl border border-phia-gray-200 p-3 text-center">
                <span className="text-2xl font-semibold text-phia-orange tabular-nums">
                  {Math.round(evaluation.taste_fit * 100)}%
                </span>
                <p className="text-[10px] font-medium text-phia-black mt-0.5">
                  {evaluation.taste_fit >= 0.80 ? "Strong fit" : evaluation.taste_fit >= 0.60 ? "Good fit" : "Fair fit"}
                </p>
              </div>

              {/* Purchase Confidence */}
              <div className="rounded-xl border border-phia-gray-200 p-3 text-center">
                <span
                  className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${
                    CONFIDENCE_COLORS[evaluation.purchase_confidence] ||
                    CONFIDENCE_COLORS.LOW
                  }`}
                >
                  {evaluation.purchase_confidence}
                </span>
                <p className="text-[10px] font-medium text-phia-black mt-1">
                  Confidence
                </p>
              </div>

              {/* Outfit Unlocks */}
              <div className="rounded-xl border border-phia-gray-200 p-3 text-center">
                <span className="text-2xl font-semibold text-phia-green tabular-nums">
                  +{evaluation.unlock_count}
                </span>
                <p className="text-[10px] font-medium text-phia-black mt-0.5">
                  Outfits
                </p>
              </div>
            </div>

            {/* Intent Match — only when intent confidence > 0.3 */}
            {evaluation.intent_match !== null &&
              intentConfidence !== undefined &&
              intentConfidence > 0.3 && (
                <div className="rounded-xl border border-phia-blue/20 bg-phia-blue/5 p-3 mb-3 flex items-center justify-between">
                  <div>
                    <p className="text-xs font-medium text-phia-black">
                      {evaluation.intent_match >= 0.80 ? "Strong" : evaluation.intent_match >= 0.60 ? "Good" : "Fair"} intent match
                    </p>
                    <p className="text-[10px] text-phia-gray-500">
                      Based on your current browsing
                    </p>
                  </div>
                  <span className="text-lg font-semibold text-phia-blue tabular-nums">
                    {Math.round(evaluation.intent_match * 100)}%
                  </span>
                </div>
              )}

            {/* Best price */}
            {evaluation.best_price > 0 && (
              <div className="rounded-xl bg-phia-gray-50 p-3 mb-3 flex items-center justify-between">
                <p className="text-xs text-phia-gray-600">Best price</p>
                <span className="text-sm font-semibold text-phia-black">
                  ${evaluation.best_price}
                </span>
              </div>
            )}

            {/* Pairs with */}
            {evaluation.pairs_with.length > 0 && (
              <div className="mt-3">
                <p className="text-xs font-medium text-phia-gray-900 mb-2">
                  Pairs with {evaluation.pairs_with.length} of your saves
                </p>
                <div className="flex gap-2 overflow-x-auto hide-scrollbar pb-1">
                  {evaluation.pairs_with.map((pair) => (
                    <div key={pair.item_id} className="w-16 shrink-0">
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

            {/* AI insight */}
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

        {/* Ask Phia about this */}
        {onAskPhia && (
          <button
            onClick={() => onAskPhia(item.item_id)}
            className="w-full mt-3 rounded-full border border-phia-gray-200 py-3 text-sm font-medium text-phia-black flex items-center justify-center gap-2"
          >
            <MessageCircle size={14} />
            Ask Phia about this
          </button>
        )}
      </div>
    </motion.div>
  );
}

export default ProductDetail;
