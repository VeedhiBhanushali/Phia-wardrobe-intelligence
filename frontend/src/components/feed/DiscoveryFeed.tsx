"use client";

import { useState, useEffect, useCallback } from "react";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ProductCard } from "@/components/ui/ProductCard";
import { ProductGridSkeleton } from "@/components/ui/Skeleton";
import { resolveImageUrl } from "@/lib/api";
import { api } from "@/lib/api";
import type {
  TasteProfile,
  WardrobeItem,
  FeedData,
} from "@/lib/store";

interface DiscoveryFeedProps {
  tasteProfile: TasteProfile | null;
  wardrobeItems: WardrobeItem[];
  skippedItemIds: string[];
  intentVector: number[] | null;
  intentConfidence: number;
  isItemSaved: (itemId: string) => boolean;
  onBookmark: (item: WardrobeItem) => void;
  onProductTap: (itemId: string) => void;
  onDismiss: (itemId: string) => void;
}

export function DiscoveryFeed({
  tasteProfile,
  wardrobeItems,
  skippedItemIds,
  intentVector,
  intentConfidence,
  isItemSaved,
  onBookmark,
  onProductTap,
  onDismiss,
}: DiscoveryFeedProps) {
  const [feed, setFeed] = useState<FeedData | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchFeed = useCallback(async () => {
    if (!tasteProfile?.taste_vector?.length) return;
    setLoading(true);
    try {
      const res = (await api.recommendations.feed({
        user_id: tasteProfile.user_id,
        wardrobe_item_ids: wardrobeItems.map((i) => i.item_id),
        taste_vector: tasteProfile.taste_vector,
        taste_modes: tasteProfile.taste_modes,
        occasion_vectors: tasteProfile.occasion_vectors,
        trend_fingerprint: tasteProfile.trend_fingerprint,
        anti_taste_vector: tasteProfile.anti_taste_vector,
        price_tier: tasteProfile.price_tier
          ? [tasteProfile.price_tier[0], tasteProfile.price_tier[1]]
          : undefined,
        aesthetic_label:
          tasteProfile.aesthetic_attributes?.silhouette?.label ?? "",
        skipped_item_ids: skippedItemIds,
        intent_vector: intentVector,
        intent_confidence: intentConfidence,
        style_attributes: tasteProfile.style_attributes ?? {},
      })) as FeedData;
      setFeed(res);
    } catch {
      // non-critical
    } finally {
      setLoading(false);
    }
  }, [tasteProfile, wardrobeItems, skippedItemIds, intentVector, intentConfidence]);

  useEffect(() => {
    fetchFeed();
  }, [fetchFeed]);

  if (!tasteProfile) {
    return (
      <div className="px-4 py-8 text-center">
        <p className="text-sm text-phia-gray-400">
          Set up your taste profile to see personalized recommendations
        </p>
      </div>
    );
  }

  if (loading && !feed) {
    return (
      <div className="px-4 py-4">
        <ProductGridSkeleton count={4} />
      </div>
    );
  }

  if (!feed) return null;

  const renderedIds = new Set<string>();
  const dedup = <T extends { item: { item_id: string } }>(items: T[]): T[] =>
    items.filter((p) => {
      if (renderedIds.has(p.item.item_id)) return false;
      renderedIds.add(p.item.item_id);
      return true;
    });

  const closetItems = dedup(feed.completeYourCloset);
  const aestheticItems = dedup(feed.yourAesthetic);

  for (const o of feed.completeYourOutfits) {
    if (o.catalog_addition) renderedIds.add(o.catalog_addition.item_id);
  }

  return (
    <div className="pb-8">
      {/* Complete Your Closet */}
      {closetItems.length > 0 && (
        <div className="mt-4">
          <div className="px-4">
            <SectionHeader
              title="Complete your closet"
              subtitle="Gap-targeted picks"
            />
          </div>
          <div className="flex gap-3 overflow-x-auto hide-scrollbar px-4 mt-2 pb-2">
            {closetItems.map((pick) => (
              <div key={pick.item.item_id} className="w-[160px] shrink-0">
                <ProductCard
                  item={pick.item}
                  isSaved={isItemSaved(pick.item.item_id)}
                  tasteFit={pick.taste_score}
                  unlockCount={pick.unlock_count}
                  onPress={() => onProductTap(pick.item.item_id)}
                  onBookmark={() => onBookmark(pick.item)}
                  onDismiss={() => onDismiss(pick.item.item_id)}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Your Aesthetic */}
      {aestheticItems.length > 0 && (
        <div className="mt-6">
          <div className="px-4">
            <SectionHeader
              title="Your aesthetic"
              subtitle="Pure taste match"
            />
          </div>
          <div className="flex gap-3 overflow-x-auto hide-scrollbar px-4 mt-2 pb-2">
            {aestheticItems.map((pick) => (
              <div key={pick.item.item_id} className="w-[160px] shrink-0">
                <ProductCard
                  item={pick.item}
                  isSaved={isItemSaved(pick.item.item_id)}
                  tasteFit={pick.taste_score}
                  onPress={() => onProductTap(pick.item.item_id)}
                  onBookmark={() => onBookmark(pick.item)}
                  onDismiss={() => onDismiss(pick.item.item_id)}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Complete Your Outfits */}
      {feed.completeYourOutfits.length > 0 && (
        <div className="px-4 mt-6">
          <SectionHeader
            title="Complete your outfits"
            subtitle="Built from your saves"
          />
          <div className="space-y-3 mt-2">
            {feed.completeYourOutfits.map((outfit, i) => (
              <div
                key={i}
                className="rounded-2xl border border-phia-gray-100 p-3"
              >
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-medium text-phia-black">
                    {outfit.title}
                  </p>
                  {outfit.harmony_score > 0 && (
                    <span className="text-[10px] text-phia-gray-400">
                      {outfit.harmony_score >= 0.80 ? "Strong" : outfit.harmony_score >= 0.60 ? "Good" : "Fair"} harmony{" "}
                      <span className="opacity-60">{Math.round(outfit.harmony_score * 100)}%</span>
                    </span>
                  )}
                </div>
                <div className="flex gap-2 overflow-x-auto hide-scrollbar">
                  {outfit.wardrobe_items.map((item) => (
                    <button
                      key={item.item_id}
                      type="button"
                      onClick={() => onProductTap(item.item_id)}
                      className="w-16 shrink-0"
                    >
                      <div className="aspect-[3/4] rounded-lg bg-phia-gray-100 overflow-hidden">
                        <img
                          src={resolveImageUrl(item.image_url)}
                          alt={item.title}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                      </div>
                      <p className="text-[9px] text-phia-gray-400 mt-0.5 truncate">
                        {item.title}
                      </p>
                    </button>
                  ))}
                  {outfit.catalog_addition && (
                    <button
                      type="button"
                      onClick={() =>
                        onProductTap(outfit.catalog_addition!.item_id)
                      }
                      className="w-16 shrink-0"
                    >
                      <div className="aspect-[3/4] rounded-lg bg-phia-blue/10 border-2 border-dashed border-phia-blue/30 overflow-hidden relative">
                        <img
                          src={resolveImageUrl(
                            outfit.catalog_addition.image_url
                          )}
                          alt={outfit.catalog_addition.title}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                        <div className="absolute bottom-0 inset-x-0 bg-phia-blue text-white text-[8px] text-center py-0.5">
                          ADD
                        </div>
                      </div>
                      <p className="text-[9px] text-phia-blue mt-0.5 truncate">
                        {outfit.catalog_addition.title}
                      </p>
                    </button>
                  )}
                </div>
                <p className="text-[11px] text-phia-gray-500 mt-2 italic">
                  {outfit.rationale}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Best Prices on Saves */}
      {feed.bestPricesOnSaves.length > 0 && (
        <div className="mt-6">
          <div className="px-4">
            <SectionHeader
              title="Your saves in budget"
              subtitle="Items in your price range"
            />
          </div>
          <div className="flex gap-3 overflow-x-auto hide-scrollbar px-4 mt-2 pb-2">
            {feed.bestPricesOnSaves.map((entry) => (
              <div key={entry.item.item_id} className="w-[140px] shrink-0">
                <ProductCard
                  item={entry.item}
                  isSaved={true}
                  onPress={() => onProductTap(entry.item.item_id)}
                  onBookmark={() => onBookmark(entry.item)}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Occasion Rows */}
      {feed.occasionRows.map((section) => {
        const picks = section.items.filter(
          (p: { item: { item_id: string } }) => {
            if (renderedIds.has(p.item.item_id)) return false;
            renderedIds.add(p.item.item_id);
            return true;
          }
        );
        if (picks.length === 0) return null;
        return (
        <div key={section.occasion} className="mt-6">
          <div className="px-4">
            <SectionHeader
              title={section.label}
              subtitle={`${picks.length} picks`}
            />
          </div>
          <div className="flex gap-3 overflow-x-auto hide-scrollbar px-4 mt-2 pb-2">
            {picks.map(
              (pick: {
                item: WardrobeItem;
                taste_score: number;
                unlock_count: number;
              }) => (
                <div key={pick.item.item_id} className="w-[160px] shrink-0">
                  <ProductCard
                    item={pick.item}
                    isSaved={isItemSaved(pick.item.item_id)}
                    tasteFit={pick.taste_score}
                    onPress={() => onProductTap(pick.item.item_id)}
                    onBookmark={() => onBookmark(pick.item)}
                    onDismiss={() => onDismiss(pick.item.item_id)}
                  />
                </div>
              )
            )}
          </div>
        </div>
        );
      })}
    </div>
  );
}

export default DiscoveryFeed;
