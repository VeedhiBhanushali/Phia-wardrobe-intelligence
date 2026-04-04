"use client";

import { useState, useEffect, useCallback } from "react";
import { TasteOnboarding } from "@/components/onboarding/TasteOnboarding";
import { AestheticProfileCard } from "@/components/wardrobe/AestheticProfileCard";
import { GapRecommendationCard } from "@/components/wardrobe/GapRecommendationCard";
import { CompleteTheLookSheet } from "@/components/wardrobe/CompleteTheLookSheet";
import { OutfitUnlockBadge } from "@/components/wardrobe/OutfitUnlockBadge";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ProductCard } from "@/components/ui/ProductCard";
import { InlineError } from "@/components/ui/ErrorBoundary";
import {
  GapRecommendationSkeleton,
  ProductGridSkeleton,
  ProfileChipsSkeleton,
} from "@/components/ui/Skeleton";
import { api, resolveImageUrl } from "@/lib/api";
import { useEventLog, type LogModule } from "@/lib/useEventLog";
import type {
  TasteProfile,
  WardrobeItem,
  GapRec,
  OccasionSection,
  OutfitBundle,
} from "@/lib/store";

interface WardrobeTabProps {
  tasteProfile: TasteProfile | null;
  wardrobeItems: WardrobeItem[];
  skippedItemIds: string[];
  onSkipItem: (itemId: string) => void;
  onTasteComplete: (profile: TasteProfile) => void;
  onSaveItem: (item: WardrobeItem) => void;
  onRemoveItem: (itemId: string) => void;
  isItemSaved: (itemId: string) => boolean;
  onProductTap: (itemId: string) => void;
}

function hasUsableTasteProfile(profile: TasteProfile | null): profile is TasteProfile {
  return !!(
    profile &&
    typeof profile.user_id === "string" &&
    profile.user_id.length > 0 &&
    Array.isArray(profile.taste_vector) &&
    profile.taste_vector.length > 0
  );
}

interface TopPick {
  item: WardrobeItem;
  taste_score: number;
  unlock_count: number;
  explanation: string;
}

export function WardrobeTab({
  tasteProfile,
  wardrobeItems,
  skippedItemIds,
  onSkipItem,
  onTasteComplete,
  onSaveItem,
  onRemoveItem,
  isItemSaved,
  onProductTap,
}: WardrobeTabProps) {
  const [gapRec, setGapRec] = useState<GapRec | null>(null);
  const [topPicks, setTopPicks] = useState<TopPick[]>([]);
  const [occasionSections, setOccasionSections] = useState<OccasionSection[]>([]);
  const [shoppingBrief, setShoppingBrief] = useState<Record<string, unknown>>({});
  const [outfitBundles, setOutfitBundles] = useState<OutfitBundle[]>([]);
  const [tasteMatched, setTasteMatched] = useState<(WardrobeItem & { taste_score?: number })[]>([]);
  const [selectedItem, setSelectedItem] = useState<WardrobeItem | null>(null);
  const [pairs, setPairs] = useState<WardrobeItem[]>([]);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [sortBy, setSortBy] = useState<"versatile" | "recent">("versatile");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const readyTasteProfile = hasUsableTasteProfile(tasteProfile) ? tasteProfile : null;
  const { log } = useEventLog(readyTasteProfile?.user_id || "");

  const handleSkip = (module: LogModule, itemId: string) => {
    log("dismiss", module, itemId);
    onSkipItem(itemId);
  };

  const fetchRecommendations = useCallback(async () => {
    if (!readyTasteProfile) return;

    setLoading(true);
    setError(null);
    try {
      const res = (await api.recommendations.wardrobe({
        user_id: readyTasteProfile.user_id,
        wardrobe_item_ids: wardrobeItems.map((i) => i.item_id),
        taste_vector: readyTasteProfile.taste_vector,
        taste_modes: readyTasteProfile.taste_modes,
        occasion_vectors: readyTasteProfile.occasion_vectors,
        trend_fingerprint: readyTasteProfile.trend_fingerprint,
        anti_taste_vector: readyTasteProfile.anti_taste_vector,
        price_tier: readyTasteProfile.price_tier
          ? [readyTasteProfile.price_tier[0], readyTasteProfile.price_tier[1]]
          : undefined,
        aesthetic_label:
          (readyTasteProfile.aesthetic_attributes?.silhouette?.label as string) ?? "",
        skipped_item_ids: skippedItemIds,
      })) as {
        gap_recommendation: GapRec | null;
        top_picks: TopPick[];
        occasion_sections: OccasionSection[];
        outfit_suggestions: OutfitBundle[];
        shopping_brief: Record<string, unknown>;
        wardrobe_stats: Record<string, unknown>;
      };

      setGapRec(res.gap_recommendation);
      setTopPicks(res.top_picks || []);
      setOccasionSections(res.occasion_sections || []);
      setShoppingBrief(res.shopping_brief || {});
      setOutfitBundles(
        (res.outfit_suggestions || []).map((b) => ({
          label: b.label,
          items: (b.items || []) as WardrobeItem[],
        }))
      );

      if (res.gap_recommendation) {
        log("impression", "wardrobe_gap", res.gap_recommendation.item.item_id, {
          score: res.gap_recommendation.confidence,
          unlock_count: res.gap_recommendation.unlock_count,
          taste_score: res.gap_recommendation.taste_score,
        });
      }
    } catch {
      setError("Couldn't load recommendations. Check that the backend is running.");
    } finally {
      setLoading(false);
    }
  }, [readyTasteProfile, wardrobeItems, skippedItemIds, log]);

  const fetchTasteMatched = useCallback(async () => {
    if (!readyTasteProfile) return;
    try {
      const res = await api.catalog.tasteSearch({
        taste_vector: readyTasteProfile.taste_vector,
        top_k: 12,
        exclude_ids: [
          ...wardrobeItems.map((i) => i.item_id),
          ...skippedItemIds,
        ],
      });
      setTasteMatched(
        (res.items || []) as unknown as (WardrobeItem & { taste_score?: number })[]
      );
    } catch {
      // non-critical
    }
  }, [readyTasteProfile, wardrobeItems, skippedItemIds]);

  useEffect(() => {
    if (readyTasteProfile) {
      fetchRecommendations();
      fetchTasteMatched();
    }
  }, [
    readyTasteProfile,
    wardrobeItems.length,
    skippedItemIds,
    fetchRecommendations,
    fetchTasteMatched,
  ]);

  const handleItemTap = (item: WardrobeItem) => {
    log("click", "wardrobe_grid", item.item_id);
    setSelectedItem(item);
    const itemPairs = wardrobeItems.filter(
      (w) => w.item_id !== item.item_id && w.slot !== item.slot
    );
    setPairs(itemPairs.slice(0, 5));
    setSheetOpen(true);
  };

  const handleBookmark = (item: WardrobeItem) => {
    if (isItemSaved(item.item_id)) {
      onRemoveItem(item.item_id);
    } else {
      log("save", "wardrobe_gap", item.item_id);
      onSaveItem(item);
    }
  };

  if (!readyTasteProfile) {
    return <TasteOnboarding onComplete={onTasteComplete} />;
  }

  const sortedWardrobe = [...wardrobeItems].sort((a, b) => {
    if (sortBy === "versatile") {
      return (b.unlock_count || 0) - (a.unlock_count || 0);
    }
    return 0;
  });

  return (
    <div className="pb-4">
      <div className="px-4 pt-2">
        {loading && !gapRec ? (
          <ProfileChipsSkeleton />
        ) : (
          <AestheticProfileCard
            attributes={readyTasteProfile.aesthetic_attributes ?? {}}
            trendFingerprint={readyTasteProfile.trend_fingerprint}
          />
        )}
      </div>

      {Object.keys(shoppingBrief).length > 0 && (
        <div className="px-4 mt-4">
          <div className="rounded-2xl bg-phia-gray-50 p-4 space-y-2">
            <p className="text-[10px] uppercase tracking-[0.15em] text-phia-gray-400">
              Shopper brief
            </p>
            {typeof shoppingBrief.top_trend === "string" &&
              shoppingBrief.top_trend.length > 0 && (
                <p className="text-sm text-phia-black">
                  <span className="text-phia-gray-500">Trend lens:</span>{" "}
                  {shoppingBrief.top_trend}
                </p>
              )}
            {Array.isArray(shoppingBrief.dominant_occasions) &&
              shoppingBrief.dominant_occasions.length > 0 && (
                <p className="text-xs text-phia-gray-600">
                  Occasions we&apos;re shopping for:{" "}
                  {(shoppingBrief.dominant_occasions as string[]).join(", ")}
                </p>
              )}
            {Array.isArray(shoppingBrief.gap_slots) &&
              (shoppingBrief.gap_slots as string[]).length > 0 && (
                <p className="text-xs text-phia-gray-500">
                  Filling gaps: {(shoppingBrief.gap_slots as string[]).join(", ")}
                </p>
              )}
          </div>
        </div>
      )}

      <div className="px-4 mt-4">
        <SectionHeader title="Top pick for you" />
        <div className="mt-2">
          {loading && !gapRec ? (
            <GapRecommendationSkeleton />
          ) : error ? (
            <InlineError message={error} onRetry={fetchRecommendations} />
          ) : gapRec?.item ? (
            <GapRecommendationCard
              item={gapRec.item}
              unlockCount={gapRec.unlock_count ?? 0}
              tasteFit={gapRec.taste_score ?? 0}
              explanation={gapRec.explanation ?? "Recommended for your style"}
              isSaved={isItemSaved(gapRec.item.item_id)}
              onSave={() => handleBookmark(gapRec.item as WardrobeItem)}
              onBrowseSimilar={() => {}}
              onDismiss={() =>
                handleSkip("wardrobe_gap", gapRec.item.item_id)
              }
            />
          ) : (
            <div className="rounded-2xl bg-phia-gray-50 p-5 text-center">
              <p className="text-sm text-phia-gray-400">
                {wardrobeItems.length === 0
                  ? "Browse and save items to get personalized recommendations"
                  : "Save more items to unlock personalized recommendations"}
              </p>
            </div>
          )}
        </div>
      </div>

      {outfitBundles.length > 0 && (
        <div className="px-4 mt-6 space-y-4">
          <SectionHeader title="Suggested outfits" subtitle="Built around your top pick" />
          {outfitBundles.map((bundle) => (
            <div key={bundle.label} className="rounded-2xl border border-phia-gray-100 p-3">
              <p className="text-xs font-medium text-phia-black mb-2">{bundle.label}</p>
              <div className="flex gap-2">
                {bundle.items.map((it) => (
                  <button
                    key={it.item_id}
                    type="button"
                    onClick={() => onProductTap(it.item_id)}
                    className="flex-1 min-w-0 rounded-xl overflow-hidden bg-phia-gray-100 aspect-[3/4]"
                  >
                    <img
                      src={resolveImageUrl(it.image_url)}
                      alt={it.title}
                      className="w-full h-full object-cover"
                    />
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {topPicks.length > 0 && (
        <div className="px-4 mt-6">
          <SectionHeader title="Also recommended" subtitle={`${topPicks.length} items`} />
          <div className="grid grid-cols-2 gap-3 mt-2">
            {topPicks.map((pick) => (
              <ProductCard
                key={pick.item.item_id}
                item={pick.item}
                isSaved={isItemSaved(pick.item.item_id)}
                tasteFit={pick.taste_score}
                unlockCount={pick.unlock_count}
                onPress={() => {
                  log("click", "top_pick", pick.item.item_id);
                  onProductTap(pick.item.item_id);
                }}
                onBookmark={() => handleBookmark(pick.item)}
                onDismiss={() => handleSkip("top_pick", pick.item.item_id)}
              />
            ))}
          </div>
        </div>
      )}

      {occasionSections.map((section) => (
        <div key={section.occasion} className="mt-6">
          <div className="px-4">
            <SectionHeader
              title={section.label}
              subtitle={`${section.items.length} picks`}
            />
          </div>
          <div className="flex gap-3 overflow-x-auto hide-scrollbar px-4 mt-2 pb-2">
            {section.items.map((pick) => (
              <div key={pick.item.item_id} className="w-[160px] shrink-0">
                <ProductCard
                  item={pick.item}
                  isSaved={isItemSaved(pick.item.item_id)}
                  tasteFit={pick.taste_score}
                  onPress={() => {
                    log("click", "occasion_row", pick.item.item_id);
                    onProductTap(pick.item.item_id);
                  }}
                  onBookmark={() => handleBookmark(pick.item)}
                  onDismiss={() =>
                    handleSkip("occasion_row", pick.item.item_id)
                  }
                />
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="px-4 mt-6">
        <div className="flex items-center justify-between mb-3">
          <SectionHeader
            title="Your wardrobe"
            subtitle={
              wardrobeItems.length > 0
                ? `${wardrobeItems.length} item${wardrobeItems.length === 1 ? "" : "s"}`
                : undefined
            }
          />
          {wardrobeItems.length > 1 && (
            <div className="flex gap-1">
              {(["versatile", "recent"] as const).map((opt) => (
                <button
                  key={opt}
                  onClick={() => setSortBy(opt)}
                  className={`px-3 py-1 rounded-full text-xs transition-colors ${
                    sortBy === opt
                      ? "bg-phia-black text-white"
                      : "text-phia-gray-400"
                  }`}
                >
                  {opt === "versatile" ? "Most versatile" : "Recent"}
                </button>
              ))}
            </div>
          )}
        </div>

        {wardrobeItems.length === 0 ? (
          <div className="rounded-2xl bg-phia-gray-50 p-6 text-center">
            <p className="text-sm text-phia-gray-400 mb-1">
              Your wardrobe is empty
            </p>
            <p className="text-xs text-phia-gray-300">
              Save items to build your wardrobe and unlock outfit combinations
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            {sortedWardrobe.map((item) => (
              <div key={item.item_id} className="relative">
                <ProductCard
                  item={item}
                  isSaved={true}
                  onPress={() => handleItemTap(item)}
                  onBookmark={() => handleBookmark(item)}
                />
                {item.unlock_count != null && item.unlock_count > 0 && (
                  <OutfitUnlockBadge count={item.unlock_count} />
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {tasteMatched.length > 0 && (
        <div className="mt-6">
          <div className="px-4">
            <SectionHeader title="Matches your taste" subtitle="Ranked by style similarity" />
          </div>
          <div className="flex gap-3 overflow-x-auto hide-scrollbar px-4 mt-2 pb-2">
            {tasteMatched.map((item) => (
              <div key={item.item_id} className="w-[160px] shrink-0">
                <ProductCard
                  item={item}
                  isSaved={isItemSaved(item.item_id)}
                  tasteFit={item.taste_score}
                  onPress={() => {
                    log("click", "taste_match", item.item_id);
                    onProductTap(item.item_id);
                  }}
                  onBookmark={() => handleBookmark(item)}
                  onDismiss={() => handleSkip("taste_match", item.item_id)}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      <CompleteTheLookSheet
        isOpen={sheetOpen}
        onClose={() => setSheetOpen(false)}
        anchorItem={selectedItem}
        pairs={pairs}
        recommendation={gapRec?.item || null}
      />
    </div>
  );
}

export default WardrobeTab;
