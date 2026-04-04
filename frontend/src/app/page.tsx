"use client";

import { useState, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { PhiaHeader } from "@/components/ui/PhiaHeader";
import { HomeTabBar } from "@/components/ui/HomeTabBar";
import { BottomNav } from "@/components/ui/BottomNav";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ProductCard } from "@/components/ui/ProductCard";
import { SearchBar } from "@/components/ui/SearchBar";
import { PillTabBar } from "@/components/ui/PillTabBar";
import { ProductGridSkeleton } from "@/components/ui/Skeleton";
import { ErrorBoundary, InlineError } from "@/components/ui/ErrorBoundary";
import { WardrobeTab } from "@/components/wardrobe/WardrobeTab";
import { ProductDetail } from "@/components/product/ProductDetail";
import { useAppState } from "@/lib/store";
import { useEventLog } from "@/lib/useEventLog";
import { api } from "@/lib/api";
import type { WardrobeItem, TasteProfile, OccasionSection } from "@/lib/store";

type BottomTab = "home" | "search" | "saved" | "profile";
type HomeTab = "explore" | "foryou" | "trending" | "wardrobe";

const OUTFIT_SLOTS = [
  { id: "", label: "All" },
  { id: "tops", label: "Tops" },
  { id: "bottoms", label: "Bottoms" },
  { id: "outerwear", label: "Outerwear" },
  { id: "shoes", label: "Shoes" },
  { id: "bags", label: "Bags" },
  { id: "accessories", label: "Accessories" },
];

export default function Home() {
  const {
    tasteProfile,
    setTasteProfile,
    wardrobeItems,
    addWardrobeItem,
    removeWardrobeItem,
    isItemSaved,
    skippedItemIds,
    addSkippedItem,
  } = useAppState();

  const [bottomTab, setBottomTab] = useState<BottomTab>("home");
  const [homeTab, setHomeTab] = useState<HomeTab>("wardrobe");
  const [catalogItems, setCatalogItems] = useState<WardrobeItem[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [slotFilter, setSlotFilter] = useState("");
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [forYouItems, setForYouItems] = useState<(WardrobeItem & { taste_score?: number })[]>([]);
  const [forYouOccasions, setForYouOccasions] = useState<OccasionSection[]>([]);
  const [forYouLoading, setForYouLoading] = useState(false);

  const { log } = useEventLog(tasteProfile?.user_id || "");

  const handleForYouDismiss = (itemId: string) => {
    log("dismiss", "foryou", itemId);
    addSkippedItem(itemId);
  };

  const fetchForYou = useCallback(async () => {
    if (!tasteProfile?.taste_vector?.length) return;
    setForYouLoading(true);
    try {
      const excludeIds = [
        ...wardrobeItems.map((i) => i.item_id),
        ...skippedItemIds,
      ];
      const res = await api.catalog.tasteSearch({
        taste_vector: tasteProfile.taste_vector,
        top_k: 20,
        exclude_ids: excludeIds,
      });
      setForYouItems(
        (res.items || []) as unknown as (WardrobeItem & { taste_score?: number })[]
      );

      const occasionLabels: Record<string, string> = {
        work: "Your work style",
        casual: "Your everyday",
        evening: "Your going-out",
        weekend: "Your weekend",
        special: "Your standout looks",
      };

      const occVecs = tasteProfile.occasion_vectors;
      if (occVecs && Object.keys(occVecs).length > 0) {
        const shownIds = new Set(excludeIds);
        (res.items || []).forEach((it) => {
          const id = (it as { item_id?: string }).item_id;
          if (id) shownIds.add(id);
        });

        const sections: OccasionSection[] = [];
        for (const [occ, vec] of Object.entries(occVecs)) {
          if (!vec?.length) continue;
          try {
            const occRes = await api.catalog.tasteSearch({
              taste_vector: vec,
              top_k: 15,
              exclude_ids: [...shownIds],
            });
            const occItems = (occRes.items || [])
              .filter(
                (it) =>
                  !shownIds.has((it as unknown as WardrobeItem).item_id)
              )
              .slice(0, 6) as unknown as (WardrobeItem & {
                taste_score?: number;
              })[];

            occItems.forEach((it) => shownIds.add(it.item_id));

            if (occItems.length > 0) {
              sections.push({
                occasion: occ,
                label: occasionLabels[occ] ?? occ,
                items: occItems.map((it) => ({
                  item: it,
                  taste_score: it.taste_score ?? 0,
                  unlock_count: 0,
                  explanation: "",
                })),
              });
            }
          } catch {
            // non-critical
          }
        }
        setForYouOccasions(sections);
      }
    } catch {
      // non-critical
    } finally {
      setForYouLoading(false);
    }
  }, [tasteProfile, wardrobeItems, skippedItemIds]);

  useEffect(() => {
    if (tasteProfile?.taste_vector?.length) {
      fetchForYou();
    }
  }, [tasteProfile, fetchForYou]);

  const fetchCatalog = useCallback(async () => {
    setCatalogLoading(true);
    setCatalogError(null);
    try {
      const res = (await api.catalog.search({
        query: searchQuery || undefined,
        slot: slotFilter || undefined,
        page: 1,
      })) as { items: WardrobeItem[] };
      setCatalogItems(res.items || []);
    } catch {
      setCatalogError("Couldn't load catalog. Is the backend running?");
    } finally {
      setCatalogLoading(false);
    }
  }, [searchQuery, slotFilter]);

  useEffect(() => {
    fetchCatalog();
  }, [fetchCatalog]);

  const handleBookmark = async (item: WardrobeItem) => {
    if (isItemSaved(item.item_id)) {
      removeWardrobeItem(item.item_id);
    } else {
      log("save", "catalog", item.item_id);
      addWardrobeItem(item);

      if (tasteProfile?.taste_vector?.length) {
        try {
          const updated = await api.taste.update({
            user_id: tasteProfile.user_id,
            taste_vector: tasteProfile.taste_vector,
            item_id: item.item_id,
            save_count: wardrobeItems.length + 1,
          });
          setTasteProfile({
            ...tasteProfile,
            taste_vector: updated.taste_vector,
            trend_fingerprint: updated.trend_fingerprint,
            aesthetic_attributes: updated.aesthetic_attributes as TasteProfile["aesthetic_attributes"],
            price_tier: updated.price_tier as [number, number],
          });
        } catch {
          // non-critical: taste still works with the original vector
        }
      }
    }
  };

  const handleProductTap = (item: WardrobeItem) => {
    log("click", "catalog", item.item_id);
    setSelectedProductId(item.item_id);
  };

  if (selectedProductId) {
    return (
      <div className="min-h-dvh pb-20">
        <ErrorBoundary>
          <ProductDetail
            itemId={selectedProductId}
            tasteProfile={tasteProfile}
            wardrobeItems={wardrobeItems}
            isSaved={isItemSaved(selectedProductId)}
            onBack={() => setSelectedProductId(null)}
            onSave={addWardrobeItem}
            onRemove={removeWardrobeItem}
          />
        </ErrorBoundary>
        <BottomNav activeTab={bottomTab} onTabChange={setBottomTab} />
      </div>
    );
  }

  const renderCatalogGrid = (items: WardrobeItem[], emptyMsg?: string) => {
    if (catalogLoading) return <ProductGridSkeleton count={4} />;
    if (catalogError) return <InlineError message={catalogError} onRetry={fetchCatalog} />;
    if (items.length === 0 && emptyMsg) {
      return (
        <div className="rounded-2xl bg-phia-gray-50 p-5 text-center">
          <p className="text-sm text-phia-gray-400">{emptyMsg}</p>
        </div>
      );
    }
    return (
      <div className="grid grid-cols-2 gap-3">
        {items.map((item) => (
          <ProductCard
            key={item.item_id}
            item={item}
            isSaved={isItemSaved(item.item_id)}
            onPress={() => handleProductTap(item)}
            onBookmark={() => handleBookmark(item)}
          />
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-dvh flex flex-col">
      <div className="flex-1 pb-20 overflow-y-auto">
        {bottomTab === "home" && (
          <>
            <PhiaHeader />
            <div className="pt-14">
              <HomeTabBar activeTab={homeTab} onTabChange={(id) => setHomeTab(id as HomeTab)} />

              <AnimatePresence mode="wait">
                {homeTab === "wardrobe" && (
                  <motion.div
                    key="wardrobe"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                  >
                    <ErrorBoundary>
                      <WardrobeTab
                        tasteProfile={tasteProfile}
                        wardrobeItems={wardrobeItems}
                        skippedItemIds={skippedItemIds}
                        onSkipItem={addSkippedItem}
                        onTasteComplete={setTasteProfile}
                        onSaveItem={addWardrobeItem}
                        onRemoveItem={removeWardrobeItem}
                        isItemSaved={isItemSaved}
                        onProductTap={setSelectedProductId}
                      />
                    </ErrorBoundary>
                  </motion.div>
                )}

                {homeTab === "explore" && (
                  <motion.div
                    key="explore"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="px-4 py-4"
                  >
                    <SectionHeader title="Browse styles" />
                    <div className="flex gap-3 overflow-x-auto hide-scrollbar mt-3 pb-2">
                      {["Night Out", "Wedding Guest", "Everyday", "Office", "Travel"].map(
                        (style) => (
                          <div
                            key={style}
                            className="w-[160px] shrink-0 aspect-[3/4] rounded-2xl bg-phia-gray-100 relative overflow-hidden"
                          >
                            <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
                            <p className="absolute bottom-4 left-4 text-white font-serif text-lg">
                              {style}
                            </p>
                          </div>
                        )
                      )}
                    </div>

                    <div className="mt-6">
                      <SectionHeader title="Trending items" />
                      <div className="mt-3">
                        {renderCatalogGrid(catalogItems.slice(0, 4))}
                      </div>
                    </div>
                  </motion.div>
                )}

                {homeTab === "foryou" && (
                  <motion.div
                    key="foryou"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="px-4 py-4"
                  >
                    <SectionHeader title="For you" />
                    <p className="text-sm text-phia-gray-400 mt-1 mb-4">
                      {tasteProfile?.trend_fingerprint
                        ? `Curated for your ${Object.keys(tasteProfile.trend_fingerprint)[0] ?? "unique"} aesthetic`
                        : tasteProfile
                        ? "Ranked by how well each item matches your Pinterest aesthetic"
                        : "Set up your taste profile in the Wardrobe tab to get personalized picks"}
                    </p>
                    {!tasteProfile ? (
                      renderCatalogGrid(catalogItems.slice(0, 8), "No items available yet")
                    ) : forYouLoading ? (
                      <ProductGridSkeleton count={6} />
                    ) : forYouItems.length > 0 ? (
                      <>
                        <div className="grid grid-cols-2 gap-3">
                          {forYouItems.slice(0, 6).map((item) => (
                            <ProductCard
                              key={item.item_id}
                              item={item}
                              isSaved={isItemSaved(item.item_id)}
                              tasteFit={item.taste_score}
                              onPress={() => handleProductTap(item)}
                              onBookmark={() => handleBookmark(item)}
                              onDismiss={() => handleForYouDismiss(item.item_id)}
                            />
                          ))}
                        </div>

                        {forYouOccasions.map((section) => (
                          <div key={section.occasion} className="mt-6">
                            <SectionHeader
                              title={section.label}
                              subtitle={`${section.items.length} picks`}
                            />
                            <div className="flex gap-3 overflow-x-auto hide-scrollbar mt-2 pb-2 -mx-4 px-4">
                              {section.items.map((pick) => (
                                <div key={pick.item.item_id} className="w-[160px] shrink-0">
                                  <ProductCard
                                    item={pick.item}
                                    isSaved={isItemSaved(pick.item.item_id)}
                                    tasteFit={pick.taste_score}
                                    onPress={() => handleProductTap(pick.item)}
                                    onBookmark={() => handleBookmark(pick.item)}
                                    onDismiss={() =>
                                      handleForYouDismiss(pick.item.item_id)
                                    }
                                  />
                                </div>
                              ))}
                            </div>
                          </div>
                        ))}

                        {forYouItems.length > 6 && (
                          <div className="mt-6">
                            <SectionHeader title="More for you" />
                            <div className="grid grid-cols-2 gap-3 mt-2">
                              {forYouItems.slice(6).map((item) => (
                                <ProductCard
                                  key={item.item_id}
                                  item={item}
                                  isSaved={isItemSaved(item.item_id)}
                                  tasteFit={item.taste_score}
                                  onPress={() => handleProductTap(item)}
                                  onBookmark={() => handleBookmark(item)}
                                  onDismiss={() => handleForYouDismiss(item.item_id)}
                                />
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      renderCatalogGrid([], "No personalized items yet")
                    )}
                  </motion.div>
                )}

                {homeTab === "trending" && (
                  <motion.div
                    key="trending"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="px-4 py-4"
                  >
                    <SectionHeader title="Trending on " />
                    <span className="font-serif font-bold italic text-xl -mt-5 block">
                      phia
                    </span>
                    <div className="mt-4">
                      {renderCatalogGrid(catalogItems.slice(4, 12), "No trending items yet")}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </>
        )}

        {bottomTab === "search" && (
          <div className="px-4 pt-14 pb-4">
            <h1 className="font-serif text-2xl text-phia-black mb-4">Search</h1>
            <SearchBar
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder="Search products"
            />
            <div className="mt-3">
              <PillTabBar
                tabs={OUTFIT_SLOTS}
                activeTab={slotFilter}
                onTabChange={setSlotFilter}
              />
            </div>
            <div className="mt-4">
              {renderCatalogGrid(catalogItems, "No results found")}
            </div>
          </div>
        )}

        {bottomTab === "saved" && (
          <div className="px-4 pt-14 pb-4">
            <div className="flex items-center justify-between mb-4">
              <h1 className="font-serif text-2xl text-phia-black">Your saved</h1>
              <button className="w-8 h-8 rounded-full border border-phia-gray-200 flex items-center justify-center text-phia-gray-400 text-lg">
                +
              </button>
            </div>
            <PillTabBar
              tabs={[
                { id: "items", label: "Items" },
                { id: "wishlists", label: "Wishlists" },
                { id: "brands", label: "Brands" },
              ]}
              activeTab="items"
              onTabChange={() => {}}
            />
            {wardrobeItems.length === 0 ? (
              <div className="mt-8 text-center">
                <p className="text-sm text-phia-gray-400">
                  No saved items yet
                </p>
                <p className="text-xs text-phia-gray-300 mt-1">
                  Browse and save items to see them here
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3 mt-4">
                {wardrobeItems.map((item) => (
                  <ProductCard
                    key={item.item_id}
                    item={item}
                    isSaved={true}
                    onPress={() => setSelectedProductId(item.item_id)}
                    onBookmark={() => removeWardrobeItem(item.item_id)}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {bottomTab === "profile" && (
          <div className="px-4 pt-14 pb-4">
            <h1 className="font-serif text-2xl text-phia-black mb-4">Profile</h1>
            <div className="rounded-2xl bg-phia-gray-50 p-6">
              <div className="w-16 h-16 rounded-full bg-phia-gray-200 mx-auto mb-3" />
              <p className="text-center text-sm text-phia-gray-400">
                Wardrobe IQ Demo
              </p>
              {tasteProfile && (
                <div className="mt-4 pt-4 border-t border-phia-gray-200">
                  <p className="text-xs text-phia-gray-400 mb-2">Your stats</p>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="text-center">
                      <p className="text-2xl font-semibold text-phia-black">
                        {wardrobeItems.length}
                      </p>
                      <p className="text-xs text-phia-gray-400">Saved items</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-semibold text-phia-black">
                        {Object.values(tasteProfile.aesthetic_attributes ?? {})[0]?.label || "—"}
                      </p>
                      <p className="text-xs text-phia-gray-400">Style</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <BottomNav activeTab={bottomTab} onTabChange={setBottomTab} />
    </div>
  );
}
