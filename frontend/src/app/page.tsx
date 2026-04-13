"use client";

import { useState, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { MessageCircle } from "lucide-react";
import { PhiaHeader } from "@/components/ui/PhiaHeader";
import { BottomNav } from "@/components/ui/BottomNav";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { ProductCard } from "@/components/ui/ProductCard";
import { SearchBar } from "@/components/ui/SearchBar";
import { PillTabBar } from "@/components/ui/PillTabBar";
import { ProductGridSkeleton } from "@/components/ui/Skeleton";
import { ErrorBoundary, InlineError } from "@/components/ui/ErrorBoundary";
import { DemoProfileSelector } from "@/components/ui/DemoProfileSelector";
import { WardrobeBuilderCard } from "@/components/builders/WardrobeBuilderCard";
import { AestheticBuilderCard } from "@/components/builders/AestheticBuilderCard";
import { DiscoveryFeed } from "@/components/feed/DiscoveryFeed";
import { ChatDrawer } from "@/components/chat/ChatDrawer";
import { ProductDetail } from "@/components/product/ProductDetail";
import { WardrobeFullView } from "@/components/wardrobe/WardrobeFullView";
import { useAppState } from "@/lib/store";
import { useEventLog } from "@/lib/useEventLog";
import { api } from "@/lib/api";
import type { WardrobeItem, TasteProfile } from "@/lib/store";

type BottomTab = "home" | "search" | "saved" | "profile";

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
    sessionViewedItems,
    addViewedItem,
    sessionIntentVector,
    setSessionIntentVector,
    sessionIntentConfidence,
    setSessionIntentConfidence,
    sessionIntentLabels,
    setSessionIntentLabels,
    chatOpen,
    setChatOpen,
    chatPreloadItemId,
    setChatPreloadItemId,
    clearAll,
  } = useAppState();

  const [bottomTab, setBottomTab] = useState<BottomTab>("home");
  const [catalogItems, setCatalogItems] = useState<WardrobeItem[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [slotFilter, setSlotFilter] = useState("");
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [wardrobeFullOpen, setWardrobeFullOpen] = useState(false);
  const [showExplainer, setShowExplainer] = useState(true);

  const { log } = useEventLog(tasteProfile?.user_id || "");

  // Compute intent when we have 3+ viewed items
  useEffect(() => {
    if (sessionViewedItems.length >= 3) {
      api.intent
        .compute({
          viewed_embeddings: sessionViewedItems.map((v) => v.embedding),
        })
        .then((res) => {
          setSessionIntentVector(res.intent_vector);
          setSessionIntentConfidence(res.confidence);
          setSessionIntentLabels(res.session_labels ?? []);
        })
        .catch(() => {});
    }
  }, [sessionViewedItems, setSessionIntentVector, setSessionIntentConfidence, setSessionIntentLabels]);

  useEffect(() => {
    if (showExplainer) {
      const t = setTimeout(() => setShowExplainer(false), 6000);
      return () => clearTimeout(t);
    }
  }, [showExplainer]);

  const handleForYouDismiss = async (itemId: string) => {
    log("dismiss", "foryou", itemId);
    addSkippedItem(itemId);

    if (tasteProfile?.style_attributes) {
      try {
        const updated = await api.taste.dismiss({
          item_id: itemId,
          style_attributes: tasteProfile.style_attributes,
          dismiss_count: skippedItemIds.length + 1,
        });
        setTasteProfile({
          ...tasteProfile,
          style_attributes: updated.style_attributes,
          style_summary: updated.style_summary,
        });
      } catch {
        // non-critical
      }
    }
  };

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
            style_attributes: tasteProfile.style_attributes ?? {},
          });
          setTasteProfile({
            ...tasteProfile,
            taste_vector: updated.taste_vector,
            trend_fingerprint: updated.trend_fingerprint,
            display_trends: updated.display_trends,
            aesthetic_attributes: updated.aesthetic_attributes as TasteProfile["aesthetic_attributes"],
            price_tier: updated.price_tier as [number, number],
            style_attributes: updated.style_attributes,
            style_summary: updated.style_summary,
          });
        } catch {
          // non-critical
        }
      }
    }
  };

  const handleProductTap = (item: WardrobeItem) => {
    log("click", "catalog", item.item_id);
    setSelectedProductId(item.item_id);
  };

  const handleDemoProfile = async (profileId: string) => {
    clearAll();
    // Load demo wardrobe items from catalog
    try {
      const res = (await api.catalog.search({ per_page: 200 })) as {
        items: WardrobeItem[];
      };

      // Demo wardrobe IDs
      const demoWardrobes: Record<string, string[]> = {
        minimalist: [
          "tops_0000", "tops_0002", "tops_0006", "tops_0012",
          "outerwear_0050", "outerwear_0056",
          "bags_0090", "accessories_0120",
        ],
        streetwear: [
          "tops_0005", "tops_0007", "tops_0019",
          "outerwear_0054", "outerwear_0058", "outerwear_0064",
          "shoes_0071", "shoes_0076", "shoes_0082", "shoes_0088",
          "bags_0096", "accessories_0123",
        ],
        smart_casual: [
          "tops_0000", "tops_0008", "tops_0015",
          "bottoms_0026", "bottoms_0030",
          "shoes_0072",
        ],
        cold_start: [],
      };

      const itemIds = new Set(demoWardrobes[profileId] || []);
      const items = (res.items || []).filter((item) =>
        itemIds.has(item.item_id)
      );
      items.forEach((item) => addWardrobeItem(item));
    } catch {
      // non-critical
    }
  };

  const renderCatalogGrid = (items: WardrobeItem[], emptyMsg?: string) => {
    if (catalogLoading) return <ProductGridSkeleton count={4} />;
    if (catalogError)
      return <InlineError message={catalogError} onRetry={fetchCatalog} />;
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
            onSave={(item) => addWardrobeItem(item)}
            onRemove={removeWardrobeItem}
            intentVector={sessionIntentVector}
            intentConfidence={sessionIntentConfidence}
            onAskPhia={(itemId) => {
              setChatPreloadItemId(itemId);
              setChatOpen(true);
            }}
            onViewItem={addViewedItem}
          />
        </ErrorBoundary>
        <BottomNav activeTab={bottomTab} onTabChange={setBottomTab} />
      </div>
    );
  }

  if (wardrobeFullOpen) {
    return (
      <div className="min-h-dvh flex flex-col">
        <div className="flex-1 pb-20 overflow-y-auto">
          <ErrorBoundary>
            <WardrobeFullView
              wardrobeItems={wardrobeItems}
              isItemSaved={isItemSaved}
              onSaveItem={(item) => {
                log("save", "wardrobe_grid", item.item_id);
                addWardrobeItem(item);
              }}
              onRemoveItem={removeWardrobeItem}
              onProductTap={setSelectedProductId}
              onBack={() => setWardrobeFullOpen(false)}
            />
          </ErrorBoundary>
        </div>
        <BottomNav
          activeTab={bottomTab}
          onTabChange={(tab) => {
            setWardrobeFullOpen(false);
            setBottomTab(tab);
          }}
        />
        <ChatDrawer
          isOpen={chatOpen}
          onClose={() => {
            setChatOpen(false);
            setChatPreloadItemId(null);
          }}
          tasteProfile={tasteProfile}
          wardrobeItems={wardrobeItems}
          onSaveItem={addWardrobeItem}
          isItemSaved={isItemSaved}
          preloadItemId={chatPreloadItemId}
          intentLabels={sessionIntentLabels}
          intentConfidence={sessionIntentConfidence}
        />
      </div>
    );
  }

  return (
    <div className="min-h-dvh flex flex-col">
      <div className="flex-1 pb-20 overflow-y-auto">
        {bottomTab === "home" && (
          <>
            <PhiaHeader />
            <div className="pt-14">
              {/* First-load explainer */}
              <AnimatePresence>
                {showExplainer && (
                  <motion.button
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    onClick={() => setShowExplainer(false)}
                    className="w-full px-4 pt-2 pb-1 text-left"
                  >
                    <p className="text-[11px] text-phia-gray-400 leading-relaxed tracking-wide">
                      <span className="font-medium text-phia-gray-500">Wardrobe IQ</span>{" "}
                      learns your taste, what you own, and what you&apos;re looking for right now.
                    </p>
                  </motion.button>
                )}
              </AnimatePresence>

              {/* Demo profile selector + chat button */}
              <div className="flex items-center justify-between px-4 py-2">
                <DemoProfileSelector onSelect={handleDemoProfile} />
                <button
                  onClick={() => {
                    setChatPreloadItemId(null);
                    setChatOpen(true);
                  }}
                  className="flex items-center gap-1.5 rounded-full bg-phia-black text-white px-3 py-1.5 text-xs font-medium"
                >
                  <MessageCircle size={12} />
                  Ask Phia
                </button>
              </div>

              {/* Builder Cards */}
              <div className="px-4 grid grid-cols-2 gap-3">
                <ErrorBoundary>
                  <WardrobeBuilderCard
                    wardrobeItems={wardrobeItems}
                    onOpen={() => setWardrobeFullOpen(true)}
                  />
                </ErrorBoundary>
                <ErrorBoundary>
                  <AestheticBuilderCard
                    tasteProfile={tasteProfile}
                    onTasteComplete={setTasteProfile}
                  />
                </ErrorBoundary>
              </div>

              {/* Discovery Feed */}
              <ErrorBoundary>
                <DiscoveryFeed
                  tasteProfile={tasteProfile}
                  wardrobeItems={wardrobeItems}
                  skippedItemIds={skippedItemIds}
                  intentVector={sessionIntentVector}
                  intentConfidence={sessionIntentConfidence}
                  isItemSaved={isItemSaved}
                  onBookmark={handleBookmark}
                  onProductTap={setSelectedProductId}
                  onDismiss={handleForYouDismiss}
                />
              </ErrorBoundary>
            </div>
          </>
        )}

        {bottomTab === "search" && (
          <div className="px-4 pt-14 pb-4">
            <h1 className="font-serif text-2xl text-phia-black mb-4">
              Search
            </h1>
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
              <h1 className="font-serif text-2xl text-phia-black">
                Your saved
              </h1>
              <span className="text-xs text-phia-gray-400">
                {wardrobeItems.length} item
                {wardrobeItems.length !== 1 ? "s" : ""}
              </span>
            </div>
            <PillTabBar
              tabs={[
                { id: "", label: "All" },
                ...OUTFIT_SLOTS.slice(1),
              ]}
              activeTab={slotFilter}
              onTabChange={setSlotFilter}
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
                {wardrobeItems
                  .filter(
                    (item) => !slotFilter || item.slot === slotFilter
                  )
                  .map((item) => (
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
            <h1 className="font-serif text-2xl text-phia-black mb-4">
              Profile
            </h1>
            <div className="rounded-2xl bg-phia-gray-50 p-6">
              <div className="w-16 h-16 rounded-full bg-phia-gray-200 mx-auto mb-3" />
              <p className="text-center text-sm text-phia-gray-400">
                Wardrobe IQ Demo
              </p>
              {tasteProfile && (
                <div className="mt-4 pt-4 border-t border-phia-gray-200">
                  <p className="text-xs text-phia-gray-400 mb-2">
                    Your stats
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="text-center">
                      <p className="text-2xl font-semibold text-phia-black">
                        {wardrobeItems.length}
                      </p>
                      <p className="text-xs text-phia-gray-400">
                        Saved items
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-semibold text-phia-black">
                        {Object.values(
                          tasteProfile.aesthetic_attributes ?? {}
                        )[0]?.label || "\u2014"}
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

      {/* Chat Drawer */}
      <ChatDrawer
        isOpen={chatOpen}
        onClose={() => {
          setChatOpen(false);
          setChatPreloadItemId(null);
        }}
        tasteProfile={tasteProfile}
        wardrobeItems={wardrobeItems}
        onSaveItem={addWardrobeItem}
        isItemSaved={isItemSaved}
        preloadItemId={chatPreloadItemId}
        intentLabels={sessionIntentLabels}
        intentConfidence={sessionIntentConfidence}
      />
    </div>
  );
}
