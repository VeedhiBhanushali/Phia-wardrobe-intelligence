"use client";

import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, Search } from "lucide-react";
import { ProductCard } from "@/components/ui/ProductCard";
import { PillTabBar } from "@/components/ui/PillTabBar";
import { ProductGridSkeleton } from "@/components/ui/Skeleton";
import { resolveImageUrl, api } from "@/lib/api";
import type { WardrobeItem } from "@/lib/store";

const OUTFIT_SLOTS = [
  { id: "", label: "All" },
  { id: "tops", label: "Tops" },
  { id: "bottoms", label: "Bottoms" },
  { id: "outerwear", label: "Outerwear" },
  { id: "shoes", label: "Shoes" },
  { id: "bags", label: "Bags" },
  { id: "accessories", label: "Accessories" },
];

interface WardrobeFullViewProps {
  wardrobeItems: WardrobeItem[];
  isItemSaved: (itemId: string) => boolean;
  onSaveItem: (item: WardrobeItem) => void;
  onRemoveItem: (itemId: string) => void;
  onProductTap: (itemId: string) => void;
  onBack: () => void;
}

export function WardrobeFullView({
  wardrobeItems,
  isItemSaved,
  onSaveItem,
  onRemoveItem,
  onProductTap,
  onBack,
}: WardrobeFullViewProps) {
  const [tab, setTab] = useState<"wardrobe" | "browse">("wardrobe");
  const [searchQuery, setSearchQuery] = useState("");
  const [slotFilter, setSlotFilter] = useState("");
  const [catalogItems, setCatalogItems] = useState<WardrobeItem[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(false);

  // Slot coverage
  const slotCounts: Record<string, number> = {};
  for (const item of wardrobeItems) {
    slotCounts[item.slot] = (slotCounts[item.slot] || 0) + 1;
  }

  const fetchCatalog = useCallback(async () => {
    setCatalogLoading(true);
    try {
      const res = (await api.catalog.search({
        query: searchQuery || undefined,
        slot: slotFilter || undefined,
        per_page: 20,
      })) as { items: WardrobeItem[] };
      setCatalogItems(res.items || []);
    } catch {
      // non-critical
    } finally {
      setCatalogLoading(false);
    }
  }, [searchQuery, slotFilter]);

  useEffect(() => {
    if (tab === "browse") fetchCatalog();
  }, [tab, fetchCatalog]);

  const handleBookmark = (item: WardrobeItem) => {
    if (isItemSaved(item.item_id)) {
      onRemoveItem(item.item_id);
    } else {
      onSaveItem(item);
    }
  };

  const filteredWardrobe = wardrobeItems.filter(
    (item) => !slotFilter || item.slot === slotFilter
  );

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.2 }}
      className="min-h-dvh pb-20"
    >
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white border-b border-phia-gray-100">
        <div className="flex items-center gap-3 px-4 py-3">
          <button
            onClick={onBack}
            className="w-9 h-9 rounded-full bg-phia-gray-50 flex items-center justify-center"
          >
            <ArrowLeft size={18} className="text-phia-black" />
          </button>
          <div className="flex-1">
            <h1 className="font-serif text-xl text-phia-black">My Wardrobe</h1>
            <p className="text-xs text-phia-gray-400">
              {wardrobeItems.length} item{wardrobeItems.length !== 1 ? "s" : ""} saved
            </p>
          </div>
        </div>

        {/* Slot coverage strip */}
        <div className="flex gap-1.5 px-4 pb-2">
          {OUTFIT_SLOTS.slice(1).map((slot) => {
            const count = slotCounts[slot.id] || 0;
            const filled = count > 0;
            return (
              <div
                key={slot.id}
                className={`flex-1 rounded-full py-1 text-center text-[9px] font-medium ${
                  filled
                    ? "bg-phia-black text-white"
                    : "bg-phia-gray-100 text-phia-gray-400"
                }`}
              >
                {slot.label}
                {filled && <span className="ml-0.5">{count}</span>}
              </div>
            );
          })}
        </div>

        {/* Tabs */}
        <div className="flex border-b border-phia-gray-100">
          {(["wardrobe", "browse"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`flex-1 py-2.5 text-sm font-medium text-center transition-colors ${
                tab === t
                  ? "text-phia-black border-b-2 border-phia-black"
                  : "text-phia-gray-400"
              }`}
            >
              {t === "wardrobe" ? "My Wardrobe" : "Browse Catalog"}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="px-4 pt-3">
        {tab === "wardrobe" ? (
          <>
            <div className="mb-3">
              <PillTabBar
                tabs={OUTFIT_SLOTS}
                activeTab={slotFilter}
                onTabChange={setSlotFilter}
              />
            </div>
            {filteredWardrobe.length === 0 ? (
              <div className="py-12 text-center">
                <p className="text-sm text-phia-gray-400">
                  {wardrobeItems.length === 0
                    ? "No items saved yet"
                    : "No items in this category"}
                </p>
                <button
                  onClick={() => setTab("browse")}
                  className="text-xs text-phia-blue mt-2"
                >
                  Browse catalog
                </button>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                {filteredWardrobe.map((item) => (
                  <ProductCard
                    key={item.item_id}
                    item={item}
                    isSaved={true}
                    onPress={() => onProductTap(item.item_id)}
                    onBookmark={() => onRemoveItem(item.item_id)}
                  />
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            <div className="flex items-center gap-2 mb-3">
              <div className="flex-1 flex items-center gap-2 rounded-full border border-phia-gray-200 px-3 py-2">
                <Search size={14} className="text-phia-gray-400" />
                <input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search catalog..."
                  className="flex-1 text-sm text-phia-black outline-none placeholder:text-phia-gray-300"
                />
              </div>
            </div>
            <div className="mb-3">
              <PillTabBar
                tabs={OUTFIT_SLOTS}
                activeTab={slotFilter}
                onTabChange={setSlotFilter}
              />
            </div>
            {catalogLoading ? (
              <ProductGridSkeleton count={6} />
            ) : (
              <div className="grid grid-cols-2 gap-3">
                {catalogItems.map((item) => (
                  <ProductCard
                    key={item.item_id}
                    item={item}
                    isSaved={isItemSaved(item.item_id)}
                    onPress={() => onProductTap(item.item_id)}
                    onBookmark={() => handleBookmark(item)}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}

export default WardrobeFullView;
