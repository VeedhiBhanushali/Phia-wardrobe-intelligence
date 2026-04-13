"use client";

import { ChevronRight } from "lucide-react";
import { resolveImageUrl } from "@/lib/api";
import type { WardrobeItem } from "@/lib/store";

interface WardrobeBuilderCardProps {
  wardrobeItems: WardrobeItem[];
  onOpen: () => void;
}

export function WardrobeBuilderCard({
  wardrobeItems,
  onOpen,
}: WardrobeBuilderCardProps) {
  const previewItems = wardrobeItems.slice(0, 3);
  const remaining = wardrobeItems.length - previewItems.length;

  return (
    <button
      type="button"
      onClick={onOpen}
      className="rounded-2xl border border-phia-gray-200 bg-white text-left w-full transition-shadow hover:shadow-sm active:scale-[0.99] transition-transform"
    >
      {/* Header */}
      <div className="px-4 pt-4 pb-3 flex items-start justify-between">
        <div>
          <h3 className="font-serif text-lg text-phia-black leading-tight">
            Your Wardrobe
          </h3>
          <p className="text-[11px] text-phia-gray-400 mt-0.5">
            your virtual closet
          </p>
        </div>
        <div className="flex items-center gap-1 mt-0.5">
          {wardrobeItems.length > 0 && (
            <span className="text-[10px] text-phia-gray-400 font-medium">
              {wardrobeItems.length}
            </span>
          )}
          <ChevronRight size={16} className="text-phia-gray-300" />
        </div>
      </div>

      {/* Preview images */}
      <div className="px-4 pb-4">
        {previewItems.length === 0 ? (
          <div className="rounded-xl bg-phia-gray-50 py-5 flex items-center justify-center">
            <p className="text-xs text-phia-gray-300">
              Tap to start building
            </p>
          </div>
        ) : (
          <div className="flex gap-2">
            {previewItems.map((item) => (
              <div
                key={item.item_id}
                className="flex-1 aspect-[3/4] rounded-xl bg-phia-gray-100 overflow-hidden"
              >
                <img
                  src={resolveImageUrl(item.image_url)}
                  alt={item.title}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
              </div>
            ))}
            {remaining > 0 && (
              <div className="flex-1 aspect-[3/4] rounded-xl bg-phia-gray-50 flex items-center justify-center">
                <span className="text-xs font-medium text-phia-gray-400">
                  +{remaining}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </button>
  );
}

export default WardrobeBuilderCard;
