"use client";

import { Bookmark, X } from "lucide-react";
import { motion } from "framer-motion";
import { resolveImageUrl } from "@/lib/api";

interface ProductItem {
  item_id: string;
  title: string;
  brand: string;
  price: number;
  image_url: string;
  slot?: string;
}

interface ProductCardProps {
  item: ProductItem;
  isSaved: boolean;
  tasteFit?: number;
  unlockCount?: number;
  onPress?: () => void;
  onBookmark?: () => void;
  onDismiss?: () => void;
}

export function ProductCard({
  item,
  isSaved,
  tasteFit,
  unlockCount,
  onPress,
  onBookmark,
  onDismiss,
}: ProductCardProps) {
  return (
    <div className="flex flex-col">
      <div
        className="relative aspect-[3/4] overflow-hidden rounded-xl cursor-pointer bg-phia-gray-100"
        onClick={onPress}
      >
        {onDismiss && (
          <motion.button
            type="button"
            whileTap={{ scale: 0.9 }}
            onClick={(e) => {
              e.stopPropagation();
              onDismiss();
            }}
            className="absolute top-2 left-2 z-10 w-7 h-7 rounded-full bg-black/50 backdrop-blur-sm flex items-center justify-center text-white"
            aria-label="Not interested"
          >
            <X size={14} strokeWidth={2} />
          </motion.button>
        )}
        <img
          src={resolveImageUrl(item.image_url)}
          alt={item.title}
          className="w-full h-full object-cover"
          loading="lazy"
        />

        <motion.button
          whileTap={{ scale: 0.85 }}
          onClick={(e) => {
            e.stopPropagation();
            onBookmark?.();
          }}
          className="absolute top-2 right-2 w-8 h-8 rounded-full bg-white/90 backdrop-blur-sm flex items-center justify-center shadow-sm"
        >
          <Bookmark
            size={16}
            className={isSaved ? "text-phia-blue" : "text-phia-gray-400"}
            fill={isSaved ? "currentColor" : "none"}
            strokeWidth={isSaved ? 2 : 1.5}
          />
        </motion.button>

        {tasteFit !== undefined && (
          <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded-full bg-black/60 backdrop-blur-sm text-white text-[10px] font-medium">
            {Math.round(tasteFit * 100)}% match
          </div>
        )}

        {unlockCount !== undefined && unlockCount > 0 && (
          <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded-full bg-black/60 backdrop-blur-sm text-white text-[10px] font-medium">
            +{unlockCount} looks
          </div>
        )}
      </div>

      <div className="mt-2 px-0.5 cursor-pointer" onClick={onPress}>
        <p className="text-[10px] tracking-[0.15em] uppercase text-phia-gray-400">
          {item.brand}
        </p>
        <p className="text-sm text-phia-black truncate mt-0.5">{item.title}</p>
        <p className="text-sm font-medium text-phia-black mt-0.5">
          ${item.price}
        </p>
      </div>
    </div>
  );
}

export default ProductCard;
