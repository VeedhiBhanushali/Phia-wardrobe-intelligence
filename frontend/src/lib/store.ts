"use client";

import { useState, useCallback, useEffect } from "react";

export interface TasteProfile {
  user_id: string;
  taste_vector: number[];
  taste_modes?: number[][];
  occasion_vectors?: Record<string, number[]>;
  trend_fingerprint?: Record<string, number>;
  display_trends?: Record<string, number>;
  anti_taste_vector?: number[];
  style_attributes?: Record<string, number>;
  style_summary?: Array<{ key: string; label: string; score: number; direction: string }>;
  aesthetic_attributes: Record<
    string,
    { label: string; confidence: number; scores: Record<string, number> }
  >;
  price_tier: [number, number];
}

export interface OccasionSection {
  occasion: string;
  label: string;
  items: {
    item: WardrobeItem;
    taste_score: number;
    unlock_count: number;
    explanation: string;
  }[];
}

export interface WardrobeItem {
  item_id: string;
  title: string;
  brand: string;
  category: string;
  slot: string;
  price: number;
  image_url: string;
  save_id?: string;
  unlock_count?: number;
}

export interface GapRec {
  item: WardrobeItem;
  unlock_count: number;
  taste_score: number;
  explanation: string;
  confidence: number;
}

export interface OutfitBundle {
  label: string;
  items: WardrobeItem[];
}

export interface SessionViewedItem {
  item_id: string;
  embedding: number[];
}

export interface ChatBlock {
  type: "text" | "items" | "outfit_bundle";
  content?: string;
  items?: WardrobeItem[];
  outfitBundle?: { items: WardrobeItem[]; title: string; occasion: string };
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  items?: WardrobeItem[];
  outfitBundle?: {
    items: WardrobeItem[];
    title: string;
    occasion: string;
  };
  blocks?: ChatBlock[];
}

export interface FeedData {
  completeYourCloset: Array<{
    item: WardrobeItem;
    taste_score: number;
    unlock_count: number;
    explanation: string;
  }>;
  yourAesthetic: Array<{
    item: WardrobeItem;
    taste_score: number;
    unlock_count: number;
  }>;
  completeYourOutfits: Array<{
    wardrobe_items: WardrobeItem[];
    catalog_addition: WardrobeItem | null;
    occasion: string;
    title: string;
    rationale: string;
    is_complete: boolean;
    harmony_score: number;
  }>;
  bestPricesOnSaves: Array<{
    item: WardrobeItem;
    price: number;
    savings_ratio: number;
  }>;
  occasionRows: OccasionSection[];
  wardrobeStats: Record<string, unknown>;
}

const STORAGE_KEYS = {
  tasteProfile: "phia_taste_profile",
  wardrobeItems: "phia_wardrobe_items",
  userId: "phia_user_id",
  skippedItemIds: "phia_skipped_item_ids",
} as const;

function safeGet<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") return fallback;
  try {
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : fallback;
  } catch {
    return fallback;
  }
}

function safeSet(key: string, value: unknown) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {}
}

function safeRemove(key: string) {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(key);
  } catch {}
}

function isValidTasteProfile(value: unknown): value is TasteProfile {
  if (!value || typeof value !== "object") return false;

  const profile = value as Partial<TasteProfile>;
  return (
    typeof profile.user_id === "string" &&
    profile.user_id.length > 0 &&
    Array.isArray(profile.taste_vector) &&
    profile.taste_vector.length > 0 &&
    Array.isArray(profile.price_tier) &&
    profile.price_tier.length === 2 &&
    !!profile.aesthetic_attributes &&
    typeof profile.aesthetic_attributes === "object"
  );
}

function isValidWardrobeItem(value: unknown): value is WardrobeItem {
  if (!value || typeof value !== "object") return false;

  const item = value as Partial<WardrobeItem>;
  return (
    typeof item.item_id === "string" &&
    typeof item.title === "string" &&
    typeof item.brand === "string" &&
    typeof item.slot === "string" &&
    typeof item.image_url === "string" &&
    typeof item.price === "number"
  );
}

function isValidSkippedList(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((x) => typeof x === "string");
}

export function useAppState() {
  const [tasteProfile, setTasteProfileState] = useState<TasteProfile | null>(null);
  const [wardrobeItems, setWardrobeItemsState] = useState<WardrobeItem[]>([]);
  const [skippedItemIds, setSkippedItemIdsState] = useState<string[]>([]);
  const [userId, setUserIdState] = useState<string>("");
  const [isHydrated, setIsHydrated] = useState(false);

  // Session state (not persisted — resets on reload)
  const [sessionViewedItems, setSessionViewedItems] = useState<SessionViewedItem[]>([]);
  const [sessionIntentVector, setSessionIntentVector] = useState<number[] | null>(null);
  const [sessionIntentConfidence, setSessionIntentConfidence] = useState(0);
  const [sessionIntentLabels, setSessionIntentLabels] = useState<string[]>([]);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatPreloadItemId, setChatPreloadItemId] = useState<string | null>(null);

  useEffect(() => {
    const storedProfile = safeGet<unknown>(STORAGE_KEYS.tasteProfile, null);
    const storedWardrobe = safeGet<unknown[]>(STORAGE_KEYS.wardrobeItems, []);
    const storedUserId = safeGet<string>(STORAGE_KEYS.userId, "");
    const storedSkipped = safeGet<unknown>(STORAGE_KEYS.skippedItemIds, []);

    if (isValidTasteProfile(storedProfile)) {
      setTasteProfileState(storedProfile);
      setUserIdState(storedProfile.user_id);
    } else {
      safeRemove(STORAGE_KEYS.tasteProfile);
      if (storedUserId) {
        setUserIdState(storedUserId);
      }
    }

    const cleanedWardrobe = storedWardrobe.filter(isValidWardrobeItem);
    setWardrobeItemsState(cleanedWardrobe);
    if (cleanedWardrobe.length !== storedWardrobe.length) {
      safeSet(STORAGE_KEYS.wardrobeItems, cleanedWardrobe);
    }

    if (!isValidTasteProfile(storedProfile) && !storedUserId) {
      safeRemove(STORAGE_KEYS.userId);
    }

    if (isValidSkippedList(storedSkipped)) {
      setSkippedItemIdsState(storedSkipped);
    } else {
      safeRemove(STORAGE_KEYS.skippedItemIds);
    }

    setIsHydrated(true);
  }, []);

  const setTasteProfile = useCallback((profile: TasteProfile | null) => {
    setTasteProfileState(profile);
    safeSet(STORAGE_KEYS.tasteProfile, profile);
    if (profile?.user_id) {
      setUserIdState(profile.user_id);
      safeSet(STORAGE_KEYS.userId, profile.user_id);
    }
  }, []);

  const addWardrobeItem = useCallback((item: WardrobeItem) => {
    setWardrobeItemsState((prev) => {
      if (prev.some((i) => i.item_id === item.item_id)) return prev;
      const next = [...prev, item];
      safeSet(STORAGE_KEYS.wardrobeItems, next);
      return next;
    });
    setSkippedItemIdsState((prev) => {
      if (!prev.includes(item.item_id)) return prev;
      const next = prev.filter((id) => id !== item.item_id);
      safeSet(STORAGE_KEYS.skippedItemIds, next);
      return next;
    });
  }, []);

  const removeWardrobeItem = useCallback((itemId: string) => {
    setWardrobeItemsState((prev) => {
      const next = prev.filter((i) => i.item_id !== itemId);
      safeSet(STORAGE_KEYS.wardrobeItems, next);
      return next;
    });
  }, []);

  const isItemSaved = useCallback(
    (itemId: string) => wardrobeItems.some((i) => i.item_id === itemId),
    [wardrobeItems]
  );

  const addSkippedItem = useCallback((itemId: string) => {
    setSkippedItemIdsState((prev) => {
      if (prev.includes(itemId)) return prev;
      const next = [...prev, itemId];
      safeSet(STORAGE_KEYS.skippedItemIds, next);
      return next;
    });
  }, []);

  const addViewedItem = useCallback((item: SessionViewedItem) => {
    setSessionViewedItems((prev) => {
      if (prev.some((v) => v.item_id === item.item_id)) return prev;
      const next = [...prev, item].slice(-10); // max 10
      return next;
    });
  }, []);

  const clearAll = useCallback(() => {
    setTasteProfileState(null);
    setWardrobeItemsState([]);
    setSkippedItemIdsState([]);
    setUserIdState("");
    setSessionViewedItems([]);
    setSessionIntentVector(null);
    setSessionIntentConfidence(0);
    setSessionIntentLabels([]);
    setChatOpen(false);
    setChatPreloadItemId(null);
    Object.values(STORAGE_KEYS).forEach((k) => {
      if (typeof window !== "undefined") localStorage.removeItem(k);
    });
  }, []);

  return {
    tasteProfile,
    setTasteProfile,
    wardrobeItems,
    addWardrobeItem,
    removeWardrobeItem,
    isItemSaved,
    skippedItemIds,
    addSkippedItem,
    userId,
    isHydrated,
    clearAll,
    // Session state
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
  };
}
