"use client";

import { useCallback } from "react";
import { api } from "@/lib/api";

export type EventType = "impression" | "click" | "save" | "dismiss" | "skip";
export type LogModule =
  | "wardrobe_gap"
  | "wardrobe_grid"
  | "catalog"
  | "product_detail"
  | "search"
  | "complete_the_look"
  | "top_pick"
  | "taste_match"
  | "foryou"
  | "occasion_row";

export function useEventLog(userId: string) {
  const log = useCallback(
    (
      eventType: EventType,
      module: LogModule,
      itemId: string,
      extra: { score?: number; unlock_count?: number; taste_score?: number } = {}
    ) => {
      if (!userId) return;

      api.events.log({
        user_id: userId,
        event_type: eventType,
        module,
        item_id: itemId,
        ...extra,
      }).catch(() => {});
    },
    [userId]
  );

  return { log };
}
