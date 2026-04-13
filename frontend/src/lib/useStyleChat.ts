"use client";

import { useState, useCallback, useRef } from "react";
import { api } from "@/lib/api";
import type {
  TasteProfile,
  WardrobeItem,
  ChatMessage,
  ChatBlock,
} from "@/lib/store";

interface SSEEvent {
  type: "text" | "item_card" | "outfit_bundle" | "done";
  content?: string;
  item?: WardrobeItem;
  items?: WardrobeItem[];
  title?: string;
  occasion?: string;
}

export function useStyleChat(
  tasteProfile: TasteProfile | null,
  wardrobeItems: WardrobeItem[]
) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (text: string) => {
      if (isStreaming) return;

      const userMsg: ChatMessage = { role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);

      const apiMessages = [...messages, userMsg].map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await api.chat.stream({
          messages: apiMessages,
          wardrobe_item_ids: wardrobeItems.map((i) => i.item_id),
          taste_vector: tasteProfile?.taste_vector || [],
          taste_modes: tasteProfile?.taste_modes,
          occasion_vectors: tasteProfile?.occasion_vectors,
          trend_fingerprint: tasteProfile?.trend_fingerprint,
          anti_taste_vector: tasteProfile?.anti_taste_vector,
          style_attributes: tasteProfile?.style_attributes ?? {},
          price_tier: tasteProfile?.price_tier
            ? [tasteProfile.price_tier[0], tasteProfile.price_tier[1]]
            : undefined,
          aesthetic_attributes: tasteProfile?.aesthetic_attributes,
        });

        if (!response.ok || !response.body) {
          throw new Error("Chat request failed");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let assistantText = "";
        let blocks: ChatBlock[] = [];
        let currentTextSegment = "";

        const updateMessage = () => {
          const snapshot = [...blocks];
          const textSnapshot = assistantText;
          setMessages((prev) => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
              updated[lastIdx] = {
                ...updated[lastIdx],
                content: textSnapshot,
                blocks: snapshot,
              };
            } else {
              updated.push({
                role: "assistant",
                content: textSnapshot,
                blocks: snapshot,
              });
            }
            return updated;
          });
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const event: SSEEvent = JSON.parse(line.slice(6));

              if (event.type === "text" && event.content) {
                assistantText += event.content;
                currentTextSegment += event.content;
                const last = blocks[blocks.length - 1];
                if (last && last.type === "text") {
                  last.content = currentTextSegment;
                } else {
                  blocks.push({ type: "text", content: currentTextSegment });
                }
                updateMessage();
              } else if (event.type === "item_card" && event.item) {
                currentTextSegment = "";
                const last = blocks[blocks.length - 1];
                if (last && last.type === "items") {
                  last.items = [...(last.items || []), event.item as WardrobeItem];
                } else {
                  blocks.push({ type: "items", items: [event.item as WardrobeItem] });
                }
                updateMessage();
              } else if (event.type === "outfit_bundle" && event.items) {
                currentTextSegment = "";
                blocks.push({
                  type: "outfit_bundle",
                  outfitBundle: {
                    items: event.items as WardrobeItem[],
                    title: event.title || "Look",
                    occasion: event.occasion || "casual",
                  },
                });
                updateMessage();
              }
            } catch {
              // skip malformed events
            }
          }
        }

        updateMessage();
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: "Sorry, something went wrong. Please try again.",
            },
          ]);
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [tasteProfile, wardrobeItems, messages, isStreaming]
  );

  const clearChat = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setIsStreaming(false);
  }, []);

  return { messages, sendMessage, isStreaming, clearChat };
}
