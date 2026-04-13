"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Send, Loader2, Bookmark } from "lucide-react";
import { useStyleChat } from "@/lib/useStyleChat";
import { resolveImageUrl } from "@/lib/api";
import type { TasteProfile, WardrobeItem, ChatMessage, ChatBlock } from "@/lib/store";

interface ChatDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  tasteProfile: TasteProfile | null;
  wardrobeItems: WardrobeItem[];
  onSaveItem: (item: WardrobeItem) => void;
  isItemSaved: (itemId: string) => boolean;
  preloadItemId?: string | null;
  intentLabels?: string[];
  intentConfidence?: number;
}

function ChatProductCard({
  item,
  isSaved,
  onSave,
  index,
}: {
  item: WardrobeItem;
  isSaved: boolean;
  onSave: () => void;
  index: number;
}) {
  return (
    <motion.div
      className="w-[120px] shrink-0 flex flex-col"
      initial={{ opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: index * 0.08, ease: "easeOut" }}
    >
      <div className="relative aspect-[3/4] overflow-hidden rounded-xl bg-phia-gray-100">
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
            onSave();
          }}
          className="absolute top-1.5 right-1.5 w-6 h-6 rounded-full bg-white/90 backdrop-blur-sm flex items-center justify-center shadow-sm"
        >
          <Bookmark
            size={11}
            className={isSaved ? "text-phia-blue" : "text-phia-gray-400"}
            fill={isSaved ? "currentColor" : "none"}
            strokeWidth={isSaved ? 2 : 1.5}
          />
        </motion.button>
      </div>
      <div className="mt-1.5 px-0.5">
        <p className="text-[9px] tracking-[0.12em] uppercase text-phia-gray-400 leading-tight">
          {item.brand}
        </p>
        <p className="text-[11px] text-phia-black truncate mt-0.5 leading-tight">
          {item.title}
        </p>
        <p className="text-[11px] font-medium text-phia-black mt-0.5">
          ${item.price}
        </p>
      </div>
    </motion.div>
  );
}

function OutfitBundleCard({
  items,
  title,
}: {
  items: WardrobeItem[];
  title: string;
}) {
  return (
    <div className="rounded-xl border border-phia-gray-200 p-3 mt-3">
      <p className="text-xs font-medium text-phia-black mb-2">{title}</p>
      <div className="flex gap-2 overflow-x-auto hide-scrollbar">
        {items.map((item, i) => (
          <motion.div
            key={item.item_id}
            className="w-16 shrink-0"
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.25, delay: i * 0.07 }}
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
          </motion.div>
        ))}
      </div>
    </div>
  );
}

function StreamingCursor() {
  return (
    <span className="inline-block w-[2px] h-[14px] bg-phia-black/70 ml-0.5 align-middle animate-blink rounded-sm" />
  );
}

/** Renders a line of text with inline **bold** and *italic* markdown. */
function InlineMd({ text }: { text: string }) {
  // Split on **bold** and *italic* tokens (bold must come first to avoid overlap)
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        if (part.startsWith("*") && part.endsWith("*")) {
          return <em key={i}>{part.slice(1, -1)}</em>;
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

/** Renders assistant markdown text with bold/italic support and proper newlines. */
function MarkdownText({
  content,
  isStreaming,
  showCursor,
}: {
  content: string;
  isStreaming: boolean;
  showCursor: boolean;
}) {
  const lines = content.split("\n");
  return (
    <p className="text-sm leading-relaxed">
      {lines.map((line, i) => (
        <span key={i}>
          {i > 0 && <br />}
          <InlineMd text={line} />
        </span>
      ))}
      {isStreaming && showCursor && <StreamingCursor />}
    </p>
  );
}

function InlineItemStrip({
  items,
  isItemSaved,
  onSaveItem,
}: {
  items: WardrobeItem[];
  isItemSaved: (id: string) => boolean;
  onSaveItem: (item: WardrobeItem) => void;
}) {
  const deduped = [...new Map(items.map((i) => [i.item_id, i])).values()];
  if (!deduped.length) return null;
  return (
    <div className="mt-3 mb-1 -mx-4">
      <div className="flex gap-2.5 overflow-x-auto hide-scrollbar px-4 pb-1">
        {deduped.map((item, i) => (
          <ChatProductCard
            key={item.item_id}
            item={item}
            isSaved={isItemSaved(item.item_id)}
            onSave={() => onSaveItem(item)}
            index={i}
          />
        ))}
        <div className="w-4 shrink-0" />
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  isItemSaved,
  onSaveItem,
  isActiveStream,
}: {
  message: ChatMessage;
  isItemSaved: (id: string) => boolean;
  onSaveItem: (item: WardrobeItem) => void;
  isActiveStream: boolean;
}) {
  const isUser = message.role === "user";
  const blocks = message.blocks;

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[85%] bg-phia-black text-white rounded-2xl rounded-br-md px-4 py-2.5">
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  if (blocks && blocks.length > 0) {
    const lastBlockIdx = blocks.length - 1;
    return (
      <div className="flex justify-start mb-4">
        <div className="w-full text-phia-black">
          {blocks.map((block: ChatBlock, i: number) => {
            if (block.type === "text" && block.content) {
              const isLastBlock = i === lastBlockIdx;
              return (
                <MarkdownText
                  key={i}
                  content={block.content}
                  isStreaming={isActiveStream && isLastBlock}
                  showCursor={isActiveStream && isLastBlock}
                />
              );
            }
            if (block.type === "items" && block.items) {
              return (
                <InlineItemStrip
                  key={i}
                  items={block.items}
                  isItemSaved={isItemSaved}
                  onSaveItem={onSaveItem}
                />
              );
            }
            if (block.type === "outfit_bundle" && block.outfitBundle) {
              return (
                <OutfitBundleCard
                  key={i}
                  items={block.outfitBundle.items}
                  title={block.outfitBundle.title}
                />
              );
            }
            return null;
          })}
        </div>
      </div>
    );
  }

  // Fallback for messages without blocks (legacy / simple text)
  const dedupedItems = message.items
    ? [...new Map(message.items.map((i) => [i.item_id, i])).values()]
    : [];

  return (
    <div className="flex justify-start mb-4">
      <div className="w-full text-phia-black">
        {message.content && (
          <MarkdownText
            content={message.content}
            isStreaming={isActiveStream}
            showCursor={!dedupedItems.length}
          />
        )}
        {dedupedItems.length > 0 && (
          <InlineItemStrip
            items={dedupedItems}
            isItemSaved={isItemSaved}
            onSaveItem={onSaveItem}
          />
        )}
        {message.outfitBundle && (
          <OutfitBundleCard
            items={message.outfitBundle.items}
            title={message.outfitBundle.title}
          />
        )}
      </div>
    </div>
  );
}

const SUGGESTED_PROMPTS = [
  "What gaps should I fill first?",
  "Build me a work outfit for tomorrow",
  "What one item would be most versatile?",
];

export function ChatDrawer({
  isOpen,
  onClose,
  tasteProfile,
  wardrobeItems,
  onSaveItem,
  isItemSaved,
  preloadItemId,
  intentLabels = [],
  intentConfidence = 0,
}: ChatDrawerProps) {
  const { messages, sendMessage, isStreaming, clearChat } = useStyleChat(
    tasteProfile,
    wardrobeItems
  );
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (isOpen && preloadItemId && messages.length === 0) {
      sendMessage(`Tell me about item ${preloadItemId} — how does it fit my wardrobe?`);
    }
  }, [isOpen, preloadItemId, messages.length, sendMessage]);

  const handleSend = () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    sendMessage(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const lastAssistantIdx = messages.reduce(
    (acc, m, i) => (m.role === "assistant" ? i : acc),
    -1
  );

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ y: "100%" }}
          animate={{ y: 0 }}
          exit={{ y: "100%" }}
          transition={{ type: "spring", damping: 30, stiffness: 300 }}
          className="fixed inset-0 z-50 bg-white flex flex-col max-w-[430px] mx-auto shadow-[0_0_40px_rgba(0,0,0,0.06)]"
        >
          {/* Header */}
          <div className="border-b border-phia-gray-100">
            <div className="flex items-center justify-between px-4 py-3">
              <div>
                <h2 className="font-serif text-lg text-phia-black">Ask Phia</h2>
                <p className="text-[10px] text-phia-gray-400">
                  {wardrobeItems.length} items loaded
                </p>
              </div>
              <div className="flex gap-2">
                {messages.length > 0 && (
                  <button
                    onClick={clearChat}
                    className="text-xs text-phia-gray-400 px-2 py-1"
                  >
                    Clear
                  </button>
                )}
                <button
                  onClick={onClose}
                  className="w-8 h-8 rounded-full bg-phia-gray-50 flex items-center justify-center"
                >
                  <X size={16} className="text-phia-gray-600" />
                </button>
              </div>
            </div>

            {/* Session intent indicator */}
            {intentLabels.length > 0 && intentConfidence > 0.3 && (
              <div className="px-4 pb-2.5 flex items-center gap-2">
                <div className="flex items-center gap-1.5">
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-phia-blue opacity-50" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-phia-blue" />
                  </span>
                  <span className="text-[10px] font-medium uppercase tracking-wider text-phia-gray-400">
                    Session
                  </span>
                </div>
                <div className="flex gap-1.5">
                  {intentLabels.map((label) => (
                    <span
                      key={label}
                      className="rounded-full bg-phia-blue/10 text-phia-blue text-[10px] font-medium px-2 py-0.5"
                    >
                      {label}
                    </span>
                  ))}
                </div>
                <div className="ml-auto flex items-center gap-1">
                  <div className="h-1 w-12 rounded-full bg-phia-gray-100 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-phia-blue transition-all duration-500"
                      style={{ width: `${Math.round(intentConfidence * 100)}%` }}
                    />
                  </div>
                  <span className="text-[9px] tabular-nums text-phia-gray-400">
                    {Math.round(intentConfidence * 100)}%
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Messages */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <p className="font-serif text-xl text-phia-black mb-2">
                  Hi! I&apos;m your stylist.
                </p>
                <p className="text-sm text-phia-gray-400 mb-6 max-w-[280px]">
                  I have your full wardrobe and taste profile loaded. Ask me
                  anything about styling, gaps, or outfit ideas.
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {SUGGESTED_PROMPTS.map((prompt) => (
                    <button
                      key={prompt}
                      onClick={() => sendMessage(prompt)}
                      className="rounded-full border border-phia-gray-200 px-3 py-1.5 text-xs text-phia-gray-600 hover:bg-phia-gray-50 transition-colors"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg, i) => (
                <MessageBubble
                  key={i}
                  message={msg}
                  isItemSaved={isItemSaved}
                  onSaveItem={onSaveItem}
                  isActiveStream={isStreaming && i === lastAssistantIdx}
                />
              ))
            )}

            {/* Thinking indicator — only while waiting for first token */}
            {isStreaming && messages[messages.length - 1]?.role === "user" && (
              <motion.div
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex gap-1.5 items-center text-phia-gray-400 py-2"
              >
                <Loader2 size={13} className="animate-spin" />
                <span className="text-xs">Thinking…</span>
              </motion.div>
            )}
          </div>

          {/* Input */}
          <div className="border-t border-phia-gray-100 px-4 py-3">
            <div className="flex gap-2 items-center">
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about styling, gaps, outfits…"
                disabled={isStreaming}
                className="flex-1 rounded-full border border-phia-gray-200 px-4 py-2.5 text-sm text-phia-black placeholder:text-phia-gray-300 outline-none focus:border-phia-gray-400 transition-colors disabled:opacity-50"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isStreaming}
                className="w-10 h-10 rounded-full bg-phia-black text-white flex items-center justify-center disabled:opacity-30 transition-opacity shrink-0"
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default ChatDrawer;
