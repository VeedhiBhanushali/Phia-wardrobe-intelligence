"use client";

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Link, Loader2, RefreshCw, X, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import type { TasteProfile } from "@/lib/store";

interface AestheticBuilderCardProps {
  tasteProfile: TasteProfile | null;
  onTasteComplete: (profile: TasteProfile) => void;
}

const ALL_SWATCHES: { label: string; hex: string }[] = [
  { label: "Neutral Tones", hex: "#c8b9a6" },
  { label: "Bold Colors", hex: "#e53935" },
  { label: "Monochrome", hex: "#212121" },
  { label: "Warm Earth Tones", hex: "#8d6e63" },
  { label: "Cool Tones", hex: "#546e7a" },
];

const COLOR_SWATCHES: Record<string, string> = Object.fromEntries(
  ALL_SWATCHES.map((s) => [s.label, s.hex])
);

function UpdateForm({
  onSubmit,
  onCancel,
  hasProfile,
}: {
  onSubmit: (fd: FormData) => void;
  onCancel?: () => void;
  hasProfile: boolean;
}) {
  const [pinterestUrl, setPinterestUrl] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handlePinterest = () => {
    if (!pinterestUrl.trim()) return;
    const fd = new FormData();
    fd.append("pinterest_url", pinterestUrl.trim());
    onSubmit(fd);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    const fd = new FormData();
    files.slice(0, 10).forEach((f) => fd.append("images", f));
    onSubmit(fd);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Link size={14} className="text-phia-gray-400 shrink-0" />
        <input
          type="url"
          value={pinterestUrl}
          onChange={(e) => setPinterestUrl(e.target.value)}
          placeholder="Pinterest board URL"
          className="flex-1 rounded-lg border border-phia-gray-200 px-3 py-2.5 text-sm text-phia-black placeholder:text-phia-gray-300 outline-none focus:border-phia-gray-400 transition-colors"
        />
        <button
          onClick={handlePinterest}
          disabled={!pinterestUrl.trim()}
          className="rounded-lg bg-phia-black text-white px-4 py-2.5 text-sm font-medium disabled:opacity-30 shrink-0"
        >
          Go
        </button>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-phia-gray-200" />
        <span className="text-[11px] text-phia-gray-300">or</span>
        <div className="flex-1 h-px bg-phia-gray-200" />
      </div>

      <button
        onClick={() => fileInputRef.current?.click()}
        className="w-full rounded-xl border border-dashed border-phia-gray-200 py-5 flex flex-col items-center justify-center gap-2 text-phia-gray-400 hover:border-phia-gray-400 transition-colors"
      >
        <Upload size={18} />
        <span className="text-xs">Upload inspiration images</span>
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        onChange={handleFileUpload}
        className="hidden"
      />

      {hasProfile && onCancel && (
        <button
          onClick={onCancel}
          className="w-full text-xs text-phia-gray-400 py-1"
        >
          Cancel
        </button>
      )}
    </div>
  );
}

function AestheticModal({
  tasteProfile,
  onClose,
  onUpdate,
}: {
  tasteProfile: TasteProfile | null;
  onClose: () => void;
  onUpdate: (profile: TasteProfile) => void;
}) {
  const [updating, setUpdating] = useState(false);
  const [showUpdateForm, setShowUpdateForm] = useState(!tasteProfile);
  const [error, setError] = useState("");

  const attrs = tasteProfile?.aesthetic_attributes;
  const aestheticLabel = attrs?.silhouette?.label || "";
  const colorStory = attrs?.color_story?.label || "";
  const formality = attrs?.formality?.label || "";
  const priceTier = tasteProfile?.price_tier;
  const displayTrends = tasteProfile?.display_trends || tasteProfile?.trend_fingerprint;
  const topTrends = displayTrends ? Object.entries(displayTrends) : [];
  const styleSummary = tasteProfile?.style_summary ?? [];

  const preferredSignals = styleSummary.filter(e => e.direction === "prefers");
  const avoidedSignals = styleSummary.filter(e => e.direction === "avoids");

  // Build a personal style narrative from extracted attributes
  const styleNarrative = (() => {
    if (!tasteProfile) return "";
    const parts: string[] = [];
    if (aestheticLabel && colorStory) {
      const colorMap: Record<string, string> = {
        "Neutral Tones": "neutral, understated tones",
        "Bold Colors": "bold, saturated color",
        "Monochrome": "a clean monochrome palette",
        "Warm Earth Tones": "warm earthy tones",
        "Cool Tones": "cool, muted tones",
      };
      parts.push(`You dress in ${aestheticLabel.toLowerCase()} silhouettes and gravitate toward ${colorMap[colorStory] || colorStory.toLowerCase()}.`);
    }
    const materialPref = preferredSignals.find(e => e.key.startsWith("material_"));
    const vibePref = preferredSignals.find(e => e.key.startsWith("vibe_"));
    const brandPref = preferredSignals.find(e => e.key === "branding_minimal");
    const details: string[] = [];
    if (materialPref) details.push(materialPref.label.toLowerCase());
    if (vibePref) details.push(vibePref.label.toLowerCase() + " pieces");
    if (brandPref) details.push("clean, logo-free styling");
    if (formality && formality !== "Casual") details.push(formality.toLowerCase() + " occasions");
    if (details.length > 0) {
      parts.push(`Your wardrobe leans toward ${details.slice(0, 2).join(" and ")}.`);
    }
    return parts.join(" ");
  })();

  const handleSubmit = async (formData: FormData) => {
    setError("");
    setUpdating(true);
    try {
      const result = await api.taste.extract(formData);
      const normalizedProfile: TasteProfile = {
        ...result,
        taste_modes: result.taste_modes ?? [],
        occasion_vectors: result.occasion_vectors ?? {},
        trend_fingerprint: result.trend_fingerprint ?? {},
        display_trends: result.display_trends ?? result.trend_fingerprint ?? {},
        anti_taste_vector: result.anti_taste_vector ?? [],
        style_attributes: result.style_attributes ?? {},
        style_summary: result.style_summary ?? [],
        aesthetic_attributes: result.aesthetic_attributes ?? {},
        price_tier: result.price_tier ?? [0, 0],
      };
      onUpdate(normalizedProfile);
      setShowUpdateForm(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setUpdating(false);
    }
  };

  return (
    <motion.div
      initial={{ y: "100%" }}
      animate={{ y: 0 }}
      exit={{ y: "100%" }}
      transition={{ type: "spring", damping: 30, stiffness: 300 }}
      className="fixed inset-0 z-50 bg-white flex flex-col max-w-[430px] mx-auto shadow-[0_0_40px_rgba(0,0,0,0.08)]"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-4 border-b border-phia-gray-100">
        <div>
          <h2 className="font-serif text-lg text-phia-black">Your Aesthetic</h2>
          {tasteProfile && (
            <p className="text-[10px] text-phia-gray-400 mt-0.5">
              Extracted from your taste profile
            </p>
          )}
        </div>
        <button
          onClick={onClose}
          className="w-8 h-8 rounded-full bg-phia-gray-50 flex items-center justify-center"
        >
          <X size={16} className="text-phia-gray-600" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-5 space-y-6">
        {updating ? (
          <div className="flex flex-col items-center py-16">
            <Loader2 size={28} className="text-phia-black animate-spin" />
            <p className="text-sm text-phia-gray-400 mt-3">Analyzing your taste…</p>
          </div>
        ) : (
          <>
            {/* Profile summary — shown when not updating */}
            {tasteProfile && !showUpdateForm && (
              <>
                {/* Aesthetic label + color swatches */}
                {aestheticLabel && (
                  <div>
                    <p className="font-serif text-3xl text-phia-black leading-tight">
                      {aestheticLabel}
                    </p>
                    <div className="flex items-center gap-1.5 mt-3">
                      {ALL_SWATCHES.map((sw) => (
                        <div
                          key={sw.label}
                          className={`w-6 h-6 rounded-full border-2 shrink-0 transition-transform ${
                            sw.label === colorStory
                              ? "border-phia-black scale-110"
                              : "border-phia-gray-200"
                          }`}
                          style={{ backgroundColor: sw.hex }}
                          title={sw.label}
                        />
                      ))}
                      {colorStory && (
                        <span className="ml-1.5 text-[11px] text-phia-gray-400">
                          {colorStory}
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Personal narrative */}
                {styleNarrative && (
                  <div className="border-l-2 border-phia-gray-100 pl-3.5">
                    <p className="text-[13px] text-phia-gray-600 leading-relaxed">
                      {styleNarrative}
                    </p>
                  </div>
                )}

                {/* What you gravitate toward */}
                {preferredSignals.length > 0 && (
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-phia-gray-400 mb-2">
                      You gravitate toward
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {preferredSignals.map((entry) => (
                        <span
                          key={entry.key}
                          className="text-[11px] font-medium px-2.5 py-1 rounded-full bg-phia-gray-50 border border-phia-gray-100 text-phia-black"
                        >
                          {entry.label}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* What you consistently skip */}
                {avoidedSignals.length > 0 && (
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-phia-gray-400 mb-2">
                      You tend to skip
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {avoidedSignals.map((entry) => (
                        <span
                          key={entry.key}
                          className="text-[11px] text-phia-gray-400 px-2.5 py-1 rounded-full bg-phia-gray-50 border border-phia-gray-100"
                        >
                          {entry.label}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Trend fingerprint */}
                {topTrends.length > 0 && (
                  <div>
                    <p className="text-[10px] uppercase tracking-wider text-phia-gray-400 mb-2">
                      Aesthetic fingerprint
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {topTrends.map(([name]) => (
                        <span
                          key={name}
                          className="rounded-full bg-phia-black text-white text-[11px] px-3 py-1 font-medium"
                        >
                          {name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Price + formality footer */}
                {(priceTier || formality) && (
                  <div className="flex items-center gap-4 pt-1 border-t border-phia-gray-100">
                    {formality && (
                      <div>
                        <p className="text-[9px] uppercase tracking-wider text-phia-gray-400 mb-0.5">Occasion</p>
                        <p className="text-xs font-medium text-phia-black">{formality}</p>
                      </div>
                    )}
                    {priceTier && (
                      <div>
                        <p className="text-[9px] uppercase tracking-wider text-phia-gray-400 mb-0.5">Budget</p>
                        <p className="text-xs font-medium text-phia-black">${priceTier[0].toFixed(0)} – ${priceTier[1].toFixed(0)}</p>
                      </div>
                    )}
                    <p className="ml-auto text-[9px] text-phia-gray-300">Refines with each save</p>
                  </div>
                )}

                {/* Update CTA */}
                <button
                  onClick={() => setShowUpdateForm(true)}
                  className="w-full flex items-center justify-center gap-2 rounded-xl border border-phia-gray-200 py-3 text-sm text-phia-gray-600 hover:bg-phia-gray-50 transition-colors"
                >
                  <RefreshCw size={14} />
                  Update taste profile
                </button>
              </>
            )}

            {/* Input form */}
            {(!tasteProfile || showUpdateForm) && (
              <>
                {!tasteProfile && (
                  <p className="text-sm text-phia-gray-600">
                    Share your style inspiration to build your taste profile.
                    Upload photos or paste a Pinterest board link.
                  </p>
                )}
                <UpdateForm
                  onSubmit={handleSubmit}
                  onCancel={tasteProfile ? () => setShowUpdateForm(false) : undefined}
                  hasProfile={!!tasteProfile}
                />
                {error && (
                  <p className="text-xs text-red-500">{error}</p>
                )}
              </>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}

export function AestheticBuilderCard({
  tasteProfile,
  onTasteComplete,
}: AestheticBuilderCardProps) {
  const [modalOpen, setModalOpen] = useState(false);

  const attrs = tasteProfile?.aesthetic_attributes;
  const aestheticLabel = attrs?.silhouette?.label || "";
  const colorStory = attrs?.color_story?.label || "";

  return (
    <>
      <button
        type="button"
        onClick={() => setModalOpen(true)}
        className="rounded-2xl border border-phia-gray-200 bg-white text-left w-full transition-shadow hover:shadow-sm active:scale-[0.99] transition-transform"
      >
        <div className="px-4 pt-4 pb-4 flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <h3 className="font-serif text-lg text-phia-black leading-tight">
              Your Aesthetic
            </h3>

            {tasteProfile && aestheticLabel ? (
              <div className="mt-1.5">
                <p className="font-serif text-base font-medium text-phia-black leading-snug">
                  {aestheticLabel}
                </p>
                <div className="flex items-center gap-1.5 mt-2">
                  {ALL_SWATCHES.map((sw) => (
                    <div
                      key={sw.label}
                      className={`w-5 h-5 rounded-full border-2 shrink-0 ${
                        sw.label === colorStory
                          ? "border-phia-black scale-110"
                          : "border-phia-gray-200"
                      }`}
                      style={{ backgroundColor: sw.hex }}
                      title={sw.label}
                    />
                  ))}
                </div>
                {/* Style preference pills — only show what they like, not what they avoid */}
                {(tasteProfile.style_summary ?? []).filter(e => e.direction === "prefers").length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2.5">
                    {(tasteProfile.style_summary ?? [])
                      .filter(e => e.direction === "prefers")
                      .slice(0, 3)
                      .map((entry) => (
                        <span
                          key={entry.key}
                          className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-phia-gray-100 text-phia-gray-600"
                        >
                          {entry.label}
                        </span>
                      ))}
                  </div>
                )}
              </div>
            ) : (
              <p className="text-[11px] text-phia-gray-400 mt-0.5">
                build your taste profile
              </p>
            )}
          </div>

          <ChevronRight size={16} className="text-phia-gray-300 mt-0.5 shrink-0" />
        </div>
      </button>

      <AnimatePresence>
        {modalOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-black/20"
              onClick={() => setModalOpen(false)}
            />
            <AestheticModal
              tasteProfile={tasteProfile}
              onClose={() => setModalOpen(false)}
              onUpdate={(profile) => {
                onTasteComplete(profile);
                setModalOpen(false);
              }}
            />
          </>
        )}
      </AnimatePresence>
    </>
  );
}

export default AestheticBuilderCard;
