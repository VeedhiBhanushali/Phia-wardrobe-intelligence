"use client";

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Link, Loader2, Sparkles, X } from "lucide-react";
import { api } from "@/lib/api";
import type { TasteProfile } from "@/lib/store";

interface TasteOnboardingProps {
  onComplete: (profile: TasteProfile) => void;
}

type Stage = "input" | "processing" | "reveal";

function isValidExtractedProfile(value: unknown): value is TasteProfile {
  if (!value || typeof value !== "object") return false;

  const profile = value as Partial<TasteProfile>;
  return (
    typeof profile.user_id === "string" &&
    profile.user_id.length > 0 &&
    Array.isArray(profile.taste_vector) &&
    profile.taste_vector.length > 0 &&
    !!profile.aesthetic_attributes &&
    typeof profile.aesthetic_attributes === "object" &&
    Array.isArray(profile.price_tier) &&
    profile.price_tier.length === 2
  );
}

export function TasteOnboarding({ onComplete }: TasteOnboardingProps) {
  const [stage, setStage] = useState<Stage>("input");
  const [pinterestUrl, setPinterestUrl] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [profile, setProfile] = useState<TasteProfile | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const valid = files.filter((f) => f.type.startsWith("image/")).slice(0, 10);

    setUploadedFiles((prev) => [...prev, ...valid].slice(0, 10));

    valid.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setPreviews((prev) => [...prev, ev.target?.result as string].slice(0, 10));
      };
      reader.readAsDataURL(file);
    });
  };

  const removeImage = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
    setPreviews((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async () => {
    setError("");
    setStage("processing");

    try {
      const formData = new FormData();

      if (pinterestUrl) {
        formData.append("pinterest_url", pinterestUrl);
      } else if (uploadedFiles.length > 0) {
        uploadedFiles.forEach((file) => formData.append("images", file));
      } else {
        setError("Upload images or paste a Pinterest board URL.");
        setStage("input");
        return;
      }

      const result = await api.taste.extract(formData);
      const normalizedProfile = {
        ...result,
        taste_modes: result.taste_modes ?? [],
        occasion_vectors: result.occasion_vectors ?? {},
        trend_fingerprint: result.trend_fingerprint ?? {},
        anti_taste_vector: result.anti_taste_vector ?? [],
        aesthetic_attributes: result.aesthetic_attributes ?? {},
        price_tier: result.price_tier ?? [0, 0],
      };

      if (!isValidExtractedProfile(normalizedProfile)) {
        throw new Error("We couldn't extract a usable style profile. Try different images.");
      }

      setProfile(normalizedProfile);
      setStage("reveal");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setStage("input");
    }
  };

  const handleContinue = () => {
    if (profile) onComplete(profile);
  };

  return (
    <div className="px-4 py-6">
      <AnimatePresence mode="wait">
        {stage === "input" && (
          <motion.div
            key="input"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.25 }}
          >
            <div className="text-center mb-8">
              <h2 className="font-serif text-2xl text-phia-black mb-2">
                Let&apos;s learn your style
              </h2>
              <p className="text-sm text-phia-gray-400">
                Import your Pinterest board or upload images that
                represent your aesthetic
              </p>
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border border-phia-gray-200 p-4">
                <label className="flex items-center gap-2 text-sm font-medium text-phia-gray-900 mb-2">
                  <Link size={16} />
                  Pinterest board URL
                </label>
                <input
                  type="url"
                  value={pinterestUrl}
                  onChange={(e) => setPinterestUrl(e.target.value)}
                  placeholder="https://pinterest.com/username/board"
                  className="w-full rounded-xl border border-phia-gray-200 px-4 py-3 text-sm text-phia-black placeholder:text-phia-gray-300 outline-none focus:border-phia-gray-400 transition-colors"
                />
              </div>

              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-phia-gray-200" />
                <span className="text-xs text-phia-gray-400">or</span>
                <div className="flex-1 h-px bg-phia-gray-200" />
              </div>

              <div className="rounded-2xl border border-phia-gray-200 p-4">
                <label className="flex items-center gap-2 text-sm font-medium text-phia-gray-900 mb-3">
                  <Upload size={16} />
                  Upload inspiration images
                </label>

                {previews.length > 0 && (
                  <div className="grid grid-cols-5 gap-2 mb-3">
                    {previews.map((src, i) => (
                      <div key={i} className="relative aspect-square rounded-lg overflow-hidden">
                        <img
                          src={src}
                          alt=""
                          className="w-full h-full object-cover"
                        />
                        <button
                          onClick={() => removeImage(i)}
                          className="absolute top-0.5 right-0.5 w-5 h-5 rounded-full bg-black/60 flex items-center justify-center"
                        >
                          <X size={10} className="text-white" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full rounded-xl border-2 border-dashed border-phia-gray-200 py-6 flex flex-col items-center gap-2 text-phia-gray-400 hover:border-phia-gray-400 transition-colors"
                >
                  <Upload size={20} />
                  <span className="text-sm">
                    {previews.length > 0
                      ? `${previews.length}/10 images`
                      : "Tap to upload (up to 10)"}
                  </span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
            </div>

            {error && (
              <p className="text-sm text-red-500 mt-3 text-center">{error}</p>
            )}

            <button
              onClick={handleSubmit}
              disabled={!pinterestUrl && uploadedFiles.length === 0}
              className="w-full mt-6 rounded-full bg-phia-black text-white py-3.5 text-sm font-medium disabled:opacity-30 disabled:cursor-not-allowed transition-opacity"
            >
              Analyze my style
            </button>
          </motion.div>
        )}

        {stage === "processing" && (
          <motion.div
            key="processing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center py-20"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
            >
              <Loader2 size={32} className="text-phia-black" />
            </motion.div>
            <p className="mt-4 text-sm text-phia-gray-400">
              Extracting your aesthetic...
            </p>
          </motion.div>
        )}

        {stage === "reveal" && profile && (
          <motion.div
            key="reveal"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <div className="text-center mb-6">
              <Sparkles size={24} className="text-phia-black mx-auto mb-2" />
              <h2 className="font-serif text-2xl text-phia-black mb-1">
                Your aesthetic
              </h2>
              <p className="text-sm text-phia-gray-400">
                Here&apos;s what we found
              </p>
            </div>

            {profile.trend_fingerprint &&
              Object.keys(profile.trend_fingerprint).length > 0 && (
                <div className="mb-5">
                  <p className="text-[10px] uppercase tracking-[0.15em] text-phia-gray-400 mb-2">
                    Your style DNA
                  </p>
                  <div className="space-y-2">
                    {Object.entries(profile.trend_fingerprint)
                      .slice(0, 5)
                      .map(([name, score], idx) => (
                        <motion.div
                          key={name}
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.08 * idx }}
                          className="rounded-2xl bg-phia-gray-50 p-4 flex items-center justify-between"
                        >
                          <p className="text-sm font-medium text-phia-black">
                            {name}
                          </p>
                          <div className="flex items-center gap-2">
                            <div className="w-20 h-1.5 rounded-full bg-phia-gray-200 overflow-hidden">
                              <div
                                className="h-full rounded-full bg-phia-black"
                                style={{ width: `${Math.round(score * 100)}%` }}
                              />
                            </div>
                            <span className="text-[11px] text-phia-gray-400 tabular-nums w-10 text-right">
                              {Math.round(score * 100)}%
                            </span>
                          </div>
                        </motion.div>
                      ))}
                  </div>
                </div>
              )}

            {profile.occasion_vectors &&
              Object.keys(profile.occasion_vectors).length > 0 && (
                <div className="mb-5">
                  <p className="text-[10px] uppercase tracking-[0.15em] text-phia-gray-400 mb-2">
                    We understand your style across
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {Object.keys(profile.occasion_vectors).map((occ, idx) => {
                      const labels: Record<string, string> = {
                        work: "Work",
                        casual: "Everyday",
                        evening: "Going out",
                        weekend: "Weekend",
                        special: "Special occasions",
                      };
                      return (
                        <motion.span
                          key={occ}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: 0.08 * idx + 0.3 }}
                          className="rounded-full bg-phia-black text-white text-xs px-3 py-1.5 font-medium"
                        >
                          {labels[occ] ?? occ}
                        </motion.span>
                      );
                    })}
                  </div>
                </div>
              )}

            <div className="space-y-3">
              {Object.entries(profile.aesthetic_attributes ?? {}).map(
                ([key, attr], idx) => (
                  <motion.div
                    key={key}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 * idx + 0.5 }}
                    className="rounded-2xl bg-phia-gray-50 p-4 flex items-center justify-between"
                  >
                    <div>
                      <p className="text-[10px] uppercase tracking-[0.15em] text-phia-gray-400 mb-0.5">
                        {key.replace("_", " ")}
                      </p>
                      <p className="text-sm font-medium text-phia-black">
                        {attr.label}
                      </p>
                    </div>
                    <div className="text-xs text-phia-gray-400">
                      {Math.round(attr.confidence * 100)}%
                    </div>
                  </motion.div>
                )
              )}
            </div>

            {Object.keys(profile.aesthetic_attributes ?? {}).length === 0 &&
              Object.keys(profile.trend_fingerprint ?? {}).length === 0 && (
              <p className="text-sm text-phia-gray-400 text-center">
                We couldn&apos;t confidently read your aesthetic from that input.
              </p>
            )}

            <button
              onClick={handleContinue}
              className="w-full mt-6 rounded-full bg-phia-black text-white py-3.5 text-sm font-medium"
            >
              See your wardrobe
            </button>

            <button
              onClick={() => {
                setProfile(null);
                setStage("input");
              }}
              className="w-full mt-3 text-sm text-phia-gray-400"
            >
              Try again
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default TasteOnboarding;
