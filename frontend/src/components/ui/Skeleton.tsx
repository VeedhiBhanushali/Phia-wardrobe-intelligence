"use client";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = "" }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse bg-phia-gray-100 rounded-xl ${className}`}
    />
  );
}

export function ProductCardSkeleton() {
  return (
    <div className="flex flex-col">
      <Skeleton className="aspect-[3/4] rounded-xl" />
      <div className="mt-2 px-0.5 space-y-1.5">
        <Skeleton className="h-2.5 w-16 rounded" />
        <Skeleton className="h-3.5 w-full rounded" />
        <Skeleton className="h-3.5 w-12 rounded" />
      </div>
    </div>
  );
}

export function ProductGridSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {Array.from({ length: count }).map((_, i) => (
        <ProductCardSkeleton key={i} />
      ))}
    </div>
  );
}

export function GapRecommendationSkeleton() {
  return (
    <div className="rounded-2xl bg-phia-gray-50 p-4 space-y-3">
      <Skeleton className="aspect-[4/5] rounded-2xl" />
      <Skeleton className="h-4 w-3/4 rounded" />
      <Skeleton className="h-3 w-1/2 rounded" />
      <div className="flex gap-2">
        <Skeleton className="h-10 flex-1 rounded-full" />
        <Skeleton className="h-10 w-28 rounded-full" />
      </div>
    </div>
  );
}

export function ProfileChipsSkeleton() {
  return (
    <div className="flex flex-wrap gap-2">
      {Array.from({ length: 4 }).map((_, i) => (
        <Skeleton key={i} className="h-7 w-24 rounded-full" />
      ))}
    </div>
  );
}

export function ScoreDisplaySkeleton() {
  return (
    <div className="grid grid-cols-3 rounded-2xl border border-phia-gray-200 p-4 gap-2">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="flex flex-col items-center gap-1.5">
          <Skeleton className="h-8 w-14 rounded" />
          <Skeleton className="h-3 w-16 rounded" />
          <Skeleton className="h-2.5 w-12 rounded" />
        </div>
      ))}
    </div>
  );
}

export function ProductDetailSkeleton() {
  return (
    <div>
      <Skeleton className="aspect-[3/4] rounded-b-3xl" />
      <div className="px-4 py-4 space-y-3">
        <Skeleton className="h-2.5 w-20 rounded" />
        <Skeleton className="h-5 w-3/4 rounded" />
        <Skeleton className="h-5 w-16 rounded" />
        <ScoreDisplaySkeleton />
        <Skeleton className="h-12 w-full rounded-full" />
      </div>
    </div>
  );
}

export default Skeleton;
