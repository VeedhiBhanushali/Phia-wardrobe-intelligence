'use client';

interface ScoreDisplayProps {
  tasteFit: number;
  unlockCount: number;
  bestPrice: number;
}

export function ScoreDisplay({
  tasteFit,
  unlockCount,
  bestPrice,
}: ScoreDisplayProps) {
  const fitPct = Math.round(tasteFit * 100);

  return (
    <div className="grid grid-cols-3 rounded-2xl border border-phia-gray-200 p-4">
      <div className="flex flex-col items-center text-center">
        <span className="text-2xl font-semibold text-phia-orange tabular-nums">
          {fitPct}%
        </span>
        <span className="text-xs font-medium text-phia-black mt-0.5">
          Taste Fit
        </span>
        <span className="text-[10px] text-phia-gray-500">match score</span>
      </div>

      <div className="flex flex-col items-center text-center border-x border-phia-gray-200">
        <span className="text-2xl font-semibold text-phia-green tabular-nums">
          +{unlockCount}
        </span>
        <span className="text-xs font-medium text-phia-black mt-0.5">
          Outfit Unlocks
        </span>
        <span className="text-[10px] text-phia-gray-500">new combos</span>
      </div>

      <div className="flex flex-col items-center text-center">
        <span className="text-2xl font-semibold text-phia-blue tabular-nums">
          ${bestPrice}
        </span>
        <span className="text-xs font-medium text-phia-black mt-0.5">
          Best Price
        </span>
        <span className="text-[10px] text-phia-gray-500">lowest found</span>
      </div>
    </div>
  );
}

export default ScoreDisplay;
