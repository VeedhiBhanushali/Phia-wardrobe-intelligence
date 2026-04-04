'use client';

interface OutfitUnlockBadgeProps {
  count: number;
}

export function OutfitUnlockBadge({ count }: OutfitUnlockBadgeProps) {
  return (
    <span className="absolute bottom-2 left-2 rounded-full bg-black/70 text-white text-[10px] font-medium px-2 py-0.5 leading-tight">
      {count > 0 ? `+${count} looks` : `${count} looks`}
    </span>
  );
}

export default OutfitUnlockBadge;
