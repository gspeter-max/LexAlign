"use client";

import { cn } from "@/lib/utils";

interface ProgressBarProps {
    percent: number;
    label?: string;
    accentColor?: string;
    className?: string;
}

export function ProgressBar({ percent, label, accentColor = "#7c3aed", className }: ProgressBarProps) {
    const clampedPercent = Math.max(0, Math.min(100, percent));
    const isComplete = clampedPercent >= 100;
    const isRunning = clampedPercent > 0 && clampedPercent < 100;

    return (
        <div className={cn("space-y-2", className)}>
            {/* Label row */}
            <div className="flex items-center justify-between">
                {label && (
                    <span className="text-sm text-white/60 truncate max-w-[70%]">
                        {label}
                    </span>
                )}
                <span
                    className="text-sm font-mono font-semibold tabular-nums"
                    style={{ color: isComplete ? "#10b981" : accentColor }}
                >
                    {clampedPercent}%
                </span>
            </div>

            {/* Bar */}
            <div className="relative h-2 w-full overflow-hidden rounded-full bg-white/5">
                {/* Animated glow background */}
                {isRunning && (
                    <div
                        className="absolute inset-0 rounded-full opacity-30 animate-pulse"
                        style={{ background: `linear-gradient(90deg, transparent, ${accentColor}40, transparent)` }}
                    />
                )}

                {/* Fill */}
                <div
                    className="h-full rounded-full transition-all duration-700 ease-out relative"
                    style={{
                        width: `${clampedPercent}%`,
                        background: isComplete
                            ? "linear-gradient(90deg, #10b981, #34d399)"
                            : `linear-gradient(90deg, ${accentColor}, ${accentColor}cc)`,
                        boxShadow: isRunning
                            ? `0 0 12px ${accentColor}60`
                            : isComplete
                                ? "0 0 12px rgba(16, 185, 129, 0.4)"
                                : "none",
                    }}
                >
                    {/* Shimmer effect while running */}
                    {isRunning && (
                        <div
                            className="absolute inset-0 rounded-full"
                            style={{
                                background: "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%)",
                                animation: "shimmer 2s infinite",
                            }}
                        />
                    )}
                </div>
            </div>

            {/* Inline CSS for shimmer animation */}
            <style jsx>{`
                @keyframes shimmer {
                    0% { transform: translateX(-100%); }
                    100% { transform: translateX(200%); }
                }
            `}</style>
        </div>
    );
}
