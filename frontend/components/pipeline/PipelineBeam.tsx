"use client";

import React, { forwardRef, useRef } from "react";
import { AnimatedBeam } from "@/components/ui/animated-beam";
import { cn } from "@/lib/utils";

const Circle = forwardRef<
    HTMLDivElement,
    { className?: string; children?: React.ReactNode; label?: string }
>(({ className, children, label }, ref) => {
    return (
        <div className="flex flex-col items-center gap-3">
            <div
                ref={ref}
                className={cn(
                    "z-10 flex h-16 w-16 items-center justify-center rounded-full border border-white/10 bg-white/5 p-3 shadow-[0_0_30px_rgba(124,58,237,0.3)] backdrop-blur-sm transition-all duration-300 hover:shadow-[0_0_50px_rgba(124,58,237,0.5)] hover:border-violet-500/50",
                    className
                )}
            >
                {children}
            </div>
            {label && (
                <span className="text-xs font-medium text-white/60 tracking-wide uppercase">
                    {label}
                </span>
            )}
        </div>
    );
});
Circle.displayName = "Circle";

export function PipelineBeam({
    className,
}: {
    className?: string;
}) {
    const containerRef = useRef<HTMLDivElement>(null);
    const downloadRef = useRef<HTMLDivElement>(null);
    const finetuneRef = useRef<HTMLDivElement>(null);
    const alignRef = useRef<HTMLDivElement>(null);

    return (
        <div
            className={cn(
                "relative flex items-center justify-center w-full px-8 py-12",
                className
            )}
            ref={containerRef}
        >
            <div className="flex w-full max-w-3xl items-center justify-between">
                {/* Download Node */}
                <Circle ref={downloadRef} label="Download">
                    <svg viewBox="0 0 24 24" fill="none" className="w-8 h-8">
                        <path
                            d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"
                            fill="#7c3aed"
                        />
                    </svg>
                </Circle>

                {/* Fine-tune Node */}
                <Circle ref={finetuneRef} label="Fine-tune">
                    <svg viewBox="0 0 24 24" fill="none" className="w-8 h-8">
                        <path
                            d="M12 2a7 7 0 017 7c0 2.62-1.44 4.9-3.57 6.15L14 22H10l-1.43-6.85A7 7 0 015 9a7 7 0 017-7zm0 2a5 5 0 00-5 5 5 5 0 005 5 5 5 0 005-5 5 5 0 00-5-5zm0 2a3 3 0 013 3 3 3 0 01-3 3 3 3 0 01-3-3 3 3 0 013-3z"
                            fill="#06b6d4"
                        />
                    </svg>
                </Circle>

                {/* Align Node */}
                <Circle ref={alignRef} label="Align">
                    <svg viewBox="0 0 24 24" fill="none" className="w-8 h-8">
                        <path
                            d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-12c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4zm0 6c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2z"
                            fill="#10b981"
                        />
                    </svg>
                </Circle>
            </div>

            {/* Beams */}
            <AnimatedBeam
                containerRef={containerRef}
                fromRef={downloadRef}
                toRef={finetuneRef}
                gradientStartColor="#7c3aed"
                gradientStopColor="#06b6d4"
                pathColor="rgba(255,255,255,0.1)"
                pathWidth={2}
                duration={4}
            />
            <AnimatedBeam
                containerRef={containerRef}
                fromRef={finetuneRef}
                toRef={alignRef}
                gradientStartColor="#06b6d4"
                gradientStopColor="#10b981"
                pathColor="rgba(255,255,255,0.1)"
                pathWidth={2}
                duration={4}
                delay={2}
            />
        </div>
    );
}
