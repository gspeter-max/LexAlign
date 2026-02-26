"use client";

import React, { useEffect, useState, useRef } from "react";
import { cn } from "@/lib/utils";

interface Sparkle {
    id: string;
    x: string;
    y: string;
    color: string;
    delay: number;
    scale: number;
    lifespan: number;
}

function generateSparkle(color: string): Sparkle {
    return {
        id: Math.random().toString(36).slice(2),
        x: `${Math.random() * 100}%`,
        y: `${Math.random() * 100}%`,
        color,
        delay: Math.random() * 1000,
        scale: Math.random() * 0.75 + 0.5,
        lifespan: Math.random() * 1000 + 500,
    };
}

interface SparklesProps {
    children?: React.ReactNode;
    className?: string;
    sparkleColor?: string;
    minSize?: number;
    maxSize?: number;
}

export function Sparkles({
    children,
    className,
    sparkleColor = "#7c3aed",
}: SparklesProps) {
    const [sparkles, setSparkles] = useState<Sparkle[]>([]);
    const prefersReducedMotion = useRef(false);

    useEffect(() => {
        prefersReducedMotion.current = window.matchMedia(
            "(prefers-reduced-motion: reduce)"
        ).matches;
    }, []);

    useEffect(() => {
        if (prefersReducedMotion.current) return;

        const generateSparkles = () => {
            const now = Date.now();
            setSparkles((prev) => {
                const fresh = prev.filter((s) => now - parseInt(s.id, 36) < s.lifespan);
                if (fresh.length < 7) {
                    return [...fresh, generateSparkle(sparkleColor)];
                }
                return fresh;
            });
        };

        const interval = setInterval(generateSparkles, 350);
        return () => clearInterval(interval);
    }, [sparkleColor]);

    return (
        <span className={cn("relative inline-block", className)}>
            {sparkles.map((sparkle) => (
                <span
                    key={sparkle.id}
                    className="pointer-events-none absolute animate-ping"
                    style={{
                        left: sparkle.x,
                        top: sparkle.y,
                        transform: `scale(${sparkle.scale})`,
                        animationDelay: `${sparkle.delay}ms`,
                        animationDuration: `${sparkle.lifespan}ms`,
                    }}
                >
                    <svg
                        width="10"
                        height="10"
                        viewBox="0 0 10 10"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M5 0L6.12 3.88L10 5L6.12 6.12L5 10L3.88 6.12L0 5L3.88 3.88L5 0Z"
                            fill={sparkle.color}
                        />
                    </svg>
                </span>
            ))}
            <span className="relative z-10">{children}</span>
        </span>
    );
}
