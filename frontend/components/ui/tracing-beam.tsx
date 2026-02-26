"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion, useTransform, useScroll, useSpring } from "motion/react";
import { cn } from "@/lib/utils";

interface TracingBeamProps {
    children: React.ReactNode;
    className?: string;
}

export function TracingBeam({ children, className }: TracingBeamProps) {
    const ref = useRef<HTMLDivElement>(null);
    const { scrollYProgress } = useScroll({
        target: ref,
        offset: ["start start", "end start"],
    });

    const contentRef = useRef<HTMLDivElement>(null);
    const [svgHeight, setSvgHeight] = useState(0);

    useEffect(() => {
        if (contentRef.current) {
            setSvgHeight(contentRef.current.offsetHeight);
        }
    }, []);

    const y1 = useSpring(
        useTransform(scrollYProgress, [0, 0.8], [50, svgHeight]),
        { stiffness: 500, damping: 90 }
    );
    const y2 = useSpring(
        useTransform(scrollYProgress, [0, 1], [50, svgHeight - 200]),
        { stiffness: 500, damping: 90 }
    );

    return (
        <motion.div
            ref={ref}
            className={cn("relative mx-auto h-full w-full max-w-4xl", className)}
        >
            <div className="absolute left-6 top-3">
                <motion.div
                    transition={{ duration: 0.2, delay: 0.5 }}
                    animate={{ boxShadow: "0px 4px 64px 0px rgba(124, 58, 237, 0.7)" }}
                    className="ml-[27px] flex h-4 w-4 items-center justify-center rounded-full border border-violet-500 shadow-sm"
                >
                    <div className="h-2 w-2 rounded-full border border-violet-300 bg-violet-500" />
                </motion.div>
                <svg
                    viewBox={`0 0 20 ${svgHeight}`}
                    width="20"
                    height={svgHeight}
                    className="ml-4 block"
                    aria-hidden="true"
                >
                    <motion.path
                        d={`M 1 0V -36 l 18 24 V ${svgHeight * 0.8} l -18 24V ${svgHeight}`}
                        fill="none"
                        stroke="#9091A0"
                        strokeOpacity="0.16"
                        className="stroke-1"
                    />
                    <motion.path
                        d={`M 1 0V -36 l 18 24 V ${svgHeight * 0.8} l -18 24V ${svgHeight}`}
                        fill="none"
                        stroke="url(#gradient)"
                        strokeWidth="1.25"
                        className="motion-reduce:hidden"
                    />
                    <defs>
                        <motion.linearGradient
                            id="gradient"
                            gradientUnits="userSpaceOnUse"
                            x1="0"
                            x2="0"
                            y1={y1}
                            y2={y2}
                        >
                            <stop stopColor="#7c3aed" stopOpacity="0" />
                            <stop stopColor="#7c3aed" />
                            <stop offset="0.325" stopColor="#06b6d4" />
                            <stop offset="1" stopColor="#06b6d4" stopOpacity="0" />
                        </motion.linearGradient>
                    </defs>
                </svg>
            </div>
            <div ref={contentRef}>{children}</div>
        </motion.div>
    );
}
