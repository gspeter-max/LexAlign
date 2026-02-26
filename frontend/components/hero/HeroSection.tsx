"use client";

import React from "react";
import { motion } from "motion/react";
import { Sparkles } from "@/components/ui/sparkles";
import { Spotlight } from "@/components/ui/spotlight";
import { cn } from "@/lib/utils";

const stats = [
    { label: "Training Methods", value: "4" },
    { label: "Quantization", value: "4/8-bit" },
    { label: "Alignment", value: "DPO+GDPO" },
    { label: "Models Supported", value: "Any LLM" },
];

export function HeroSection({ className }: { className?: string }) {
    return (
        <section
            className={cn(
                "relative min-h-screen flex flex-col items-center justify-center overflow-hidden px-4",
                className
            )}
        >
            {/* Spotlight effect */}
            <Spotlight className="inset-0 z-0" fill="rgba(124,58,237,0.7)" />

            {/* Radial gradient background */}
            <div
                className="absolute inset-0 z-0"
                style={{
                    background:
                        "radial-gradient(ellipse 80% 80% at 50% -20%, rgba(124,58,237,0.15), transparent)",
                }}
            />

            {/* Grid pattern */}
            <div
                className="absolute inset-0 z-0 opacity-20"
                style={{
                    backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' width='32' height='32' fill='none' stroke='rgb(148 163 184 / 0.15)'%3e%3cpath d='M0 .5H31.5V32'/%3e%3c/svg%3e")`,
                }}
            />

            <div className="relative z-10 flex flex-col items-center text-center max-w-5xl mx-auto">
                {/* Badge */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <span className="inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-500/10 px-4 py-1.5 text-sm font-medium text-violet-300 mb-8">
                        <span className="h-1.5 w-1.5 rounded-full bg-violet-400 animate-pulse" />
                        Open Source ML Pipeline
                    </span>
                </motion.div>

                {/* Headline */}
                <motion.h1
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                    className="text-6xl sm:text-7xl lg:text-8xl font-black tracking-tight text-white leading-none mb-6"
                >
                    <Sparkles sparkleColor="#7c3aed">
                        <span
                            className="bg-gradient-to-b from-white via-white/90 to-white/50 bg-clip-text text-transparent"
                        >
                            Lex
                        </span>
                    </Sparkles>
                    <span
                        className="bg-gradient-to-r from-violet-400 via-cyan-400 to-emerald-400 bg-clip-text text-transparent"
                    >
                        Align
                    </span>
                </motion.h1>

                {/* Subheading */}
                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    className="text-lg sm:text-xl text-white/60 max-w-2xl leading-relaxed mb-12"
                >
                    Download, fine-tune with LoRA/QLoRA, and align LLMs with DPO or GDPO —
                    all from a single declarative{" "}
                    <span className="text-violet-400 font-mono">YAML</span> config.
                </motion.p>

                {/* CTA buttons */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                    className="flex flex-col sm:flex-row items-center gap-4 mb-20"
                >
                    <a
                        href="#pipeline"
                        className="group relative inline-flex h-12 items-center justify-center gap-2 overflow-hidden rounded-xl bg-violet-600 px-8 font-semibold text-white shadow-[0_0_30px_rgba(124,58,237,0.5)] transition-all hover:bg-violet-500 hover:shadow-[0_0_50px_rgba(124,58,237,0.7)]"
                    >
                        <span>See the Pipeline</span>
                        <svg className="h-4 w-4 transition-transform group-hover:translate-x-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                            <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                    </a>
                    <a
                        href="https://github.com"
                        className="inline-flex h-12 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-8 font-semibold text-white backdrop-blur-sm transition-all hover:bg-white/10 hover:border-white/20"
                    >
                        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
                        </svg>
                        GitHub
                    </a>
                </motion.div>

                {/* Stats */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.4 }}
                    className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-white/5 rounded-2xl overflow-hidden border border-white/10"
                >
                    {stats.map((stat) => (
                        <div
                            key={stat.label}
                            className="flex flex-col items-center gap-1 bg-black/20 px-6 py-5 backdrop-blur-sm"
                        >
                            <span className="text-2xl font-black text-white">{stat.value}</span>
                            <span className="text-xs text-white/50 uppercase tracking-widest whitespace-nowrap">
                                {stat.label}
                            </span>
                        </div>
                    ))}
                </motion.div>
            </div>

            {/* Scroll indicator */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.5 }}
                className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
            >
                <span className="text-xs text-white/30 uppercase tracking-widest">Scroll</span>
                <motion.div
                    animate={{ y: [0, 6, 0] }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                    className="h-6 w-px bg-gradient-to-b from-white/30 to-transparent"
                />
            </motion.div>
        </section>
    );
}
