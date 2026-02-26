"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface LogTerminalProps {
    logs: string[];
    isRunning?: boolean;
    className?: string;
}

function colorLine(line: string): string {
    if (line.startsWith("❌") || line.includes("Error")) return "text-red-400";
    if (line.startsWith("✅") || line.startsWith("🎉")) return "text-emerald-400";
    if (line.startsWith("🚀") || line.startsWith("⬇️")) return "text-cyan-400";
    if (line.startsWith("📋") || line.startsWith("📂") || line.startsWith("🤖") || line.startsWith("🔧")) return "text-violet-400";
    if (line.startsWith("🏋️") || line.startsWith("🔐")) return "text-amber-400";
    if (line.startsWith("   [DRY RUN]")) return "text-yellow-400";
    if (line.startsWith("   ")) return "text-white/60";
    return "text-white/80";
}

export function LogTerminal({ logs, isRunning, className }: LogTerminalProps) {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    return (
        <div
            className={cn(
                "relative rounded-xl border border-white/10 bg-black/60 backdrop-blur-sm overflow-hidden",
                className
            )}
        >
            {/* Terminal header */}
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/10 bg-white/[0.02]">
                <span className="h-3 w-3 rounded-full bg-red-500/70" />
                <span className="h-3 w-3 rounded-full bg-yellow-500/70" />
                <span className="h-3 w-3 rounded-full bg-green-500/70" />
                <span className="ml-3 text-xs text-white/40 font-mono">pipeline output</span>
                {isRunning && (
                    <span className="ml-auto flex items-center gap-1.5 text-xs text-emerald-400">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        Running
                    </span>
                )}
            </div>

            {/* Log output */}
            <div className="h-72 overflow-y-auto p-4 font-mono text-sm space-y-0.5 scroll-smooth">
                {logs.length === 0 ? (
                    <p className="text-white/20 italic">Waiting for output...</p>
                ) : (
                    logs.map((line, i) => (
                        <div key={i} className={cn("leading-relaxed whitespace-pre-wrap break-all", colorLine(line))}>
                            <span className="text-white/20 select-none mr-2 text-xs">
                                {String(i + 1).padStart(3, "0")}
                            </span>
                            {line}
                        </div>
                    ))
                )}
                <div ref={bottomRef} />
            </div>
        </div>
    );
}
