"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const stages = [
    {
        href: "/dashboard",
        label: "Overview",
        icon: (
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <rect x="3" y="3" width="7" height="7" rx="1" />
                <rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" />
                <rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
        ),
        step: null,
    },
    {
        href: "/dashboard/download",
        label: "Download",
        icon: (
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 12l-4 4m0 0l-4-4m4 4V4" />
            </svg>
        ),
        step: "01",
        color: "text-violet-400",
    },
    {
        href: "/dashboard/finetune",
        label: "Fine-Tune",
        icon: (
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <path d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
        ),
        step: "02",
        color: "text-cyan-400",
    },
    {
        href: "/dashboard/align",
        label: "Align",
        icon: (
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
        ),
        step: "03",
        color: "text-emerald-400",
    },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className="flex flex-col w-56 shrink-0 border-r border-white/5 bg-black/20 backdrop-blur-sm">
            {/* Logo */}
            <div className="flex items-center gap-2.5 px-5 py-5 border-b border-white/5">
                <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center">
                    <span className="text-xs font-black text-white">L</span>
                </div>
                <span className="font-black text-base tracking-tight">LexAlign</span>
            </div>

            {/* Nav */}
            <nav className="flex-1 px-3 py-4 space-y-1">
                {stages.map(({ href, label, icon, step, color }) => {
                    const active = pathname === href;
                    return (
                        <Link
                            key={href}
                            href={href}
                            className={cn(
                                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all",
                                active
                                    ? "bg-white/10 text-white"
                                    : "text-white/50 hover:text-white hover:bg-white/5"
                            )}
                        >
                            <span className={cn(active ? "text-white" : color ?? "text-white/40")}>{icon}</span>
                            <span>{label}</span>
                            {step && (
                                <span
                                    className={cn(
                                        "ml-auto text-xs font-mono",
                                        active ? "text-white/60" : "text-white/20"
                                    )}
                                >
                                    {step}
                                </span>
                            )}
                        </Link>
                    );
                })}
            </nav>

            {/* Footer */}
            <div className="px-4 py-4 border-t border-white/5">
                <Link
                    href="/"
                    className="flex items-center gap-2 text-xs text-white/30 hover:text-white/60 transition-colors"
                >
                    <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                        <path d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                    </svg>
                    Back to homepage
                </Link>
            </div>
        </aside>
    );
}
