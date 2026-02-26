import { cn } from "@/lib/utils";

type StageStatus = "idle" | "running" | "done" | "error";

interface StageCardProps {
    step: string;
    title: string;
    description: string;
    status: StageStatus;
    href: string;
    accentColor: string;
}

const statusConfig: Record<StageStatus, { label: string; dot: string; badge: string }> = {
    idle: { label: "Idle", dot: "bg-white/20", badge: "bg-white/5 text-white/40" },
    running: { label: "Running", dot: "bg-amber-400 animate-pulse", badge: "bg-amber-500/10 text-amber-400" },
    done: { label: "Done", dot: "bg-emerald-400", badge: "bg-emerald-500/10 text-emerald-400" },
    error: { label: "Error", dot: "bg-red-400", badge: "bg-red-500/10 text-red-400" },
};

export function StageCard({ step, title, description, status, href, accentColor }: StageCardProps) {
    const cfg = statusConfig[status];

    return (
        <a
            href={href}
            className="group relative flex flex-col gap-4 rounded-xl border border-white/10 bg-white/[0.02] p-6 hover:bg-white/[0.04] hover:border-white/20 transition-all"
        >
            {/* Glow on hover */}
            <div
                className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ background: `radial-gradient(ellipse 60% 60% at 50% 0%, ${accentColor}15, transparent)` }}
            />

            <div className="relative flex items-start justify-between">
                <div>
                    <span className="text-xs font-mono text-white/30">Step {step}</span>
                    <h3 className="text-lg font-bold text-white mt-1">{title}</h3>
                </div>
                <span className={cn("flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium", cfg.badge)}>
                    <span className={cn("h-1.5 w-1.5 rounded-full", cfg.dot)} />
                    {cfg.label}
                </span>
            </div>

            <p className="relative text-sm text-white/50">{description}</p>

            <div className="relative flex items-center gap-1.5 text-xs font-medium" style={{ color: accentColor }}>
                Open
                <svg className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                    <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
            </div>
        </a>
    );
}
