"use client";

import { useEffect, useState } from "react";
import { StageCard } from "@/components/dashboard/StageCard";
import { LogTerminal } from "@/components/dashboard/LogTerminal";

const API = "http://localhost:8000/api";

type JobStatus = "idle" | "running" | "done" | "error";

interface ServerStatus {
    status: JobStatus;
    job_type: string;
    logs: string[];
}

const pipelineSteps = [
    { key: "download", step: "01", title: "Download", href: "/dashboard/download", color: "#7c3aed", description: "Pull models and datasets from the HuggingFace Hub using YAML config." },
    { key: "finetune", step: "02", title: "Fine-Tune", href: "/dashboard/finetune", color: "#06b6d4", description: "Train with LoRA or QLoRA. Configure rank, alpha, learning rate and epochs." },
    { key: "align", step: "03", title: "Align", href: "/dashboard/align", color: "#10b981", description: "Align model to human preferences using DPO or GDPO with chosen/rejected pairs." },
];

export default function DashboardOverview() {
    const [server, setServer] = useState<ServerStatus>({ status: "idle", job_type: "", logs: [] });

    useEffect(() => {
        const poll = async () => {
            try {
                const res = await fetch(`${API}/status`);
                if (res.ok) setServer(await res.json());
            } catch { }
        };
        poll();
        const id = setInterval(poll, 3000);
        return () => clearInterval(id);
    }, []);

    return (
        <div className="p-8 max-w-5xl mx-auto">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-black text-white">Pipeline Dashboard</h1>
                <p className="text-white/50 mt-1">
                    Run the full LexAlign pipeline from your browser — Download → Fine-tune → Align.
                </p>
            </div>

            {/* Status banner */}
            {server.status !== "idle" && (
                <div
                    className={`mb-6 flex items-center gap-3 rounded-xl border px-5 py-3.5 text-sm font-medium ${server.status === "running"
                        ? "border-amber-500/20 bg-amber-500/5 text-amber-400"
                        : server.status === "done"
                            ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-400"
                            : "border-red-500/20 bg-red-500/5 text-red-400"
                        }`}
                >
                    <span
                        className={`h-2 w-2 rounded-full ${server.status === "running"
                            ? "bg-amber-400 animate-pulse"
                            : server.status === "done"
                                ? "bg-emerald-400"
                                : "bg-red-400"
                            }`}
                    />
                    {server.status === "running"
                        ? `Running: ${server.job_type}`
                        : server.status === "done"
                            ? `Completed: ${server.job_type}`
                            : `Error in: ${server.job_type}`}
                </div>
            )}

            {/* Pipeline flow indicator */}
            <div className="mb-8 flex items-center justify-center gap-0">
                {pipelineSteps.map((s, i) => {
                    const isDone = server.status === "done" && server.job_type === s.key;
                    const isRunning = server.status === "running" && server.job_type === s.key;

                    return (
                        <div key={s.key} className="flex items-center">
                            <a
                                href={s.href}
                                className="flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-all hover:scale-105"
                                style={{
                                    borderColor: isDone ? "#10b981" : isRunning ? "#f59e0b" : `${s.color}30`,
                                    background: isDone ? "rgba(16,185,129,0.1)" : isRunning ? "rgba(245,158,11,0.1)" : `${s.color}08`,
                                    color: isDone ? "#10b981" : isRunning ? "#f59e0b" : s.color,
                                }}
                            >
                                {isDone ? (
                                    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                                        <path d="M5 13l4 4L19 7" />
                                    </svg>
                                ) : isRunning ? (
                                    <span className="h-2.5 w-2.5 rounded-full bg-amber-400 animate-pulse" />
                                ) : (
                                    <span className="text-xs font-mono opacity-60">{s.step}</span>
                                )}
                                {s.title}
                            </a>
                            {i < pipelineSteps.length - 1 && (
                                <div className="w-8 h-px bg-white/10 mx-1" />
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Stage cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                {pipelineSteps.map(s => (
                    <StageCard
                        key={s.key}
                        step={s.step}
                        title={s.title}
                        description={s.description}
                        status={server.job_type === s.key ? server.status : "idle"}
                        href={s.href}
                        accentColor={s.color}
                    />
                ))}
            </div>

            {/* Live log */}
            {server.logs.length > 0 && (
                <div>
                    <h2 className="text-sm font-semibold text-white/40 uppercase tracking-widest mb-3">
                        Last Job Output
                    </h2>
                    <LogTerminal logs={server.logs} isRunning={server.status === "running"} />
                </div>
            )}

            {/* Quick start */}
            {server.status === "idle" && server.logs.length === 0 && (
                <div className="rounded-xl border border-dashed border-white/10 p-10 text-center">
                    <div className="mb-4">
                        <div className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-violet-500/10 mb-3">
                            <svg className="h-6 w-6 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                                <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 12l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                        </div>
                    </div>
                    <p className="text-white/40 text-sm font-medium">No job has run yet.</p>
                    <p className="text-white/20 text-xs mt-1 mb-4">Start with Step 01 → Download to pull a model from HuggingFace.</p>
                    <a href="/dashboard/download" className="btn-primary inline-flex items-center gap-2">
                        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                            <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M16 12l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Start Downloading
                    </a>
                </div>
            )}
        </div>
    );
}
