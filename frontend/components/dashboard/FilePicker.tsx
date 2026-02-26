"use client";

import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

const API = "http://localhost:8000/api";

interface FileEntry {
    name: string;
    path: string;
    size_mb: number;
}

interface FilePickerProps {
    type: "models" | "datasets" | "checkpoints";
    value: string;
    onChange: (path: string) => void;
    label?: string;
    placeholder?: string;
    accentColor?: string;
    className?: string;
}

export function FilePicker({
    type,
    value,
    onChange,
    label,
    placeholder = "Select or type a path...",
    accentColor = "#7c3aed",
    className,
}: FilePickerProps) {
    const [files, setFiles] = useState<FileEntry[]>([]);
    const [loading, setLoading] = useState(false);
    const [open, setOpen] = useState(false);
    const [error, setError] = useState(false);

    const fetchFiles = async () => {
        setLoading(true);
        setError(false);
        try {
            const res = await fetch(`${API}/files/${type}`);
            if (res.ok) {
                setFiles(await res.json());
            } else {
                setError(true);
            }
        } catch {
            setError(true);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchFiles();
    }, [type]);

    return (
        <div className={cn("relative", className)}>
            {label && (
                <label className="block text-sm font-semibold text-white/60 mb-2">{label}</label>
            )}

            <div className="flex gap-2">
                {/* Input with dropdown trigger */}
                <div className="relative flex-1">
                    <input
                        type="text"
                        value={value}
                        onChange={(e) => onChange(e.target.value)}
                        placeholder={placeholder}
                        className="input-field pr-10"
                    />
                    <button
                        type="button"
                        onClick={() => {
                            setOpen(!open);
                            if (!open) fetchFiles();
                        }}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-white/10 transition-colors"
                        title="Browse files"
                    >
                        <svg className="h-4 w-4 text-white/40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                            <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                        </svg>
                    </button>
                </div>

                {/* Refresh button */}
                <button
                    type="button"
                    onClick={fetchFiles}
                    className="btn-ghost flex items-center gap-1.5 text-xs"
                    title="Refresh file list"
                >
                    <svg className={cn("h-3.5 w-3.5", loading && "animate-spin")} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                        <path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Scan
                </button>
            </div>

            {/* Dropdown */}
            {open && (
                <div className="absolute z-50 mt-1 w-full rounded-xl border border-white/10 bg-[#0d0d1a] backdrop-blur-xl shadow-2xl overflow-hidden">
                    {loading ? (
                        <div className="p-4 text-center text-sm text-white/30">
                            <span className="inline-block animate-spin mr-2">⟳</span>
                            Scanning...
                        </div>
                    ) : error ? (
                        <div className="p-4 text-center text-sm text-white/30">
                            <p className="text-red-400/60 mb-1">Could not connect to API</p>
                            <p className="text-xs text-white/20">Type a path manually instead</p>
                        </div>
                    ) : files.length === 0 ? (
                        <div className="p-4 text-center text-sm text-white/30">
                            <p className="mb-1">No {type} found</p>
                            <p className="text-xs text-white/20">Download some first, then refresh</p>
                        </div>
                    ) : (
                        <div className="max-h-48 overflow-y-auto">
                            {files.map((file) => (
                                <button
                                    key={file.path}
                                    type="button"
                                    onClick={() => {
                                        onChange(file.path);
                                        setOpen(false);
                                    }}
                                    className={cn(
                                        "w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-white/5 transition-colors border-b border-white/5 last:border-0",
                                        value === file.path && "bg-white/5"
                                    )}
                                >
                                    <div className="flex items-center gap-2.5 min-w-0">
                                        <svg className="h-4 w-4 shrink-0" style={{ color: accentColor }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                                            <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                                        </svg>
                                        <div className="min-w-0">
                                            <div className="text-sm font-medium text-white truncate">{file.name}</div>
                                            <div className="text-xs text-white/30 font-mono truncate">{file.path}</div>
                                        </div>
                                    </div>
                                    <span className="text-xs text-white/30 font-mono shrink-0 ml-3">
                                        {file.size_mb} MB
                                    </span>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Click outside to close */}
            {open && (
                <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
            )}
        </div>
    );
}
