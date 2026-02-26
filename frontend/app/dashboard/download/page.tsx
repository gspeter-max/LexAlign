"use client";

import { useState } from "react";
import { LogTerminal } from "@/components/dashboard/LogTerminal";
import { ProgressBar } from "@/components/dashboard/ProgressBar";

const API = "http://localhost:8000/api";

interface ModelEntry { repo: string; files: string; output_dir: string }
interface DatasetEntry { repo: string; files: string; output_dir: string }

function parseSSEEvent(data: string): { type: string;[key: string]: any } | null {
    try {
        const parsed = JSON.parse(data);
        if (parsed && typeof parsed === "object" && parsed.type) return parsed;
    } catch { }
    return null;
}

export default function DownloadPage() {
    const [hfToken, setHfToken] = useState("");
    const [models, setModels] = useState<ModelEntry[]>([{ repo: "gpt2", files: "*.json, *.bin", output_dir: "./models/gpt2" }]);
    const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
    const [dryRun, setDryRun] = useState(true);
    const [modelsOnly, setModelsOnly] = useState(false);
    const [datasetsOnly, setDatasetsOnly] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState({ current: 0, total: 0, label: "", percent: 0 });

    const addModel = () => setModels(p => [...p, { repo: "", files: "*", output_dir: "./models/" }]);
    const addDataset = () => setDatasets(p => [...p, { repo: "", files: "*", output_dir: "./data/" }]);
    const removeModel = (i: number) => setModels(p => p.filter((_, idx) => idx !== i));
    const removeDataset = (i: number) => setDatasets(p => p.filter((_, idx) => idx !== i));

    const run = async () => {
        setLogs([]);
        setRunning(true);
        setProgress({ current: 0, total: 0, label: "", percent: 0 });

        const body = {
            hf_token: hfToken,
            models: models.map(m => ({ repo: m.repo, files: m.files.split(",").map(f => f.trim()), output_dir: m.output_dir })),
            datasets: datasets.map(d => ({ repo: d.repo, files: d.files.split(",").map(f => f.trim()), output_dir: d.output_dir })),
            dry_run: dryRun,
            models_only: modelsOnly,
            datasets_only: datasetsOnly,
        };

        try {
            const res = await fetch(`${API}/download`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });

            const reader = res.body!.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() ?? "";
                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6).trim();
                        if (data === "[DONE]") { setRunning(false); return; }
                        if (!data) continue;

                        const event = parseSSEEvent(data);
                        if (event?.type === "progress") {
                            setProgress({
                                current: event.current,
                                total: event.total,
                                label: event.label,
                                percent: event.total > 0 ? Math.round((event.current / event.total) * 100) : 0,
                            });
                        } else {
                            setLogs(prev => [...prev, data]);
                        }
                    }
                }
            }
        } catch (e: any) {
            setLogs(prev => [...prev, `❌ Network error: ${e.message}`]);
        } finally {
            setRunning(false);
        }
    };

    return (
        <div className="p-8 max-w-4xl mx-auto">
            <div className="mb-8">
                <span className="text-xs font-mono text-violet-400">Step 01</span>
                <h1 className="text-3xl font-black text-white mt-1">Download</h1>
                <p className="text-white/50 mt-1">Pull models and datasets from the HuggingFace Hub.</p>
            </div>

            <div className="space-y-6">
                {/* HF Token */}
                <Field label="HuggingFace Token" required>
                    <input
                        type="password"
                        value={hfToken}
                        onChange={e => setHfToken(e.target.value)}
                        placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
                        className="input-field"
                    />
                </Field>

                {/* Models */}
                <div>
                    <div className="flex items-center justify-between mb-3">
                        <label className="text-sm font-semibold text-white/70">Models</label>
                        <button onClick={addModel} className="btn-ghost text-xs">+ Add model</button>
                    </div>
                    <div className="space-y-3">
                        {models.map((m, i) => (
                            <div key={i} className="grid grid-cols-3 gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-3">
                                <input value={m.repo} onChange={e => setModels(p => p.map((x, j) => j === i ? { ...x, repo: e.target.value } : x))} placeholder="repo/model-name" className="input-field" />
                                <input value={m.files} onChange={e => setModels(p => p.map((x, j) => j === i ? { ...x, files: e.target.value } : x))} placeholder="*.json, *.bin" className="input-field" />
                                <div className="flex gap-2">
                                    <input value={m.output_dir} onChange={e => setModels(p => p.map((x, j) => j === i ? { ...x, output_dir: e.target.value } : x))} placeholder="./models/name" className="input-field flex-1" />
                                    <button onClick={() => removeModel(i)} className="text-red-400/60 hover:text-red-400 transition-colors text-lg leading-none px-1">×</button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Datasets */}
                <div>
                    <div className="flex items-center justify-between mb-3">
                        <label className="text-sm font-semibold text-white/70">Datasets</label>
                        <button onClick={addDataset} className="btn-ghost text-xs">+ Add dataset</button>
                    </div>
                    <div className="space-y-3">
                        {datasets.map((d, i) => (
                            <div key={i} className="grid grid-cols-3 gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-3">
                                <input value={d.repo} onChange={e => setDatasets(p => p.map((x, j) => j === i ? { ...x, repo: e.target.value } : x))} placeholder="repo/dataset-name" className="input-field" />
                                <input value={d.files} onChange={e => setDatasets(p => p.map((x, j) => j === i ? { ...x, files: e.target.value } : x))} placeholder="*.json" className="input-field" />
                                <div className="flex gap-2">
                                    <input value={d.output_dir} onChange={e => setDatasets(p => p.map((x, j) => j === i ? { ...x, output_dir: e.target.value } : x))} placeholder="./data/name" className="input-field flex-1" />
                                    <button onClick={() => removeDataset(i)} className="text-red-400/60 hover:text-red-400 transition-colors text-lg leading-none px-1">×</button>
                                </div>
                            </div>
                        ))}
                        {datasets.length === 0 && <p className="text-xs text-white/20 italic">No datasets added</p>}
                    </div>
                </div>

                {/* Options */}
                <div className="flex flex-wrap gap-4">
                    <Toggle label="Dry Run" checked={dryRun} onChange={setDryRun} />
                    <Toggle label="Models Only" checked={modelsOnly} onChange={setModelsOnly} />
                    <Toggle label="Datasets Only" checked={datasetsOnly} onChange={setDatasetsOnly} />
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                    <button onClick={run} disabled={running || !hfToken} className="btn-primary" id="download-run-btn">
                        {running ? "Downloading..." : dryRun ? "Dry Run" : "Download"}
                    </button>
                </div>

                {/* Progress Bar */}
                {(running || progress.percent > 0) && (
                    <ProgressBar
                        percent={progress.percent}
                        label={progress.label ? `Downloading: ${progress.label}` : "Starting..."}
                        accentColor="#7c3aed"
                    />
                )}

                <LogTerminal logs={logs} isRunning={running} />
            </div>
        </div>
    );
}

function Field({ label, required, children }: { label: string; required?: boolean; children: React.ReactNode }) {
    return (
        <div>
            <label className="block text-sm font-semibold text-white/70 mb-2">
                {label} {required && <span className="text-violet-400">*</span>}
            </label>
            {children}
        </div>
    );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
    return (
        <label className="flex items-center gap-2.5 cursor-pointer group">
            <div
                onClick={() => onChange(!checked)}
                className={`relative h-5 w-9 rounded-full transition-colors ${checked ? "bg-violet-600" : "bg-white/10"}`}
            >
                <div className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-4" : "translate-x-0.5"}`} />
            </div>
            <span className="text-sm text-white/60 group-hover:text-white/80 transition-colors">{label}</span>
        </label>
    );
}
