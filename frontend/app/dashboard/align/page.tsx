"use client";

import { useState } from "react";
import { LogTerminal } from "@/components/dashboard/LogTerminal";
import { ProgressBar } from "@/components/dashboard/ProgressBar";
import { MetricsChart } from "@/components/dashboard/MetricsChart";
import { FilePicker } from "@/components/dashboard/FilePicker";

const API = "http://localhost:8000/api";

interface MetricPoint {
    epoch: number;
    loss: number;
    lr: number;
}

function parseSSEEvent(data: string): { type: string;[key: string]: any } | null {
    try {
        const parsed = JSON.parse(data);
        if (parsed && typeof parsed === "object" && parsed.type) return parsed;
    } catch { }
    return null;
}

export default function AlignPage() {
    const [form, setForm] = useState({
        model_path: "./checkpoints/finetuned",
        dataset_path: "./data/preferences",
        method: "dpo",
        device: "cpu",
        beta: 0.1,
        learning_rate: 1e-5,
        batch_size: 4,
        num_epochs: 1,
        prompt_field: "prompt",
        chosen_field: "chosen",
        rejected_field: "rejected",
        output_dir: "./checkpoints/aligned",
        use_lora: true,
        lora_r: 16,
        lora_alpha: 32,
        dry_run: true,
    });

    const [logs, setLogs] = useState<string[]>([]);
    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState({ epoch: 0, total_epochs: 0, percent: 0 });
    const [metrics, setMetrics] = useState<MetricPoint[]>([]);
    const set = (k: string, v: any) => setForm(p => ({ ...p, [k]: v }));

    const run = async () => {
        setLogs([]);
        setRunning(true);
        setProgress({ epoch: 0, total_epochs: 0, percent: 0 });
        setMetrics([]);

        try {
            const res = await fetch(`${API}/align`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(form),
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
                                epoch: event.epoch,
                                total_epochs: event.total_epochs,
                                percent: event.percent,
                            });
                        } else if (event?.type === "metrics") {
                            setMetrics(prev => [...prev, {
                                epoch: event.epoch,
                                loss: event.loss,
                                lr: event.lr,
                            }]);
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
                <span className="text-xs font-mono text-emerald-400">Step 03</span>
                <h1 className="text-3xl font-black text-white mt-1">Align</h1>
                <p className="text-white/50 mt-1">Align your model with human preferences using DPO or GDPO.</p>
            </div>

            <div className="grid grid-cols-2 gap-6 mb-6">
                <FilePicker
                    type="checkpoints"
                    value={form.model_path}
                    onChange={v => set("model_path", v)}
                    label="Model Path"
                    placeholder="./checkpoints/finetuned"
                    accentColor="#10b981"
                />
                <FilePicker
                    type="datasets"
                    value={form.dataset_path}
                    onChange={v => set("dataset_path", v)}
                    label="Dataset Path"
                    placeholder="./data/preferences"
                    accentColor="#10b981"
                />
                <Field label="Output Dir">
                    <input className="input-field" value={form.output_dir} onChange={e => set("output_dir", e.target.value)} />
                </Field>
                <Field label="Alignment Method">
                    <div className="flex gap-2">
                        {["dpo", "gdpo"].map(m => (
                            <button key={m} onClick={() => set("method", m)}
                                className={`flex-1 rounded-lg border py-2 text-sm font-semibold uppercase tracking-wide transition-all ${form.method === m
                                    ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                                    : "border-white/10 text-white/40 hover:border-white/20"
                                    }`}>{m}</button>
                        ))}
                    </div>
                </Field>
                <Field label="Device">
                    <div className="flex gap-2">
                        {["cpu", "cuda"].map(d => (
                            <button key={d} onClick={() => set("device", d)}
                                className={`flex-1 rounded-lg border py-2 text-sm font-semibold uppercase tracking-wide transition-all ${form.device === d
                                    ? "border-violet-500/50 bg-violet-500/10 text-violet-400"
                                    : "border-white/10 text-white/40 hover:border-white/20"
                                    }`}>{d}</button>
                        ))}
                    </div>
                </Field>
            </div>

            {/* Preference dataset fields */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5 mb-6">
                <h3 className="text-sm font-semibold text-white/60 uppercase tracking-widest mb-4">Dataset Fields</h3>
                <div className="grid grid-cols-3 gap-4">
                    <Field label="Prompt Field">
                        <input className="input-field" value={form.prompt_field} onChange={e => set("prompt_field", e.target.value)} />
                    </Field>
                    <Field label="Chosen Field">
                        <input className="input-field" value={form.chosen_field} onChange={e => set("chosen_field", e.target.value)} />
                    </Field>
                    <Field label="Rejected Field">
                        <input className="input-field" value={form.rejected_field} onChange={e => set("rejected_field", e.target.value)} />
                    </Field>
                </div>
            </div>

            {/* Training hyperparameters */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5 mb-6">
                <h3 className="text-sm font-semibold text-white/60 uppercase tracking-widest mb-4">Hyperparameters</h3>
                <div className="grid grid-cols-2 gap-4">
                    <SliderField label="Beta (KL penalty)" value={form.beta} min={0.01} max={1} step={0.01} onChange={v => set("beta", v)} accent="#10b981" />
                    <SliderField label="Learning Rate" value={form.learning_rate} min={1e-7} max={1e-3} step={1e-7} onChange={v => set("learning_rate", v)} accent="#10b981" display={v => v.toExponential(1)} />
                    <SliderField label="Batch Size" value={form.batch_size} min={1} max={32} step={1} onChange={v => set("batch_size", v)} accent="#10b981" />
                    <SliderField label="Epochs" value={form.num_epochs} min={1} max={10} step={1} onChange={v => set("num_epochs", v)} accent="#10b981" />
                    <SliderField label="LoRA Rank" value={form.lora_r} min={1} max={64} step={1} onChange={v => set("lora_r", v)} accent="#10b981" />
                    <SliderField label="LoRA Alpha" value={form.lora_alpha} min={1} max={128} step={1} onChange={v => set("lora_alpha", v)} accent="#10b981" />
                </div>
            </div>

            <div className="flex items-center gap-6 mb-6">
                <Toggle label="Dry Run" checked={form.dry_run} onChange={v => set("dry_run", v)} />
                <Toggle label="Use LoRA" checked={form.use_lora} onChange={v => set("use_lora", v)} />
            </div>

            <div className="flex gap-3 mb-6">
                <button onClick={run} disabled={running} className="btn-primary" id="align-run-btn">
                    {running ? "Aligning..." : form.dry_run ? "Dry Run" : "Start Alignment"}
                </button>
            </div>

            {/* Progress Bar */}
            {(running || progress.percent > 0) && (
                <div className="mb-6">
                    <ProgressBar
                        percent={progress.percent}
                        label={progress.epoch > 0 ? `Epoch ${progress.epoch} of ${progress.total_epochs}` : "Initializing..."}
                        accentColor="#10b981"
                    />
                </div>
            )}

            {/* Metrics Chart */}
            {metrics.length > 0 && (
                <div className="mb-6">
                    <MetricsChart data={metrics} accentColor="#10b981" />
                </div>
            )}

            <LogTerminal logs={logs} isRunning={running} />
        </div>
    );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div>
            <label className="block text-sm font-semibold text-white/60 mb-2">{label}</label>
            {children}
        </div>
    );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
    return (
        <label className="flex items-center gap-2.5 cursor-pointer">
            <div onClick={() => onChange(!checked)}
                className={`relative h-5 w-9 rounded-full transition-colors ${checked ? "bg-emerald-600" : "bg-white/10"}`}>
                <div className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-4" : "translate-x-0.5"}`} />
            </div>
            <span className="text-sm text-white/60">{label}</span>
        </label>
    );
}

function SliderField({ label, value, min, max, step, onChange, accent, display }: {
    label: string; value: number; min: number; max: number; step: number;
    onChange: (v: number) => void; accent: string; display?: (v: number) => string;
}) {
    return (
        <div>
            <div className="flex justify-between mb-1.5">
                <label className="text-sm text-white/60">{label}</label>
                <span className="text-sm font-mono font-semibold text-white">{display ? display(value) : value}</span>
            </div>
            <input type="range" min={min} max={max} step={step} value={value}
                onChange={e => onChange(parseFloat(e.target.value))}
                className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                style={{ accentColor: accent }} />
        </div>
    );
}
