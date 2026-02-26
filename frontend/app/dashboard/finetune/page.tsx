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

export default function FinetunePage() {
    const [form, setForm] = useState({
        model_path: "./models/gpt2",
        dataset_path: "./data/my-dataset",
        method: "lora",
        device: "cpu",
        lora_r: 16,
        lora_alpha: 32,
        lora_dropout: 0.05,
        learning_rate: 0.0003,
        batch_size: 4,
        num_epochs: 3,
        max_seq_length: 512,
        quantization_bits: 4,
        output_dir: "./checkpoints/finetuned",
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
            const res = await fetch(`${API}/finetune`, {
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
                <span className="text-xs font-mono text-cyan-400">Step 02</span>
                <h1 className="text-3xl font-black text-white mt-1">Fine-Tune</h1>
                <p className="text-white/50 mt-1">Train your model with LoRA or QLoRA.</p>
            </div>

            <div className="grid grid-cols-2 gap-6 mb-6">
                <FilePicker
                    type="models"
                    value={form.model_path}
                    onChange={v => set("model_path", v)}
                    label="Model Path"
                    placeholder="./models/gpt2"
                    accentColor="#06b6d4"
                />
                <FilePicker
                    type="datasets"
                    value={form.dataset_path}
                    onChange={v => set("dataset_path", v)}
                    label="Dataset Path"
                    placeholder="./data/my-dataset"
                    accentColor="#06b6d4"
                />
                <Field label="Output Dir">
                    <input className="input-field" value={form.output_dir} onChange={e => set("output_dir", e.target.value)} placeholder="./checkpoints/finetuned" />
                </Field>
                <Field label="Method">
                    <div className="flex gap-2">
                        {["lora", "qlora"].map(m => (
                            <button
                                key={m}
                                onClick={() => set("method", m)}
                                className={`flex-1 rounded-lg border py-2 text-sm font-semibold uppercase tracking-wide transition-all ${form.method === m
                                    ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-400"
                                    : "border-white/10 text-white/40 hover:border-white/20 hover:text-white/60"
                                    }`}
                            >
                                {m}
                            </button>
                        ))}
                    </div>
                </Field>
                <Field label="Device">
                    <div className="flex gap-2">
                        {["cpu", "cuda"].map(d => (
                            <button
                                key={d}
                                onClick={() => set("device", d)}
                                className={`flex-1 rounded-lg border py-2 text-sm font-semibold uppercase tracking-wide transition-all ${form.device === d
                                    ? "border-violet-500/50 bg-violet-500/10 text-violet-400"
                                    : "border-white/10 text-white/40 hover:border-white/20"
                                    }`}
                            >
                                {d}
                            </button>
                        ))}
                    </div>
                </Field>
            </div>

            {/* Hyperparameters */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5 mb-6">
                <h3 className="text-sm font-semibold text-white/60 uppercase tracking-widest mb-4">Hyperparameters</h3>
                <div className="grid grid-cols-2 gap-4">
                    <SliderField label="LoRA Rank (r)" value={form.lora_r} min={1} max={128} step={1} onChange={v => set("lora_r", v)} accent="#06b6d4" />
                    <SliderField label="LoRA Alpha" value={form.lora_alpha} min={1} max={256} step={1} onChange={v => set("lora_alpha", v)} accent="#06b6d4" />
                    <SliderField label="Learning Rate" value={form.learning_rate} min={1e-6} max={1e-2} step={1e-6} onChange={v => set("learning_rate", v)} accent="#06b6d4" display={v => v.toExponential(1)} />
                    <SliderField label="Epochs" value={form.num_epochs} min={1} max={20} step={1} onChange={v => set("num_epochs", v)} accent="#06b6d4" />
                    <SliderField label="Batch Size" value={form.batch_size} min={1} max={64} step={1} onChange={v => set("batch_size", v)} accent="#06b6d4" />
                    <SliderField label="Max Seq Length" value={form.max_seq_length} min={64} max={4096} step={64} onChange={v => set("max_seq_length", v)} accent="#06b6d4" />
                    {form.method === "qlora" && (
                        <Field label="Quantization Bits">
                            <div className="flex gap-2">
                                {[4, 8].map(b => (
                                    <button key={b} onClick={() => set("quantization_bits", b)}
                                        className={`flex-1 rounded-lg border py-2 text-sm font-semibold transition-all ${form.quantization_bits === b
                                            ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-400"
                                            : "border-white/10 text-white/40 hover:border-white/20"
                                            }`}>{b}-bit</button>
                                ))}
                            </div>
                        </Field>
                    )}
                </div>
            </div>

            <div className="flex items-center gap-4 mb-6">
                <label className="flex items-center gap-2.5 cursor-pointer">
                    <div onClick={() => set("dry_run", !form.dry_run)}
                        className={`relative h-5 w-9 rounded-full transition-colors ${form.dry_run ? "bg-violet-600" : "bg-white/10"}`}>
                        <div className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${form.dry_run ? "translate-x-4" : "translate-x-0.5"}`} />
                    </div>
                    <span className="text-sm text-white/60">Dry Run</span>
                </label>
            </div>

            <div className="flex gap-3 mb-6">
                <button onClick={run} disabled={running} className="btn-primary" id="finetune-run-btn">
                    {running ? "Training..." : form.dry_run ? "Dry Run" : "Start Training"}
                </button>
            </div>

            {/* Progress Bar */}
            {(running || progress.percent > 0) && (
                <div className="mb-6">
                    <ProgressBar
                        percent={progress.percent}
                        label={progress.epoch > 0 ? `Epoch ${progress.epoch} of ${progress.total_epochs}` : "Initializing..."}
                        accentColor="#06b6d4"
                    />
                </div>
            )}

            {/* Metrics Chart */}
            {metrics.length > 0 && (
                <div className="mb-6">
                    <MetricsChart data={metrics} accentColor="#06b6d4" />
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

function SliderField({
    label, value, min, max, step, onChange, accent, display,
}: {
    label: string; value: number; min: number; max: number; step: number;
    onChange: (v: number) => void; accent: string; display?: (v: number) => string;
}) {
    return (
        <div>
            <div className="flex items-center justify-between mb-1.5">
                <label className="text-sm text-white/60">{label}</label>
                <span className="text-sm font-mono font-semibold text-white">{display ? display(value) : value}</span>
            </div>
            <input
                type="range" min={min} max={max} step={step} value={value}
                onChange={e => onChange(parseFloat(e.target.value))}
                className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                style={{ accentColor: accent }}
            />
        </div>
    );
}
