import { HeroSection } from "@/components/hero/HeroSection";
import { PipelineBeam } from "@/components/pipeline/PipelineBeam";
import { FeatureBento } from "@/components/features/FeatureBento";
import { Marquee } from "@/components/ui/marquee";
import { TracingBeam } from "@/components/ui/tracing-beam";

const models = [
  { name: "GPT-2", color: "#7c3aed" },
  { name: "LLaMA 3", color: "#06b6d4" },
  { name: "Mistral 7B", color: "#10b981" },
  { name: "Phi-3", color: "#f59e0b" },
  { name: "Falcon 7B", color: "#ef4444" },
  { name: "Gemma 2B", color: "#8b5cf6" },
  { name: "Qwen 2.5", color: "#06b6d4" },
  { name: "DeepSeek", color: "#3b82f6" },
];

function ModelChip({ name, color }: { name: string; color: string }) {
  return (
    <div
      className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 backdrop-blur-sm"
      style={{ boxShadow: `0 0 20px ${color}20` }}
    >
      <span style={{ color }} className="h-2 w-2 rounded-full bg-current" />
      <span className="text-sm font-medium text-white/80 whitespace-nowrap">
        {name}
      </span>
    </div>
  );
}

export default function Home() {
  return (
    <main className="min-h-screen bg-[#07070f] text-white overflow-x-hidden">
      {/* Nav */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 backdrop-blur-md border-b border-white/5 bg-black/20">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center">
            <span className="text-xs font-black text-white">L</span>
          </div>
          <span className="font-black text-lg tracking-tight">LexAlign</span>
        </div>
        <div className="hidden md:flex items-center gap-8 text-sm text-white/60">
          <a href="#pipeline" className="hover:text-white transition-colors">Pipeline</a>
          <a href="#features" className="hover:text-white transition-colors">Features</a>
          <a href="#models" className="hover:text-white transition-colors">Models</a>
        </div>
        <a
          href="/dashboard"
          className="inline-flex h-9 items-center gap-2 rounded-lg bg-gradient-to-r from-violet-600 to-cyan-600 px-4 text-sm font-semibold text-white hover:from-violet-500 hover:to-cyan-500 transition-all shadow-lg shadow-violet-500/20"
        >
          Launch Dashboard
          <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M5 12h14M12 5l7 7-7 7" />
          </svg>
        </a>
      </nav>

      {/* Hero */}
      <HeroSection />

      {/* Pipeline section */}
      <section id="pipeline" className="relative py-24 px-4">
        <div className="max-w-5xl mx-auto">
          {/* Glowing divider */}
          <div className="flex flex-col items-center text-center mb-16">
            <span className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-4">
              The Pipeline
            </span>
            <h2 className="text-4xl sm:text-5xl font-black text-white mb-4">
              Three stages.{" "}
              <span className="bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
                One command.
              </span>
            </h2>
            <p className="text-white/50 max-w-xl text-lg">
              A unified YAML-driven workflow from raw HuggingFace model to a
              preference-aligned LLM.
            </p>
          </div>

          {/* Animated Beam pipeline */}
          <div className="relative rounded-2xl border border-white/10 bg-white/[0.02] backdrop-blur-sm p-8">
            {/* Glow */}
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-violet-500/5 via-cyan-500/5 to-emerald-500/5" />
            <PipelineBeam />

            {/* Stage descriptions */}
            <div className="grid grid-cols-3 gap-6 mt-6">
              {[
                {
                  step: "01",
                  title: "Download",
                  desc: "Pull any model or dataset from HuggingFace Hub via YAML config.",
                  color: "violet",
                },
                {
                  step: "02",
                  title: "Fine-Tune",
                  desc: "LoRA or QLoRA training with configurable rank, alpha, and target modules.",
                  color: "cyan",
                },
                {
                  step: "03",
                  title: "Align",
                  desc: "DPO or GDPO alignment with preference datasets for RLHF-quality results.",
                  color: "emerald",
                },
              ].map(({ step, title, desc, color }) => (
                <div key={step} className="text-center">
                  <div
                    className={`text-xs font-mono font-bold mb-2 text-${color}-400`}
                  >
                    Step {step}
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2">{title}</h3>
                  <p className="text-sm text-white/50">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Code snippet section */}
      <section className="py-16 px-4">
        <div className="max-w-3xl mx-auto">
          <div className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-sm overflow-hidden">
            {/* Terminal header */}
            <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10 bg-white/[0.02]">
              <span className="h-3 w-3 rounded-full bg-red-500/80" />
              <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
              <span className="h-3 w-3 rounded-full bg-green-500/80" />
              <span className="ml-4 text-xs text-white/40 font-mono">
                finetune.yaml
              </span>
            </div>
            <pre className="p-6 text-sm leading-relaxed overflow-x-auto">
              <code className="font-mono">
                {`model:\n  path: "./models/gpt2"\n\ndataset:\n  path: "./data/my-dataset"\n  format: "auto"\n\ntraining:\n  method: "lora"         `}
                <span className="text-emerald-400">{`# or "qlora"`}</span>
                {`\n  lora_r: 16\n  lora_alpha: 32\n  learning_rate: 3e-4\n  num_epochs: 3\n\ndevice: "cuda"`}
              </code>
            </pre>
          </div>
          <div className="mt-4 text-center">
            <p className="text-sm text-white/40 font-mono">
              $ python finetune.py --config config/finetune.yaml
            </p>
          </div>
        </div>
      </section>

      {/* Features section */}
      <section id="features" className="py-24 px-4">
        <TracingBeam className="px-6">
          <div className="max-w-5xl mx-auto">
            <div className="flex flex-col items-center text-center mb-16">
              <span className="text-xs font-semibold uppercase tracking-widest text-violet-400 mb-4">
                Capabilities
              </span>
              <h2 className="text-4xl sm:text-5xl font-black text-white mb-4">
                Everything you need to{" "}
                <span className="bg-gradient-to-r from-violet-400 to-emerald-400 bg-clip-text text-transparent">
                  train + align
                </span>
              </h2>
              <p className="text-white/50 max-w-xl text-lg">
                Modular, YAML-configured, and fully reproducible.
              </p>
            </div>
            <FeatureBento />
          </div>
        </TracingBeam>
      </section>

      {/* Models marquee section */}
      <section id="models" className="py-16">
        <div className="max-w-5xl mx-auto px-4 text-center mb-10">
          <span className="text-xs font-semibold uppercase tracking-widest text-white/40">
            Works with any HuggingFace model
          </span>
        </div>
        <div className="relative overflow-hidden">
          {/* Left/right fade */}
          <div className="pointer-events-none absolute left-0 top-0 bottom-0 w-32 z-10 bg-gradient-to-r from-[#07070f] to-transparent" />
          <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-32 z-10 bg-gradient-to-l from-[#07070f] to-transparent" />
          <Marquee pauseOnHover className="[--duration:30s]">
            {models.map((m) => (
              <ModelChip key={m.name} {...m} />
            ))}
          </Marquee>
          <Marquee reverse pauseOnHover className="[--duration:30s] mt-4">
            {[...models].reverse().map((m) => (
              <ModelChip key={`r-${m.name}`} {...m} />
            ))}
          </Marquee>
        </div>
      </section>

      {/* CTA footer section */}
      <section className="py-32 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <div className="relative rounded-3xl border border-violet-500/20 bg-gradient-to-b from-violet-900/20 to-transparent p-16">
            <div
              className="absolute inset-0 rounded-3xl opacity-50"
              style={{
                background:
                  "radial-gradient(ellipse 60% 60% at 50% 50%, rgba(124,58,237,0.15), transparent)",
              }}
            />
            <div className="relative z-10">
              <h2 className="text-5xl font-black text-white mb-6">
                Start aligning{" "}
                <span className="bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
                  today
                </span>
              </h2>
              <p className="text-white/60 text-lg mb-10">
                One pip install. One YAML file. Full control over your LLM.
              </p>
              <div className="inline-flex items-center gap-4 rounded-xl border border-white/10 bg-black/30 px-6 py-4 font-mono text-sm text-white/80 backdrop-blur-sm">
                <span className="text-violet-400">$</span>
                pip install lexalign
              </div>
              <div className="mt-6">
                <a
                  href="/dashboard"
                  className="inline-flex h-12 items-center gap-2.5 rounded-xl bg-gradient-to-r from-violet-600 to-cyan-600 px-8 text-base font-bold text-white hover:from-violet-500 hover:to-cyan-500 transition-all shadow-xl shadow-violet-500/25 hover:shadow-violet-500/40"
                >
                  Launch Dashboard
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                    <path d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8 px-6">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="h-5 w-5 rounded bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center">
              <span className="text-[9px] font-black text-white">L</span>
            </div>
            <span className="text-sm text-white/40">
              LexAlign — Open Source LLM Pipeline
            </span>
          </div>
          <p className="text-sm text-white/30">
            Built with Next.js + Magic UI + Aceternity UI
          </p>
        </div>
      </footer>
    </main>
  );
}
