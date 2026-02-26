"use client";

import { cn } from "@/lib/utils";

interface MetricPoint {
    epoch: number;
    loss: number;
    lr: number;
}

interface MetricsChartProps {
    data: MetricPoint[];
    accentColor?: string;
    className?: string;
}

export function MetricsChart({ data, accentColor = "#06b6d4", className }: MetricsChartProps) {
    if (data.length === 0) {
        return (
            <div className={cn(
                "rounded-xl border border-white/10 bg-white/[0.02] p-6 text-center",
                className
            )}>
                <p className="text-sm text-white/20 italic">No training metrics yet.</p>
            </div>
        );
    }

    const width = 400;
    const height = 160;
    const padding = { top: 20, right: 50, bottom: 30, left: 50 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Loss chart
    const losses = data.map(d => d.loss);
    const lrs = data.map(d => d.lr);
    const epochs = data.map(d => d.epoch);

    const minLoss = Math.min(...losses);
    const maxLoss = Math.max(...losses);
    const lossRange = maxLoss - minLoss || 1;

    const minLr = Math.min(...lrs);
    const maxLr = Math.max(...lrs);
    const lrRange = maxLr - minLr || 1;

    const minEpoch = Math.min(...epochs);
    const maxEpoch = Math.max(...epochs);
    const epochRange = maxEpoch - minEpoch || 1;

    function scaleX(epoch: number): number {
        return padding.left + ((epoch - minEpoch) / epochRange) * chartW;
    }

    function scaleLoss(loss: number): number {
        return padding.top + (1 - (loss - minLoss) / lossRange) * chartH;
    }

    function scaleLr(lr: number): number {
        return padding.top + (1 - (lr - minLr) / lrRange) * chartH;
    }

    // Build SVG path for loss
    const lossPath = data
        .map((d, i) => `${i === 0 ? "M" : "L"} ${scaleX(d.epoch)} ${scaleLoss(d.loss)}`)
        .join(" ");

    // Build SVG path for lr
    const lrPath = data
        .map((d, i) => `${i === 0 ? "M" : "L"} ${scaleX(d.epoch)} ${scaleLr(d.lr)}`)
        .join(" ");

    // Area fill for loss
    const lossArea = lossPath +
        ` L ${scaleX(data[data.length - 1].epoch)} ${padding.top + chartH}` +
        ` L ${scaleX(data[0].epoch)} ${padding.top + chartH} Z`;

    const lastMetric = data[data.length - 1];

    return (
        <div className={cn(
            "rounded-xl border border-white/10 bg-white/[0.02] backdrop-blur-sm overflow-hidden",
            className
        )}>
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
                <span className="text-xs font-semibold text-white/40 uppercase tracking-widest">
                    Training Metrics
                </span>
                <div className="flex items-center gap-4 text-xs">
                    <span className="flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-full" style={{ background: accentColor }} />
                        <span className="text-white/50">Loss: <span className="text-white font-mono">{lastMetric.loss.toFixed(4)}</span></span>
                    </span>
                    <span className="flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-full bg-amber-400" />
                        <span className="text-white/50">LR: <span className="text-white font-mono">{lastMetric.lr.toExponential(1)}</span></span>
                    </span>
                </div>
            </div>

            {/* Chart */}
            <div className="p-4">
                <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
                    {/* Grid lines */}
                    {[0, 0.25, 0.5, 0.75, 1].map(frac => {
                        const y = padding.top + frac * chartH;
                        return (
                            <line key={frac} x1={padding.left} y1={y} x2={padding.left + chartW} y2={y}
                                stroke="rgba(255,255,255,0.05)" strokeWidth={0.5} />
                        );
                    })}

                    {/* Loss area fill */}
                    <path d={lossArea} fill={`${accentColor}15`} />

                    {/* Loss line */}
                    <path d={lossPath} fill="none" stroke={accentColor} strokeWidth={2}
                        strokeLinecap="round" strokeLinejoin="round" />

                    {/* LR line */}
                    <path d={lrPath} fill="none" stroke="#f59e0b" strokeWidth={1.5}
                        strokeDasharray="4 3" strokeLinecap="round" />

                    {/* Loss dots */}
                    {data.map((d, i) => (
                        <circle key={`loss-${i}`} cx={scaleX(d.epoch)} cy={scaleLoss(d.loss)}
                            r={3} fill={accentColor} stroke="#07070f" strokeWidth={1.5} />
                    ))}

                    {/* X-axis labels */}
                    {data.map((d, i) => (
                        <text key={`x-${i}`} x={scaleX(d.epoch)} y={height - 5}
                            textAnchor="middle" fill="rgba(255,255,255,0.3)"
                            fontSize={10} fontFamily="monospace">
                            {d.epoch}
                        </text>
                    ))}

                    {/* Y-axis labels (loss) */}
                    <text x={padding.left - 8} y={padding.top + 4} textAnchor="end"
                        fill="rgba(255,255,255,0.3)" fontSize={9} fontFamily="monospace">
                        {maxLoss.toFixed(2)}
                    </text>
                    <text x={padding.left - 8} y={padding.top + chartH + 4} textAnchor="end"
                        fill="rgba(255,255,255,0.3)" fontSize={9} fontFamily="monospace">
                        {minLoss.toFixed(2)}
                    </text>

                    {/* Axis label */}
                    <text x={padding.left + chartW / 2} y={height - 0} textAnchor="middle"
                        fill="rgba(255,255,255,0.2)" fontSize={9}>
                        Epoch
                    </text>
                </svg>
            </div>
        </div>
    );
}
