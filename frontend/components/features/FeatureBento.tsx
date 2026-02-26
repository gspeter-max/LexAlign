"use client";

import React from "react";
import { BentoCard, BentoGrid } from "@/components/ui/bento-grid";
import {
    LayersIcon,
    CubeIcon,
    TargetIcon,
    LightningBoltIcon,
    DownloadIcon,
    CheckCircledIcon,
} from "@radix-ui/react-icons";

const features = [
    {
        Icon: LayersIcon,
        name: "LoRA Fine-Tuning",
        description:
            "Low-Rank Adaptation training with configurable rank, alpha, and dropout. Target any attention layers in your model.",
        href: "#",
        cta: "Learn more",
        className: "col-span-3 lg:col-span-1",
        background: (
            <div className="absolute inset-0 bg-gradient-to-br from-violet-900/30 via-transparent to-transparent" />
        ),
    },
    {
        Icon: CubeIcon,
        name: "QLoRA Quantization",
        description:
            "4-bit and 8-bit quantized LoRA training via bitsandbytes. Train large models on consumer GPUs.",
        href: "#",
        cta: "Learn more",
        className: "col-span-3 lg:col-span-2",
        background: (
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-900/30 via-transparent to-transparent" />
        ),
    },
    {
        Icon: TargetIcon,
        name: "DPO Alignment",
        description:
            "Direct Preference Optimization — align fine-tuned models to human preferences using chosen/rejected pairs.",
        href: "#",
        cta: "Learn more",
        className: "col-span-3 lg:col-span-2",
        background: (
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-900/30 via-transparent to-transparent" />
        ),
    },
    {
        Icon: LightningBoltIcon,
        name: "GDPO Alignment",
        description:
            "Group Delay Policy Optimization — next-gen alignment method with group-level preference modeling.",
        href: "#",
        cta: "Learn more",
        className: "col-span-3 lg:col-span-1",
        background: (
            <div className="absolute inset-0 bg-gradient-to-br from-amber-900/30 via-transparent to-transparent" />
        ),
    },
    {
        Icon: DownloadIcon,
        name: "HuggingFace Integration",
        description:
            "Declarative YAML config to download any model or dataset from the HuggingFace Hub. Supports glob patterns.",
        href: "#",
        cta: "Learn more",
        className: "col-span-3 lg:col-span-1",
        background: (
            <div className="absolute inset-0 bg-gradient-to-br from-pink-900/30 via-transparent to-transparent" />
        ),
    },
    {
        Icon: CheckCircledIcon,
        name: "Smart Checkpointing",
        description:
            "Auto-save checkpoints every N steps. Resume interrupted training from any checkpoint seamlessly.",
        href: "#",
        cta: "Learn more",
        className: "col-span-3 lg:col-span-2",
        background: (
            <div className="absolute inset-0 bg-gradient-to-br from-blue-900/30 via-transparent to-transparent" />
        ),
    },
];

export function FeatureBento() {
    return (
        <BentoGrid className="auto-rows-[14rem]">
            {features.map((feature) => (
                <BentoCard key={feature.name} {...feature} />
            ))}
        </BentoGrid>
    );
}
