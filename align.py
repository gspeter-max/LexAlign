#!/usr/bin/env python3
# align.py
"""LexAlign - Align models using DPO or GDPO."""

import click
from pathlib import Path
from rich.console import Console

from lexalign.config.align_parser import AlignConfigParser, ConfigError
from lexalign.utils.device import DeviceManager
from lexalign.aligner.dataset_prep import PreferenceDataset, DatasetError
from lexalign.finetuner.checkpoint import CheckpointManager
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

console = Console()


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Alignment config file")
@click.option("--resume", "resume_path", default=None, type=click.Path(exists=True), help="Resume from checkpoint")
@click.option("--device", "device_override", default=None, type=click.Choice(["cuda", "cpu"]), help="Override device")
@click.option("--dry-run", is_flag=True, help="Show config without training")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def align(config_path: str, resume_path: str, device_override: str, dry_run: bool, verbose: bool):
    """
    Align a fine-tuned model using DPO or GDPO.

    Example:
        python align.py --config config/align.yaml
    """
    try:
        # Parse config
        console.print("[cyan]Loading configuration...[/cyan]")
        with open(config_path) as f:
            yaml_content = f.read()

        parser = AlignConfigParser()
        ft_config = parser.parse(yaml_content)

        # Device management
        device_manager = DeviceManager()
        device, fell_back = device_manager.get_device(
            device_override or ft_config.get("device", "cuda")
        )
        ft_config["device"] = device

        if fell_back:
            console.print("[yellow]Warning: CUDA requested but unavailable. Using CPU.[/yellow]")

        # Dry run - just show config
        if dry_run:
            console.print("[green]Configuration (dry run):[/green]")
            console.print(f"  Model: {ft_config['model']['path']}")
            console.print(f"  Dataset: {ft_config['dataset']['path']}")
            console.print(f"  Method: {ft_config['alignment']['method']}")
            console.print(f"  Device: {device}")
            console.print("[green]Dry run complete. No training performed.[/green]")
            return

        # Validate model exists
        model_path = Path(ft_config["model"]["path"])
        if not model_path.exists():
            console.print(f"[red]Error: Model not found at {ft_config['model']['path']}[/red]")
            console.print("[yellow]→ Run: python finetune.py --config config/finetune.yaml[/yellow]")
            raise click.Abort()

        # Check if resuming from checkpoint
        if resume_path:
            console.print(f"[cyan]Resuming from checkpoint: {resume_path}[/cyan]")
            checkpoint_manager = CheckpointManager()
            resume_step = checkpoint_manager.get_step(resume_path)
            console.print(f"[cyan]Checkpoint step: {resume_step}[/cyan]")
            ft_config["alignment"]["resume_from_checkpoint"] = resume_path

        # Validate dataset exists
        dataset_path = Path(ft_config["dataset"]["path"])
        if not dataset_path.exists():
            console.print(f"[red]Error: Dataset not found at {ft_config['dataset']['path']}[/red]")
            raise click.Abort()

        # Load and validate dataset
        console.print("[cyan]Loading preference dataset...[/cyan]")
        dataset_prep = PreferenceDataset()
        train_dataset = dataset_prep.load_and_validate(ft_config["dataset"])
        console.print(f"[green]Loaded {len(train_dataset)} preference pairs[/green]")

        # Prepare dataset for DPO format
        prompt_field = ft_config["dataset"]["prompt_field"]
        chosen_field = ft_config["dataset"]["chosen_field"]
        rejected_field = ft_config["dataset"]["rejected_field"]

        # Load model and tokenizer
        console.print("[cyan]Loading model...[/cyan]")
        model_path = ft_config["model"]["path"]
        base_model = ft_config["model"].get("base_model", model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            base_model if Path(base_model).exists() else model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Create reference model (frozen copy)
        console.print("[cyan]Creating reference model...[/cyan]")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        for param in ref_model.parameters():
            param.requires_grad = False

        # Apply LoRA if requested
        if ft_config["alignment"].get("use_lora", True):
            console.print("[cyan]Applying LoRA...[/cyan]")
            lora_config = LoraConfig(
                r=ft_config["alignment"].get("lora_r", 16),
                lora_alpha=ft_config["alignment"].get("lora_alpha", 32),
                lora_dropout=ft_config["alignment"].get("lora_dropout", 0.05),
                target_modules=ft_config["alignment"].get("target_modules", None),
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Move to device
        console.print(f"[cyan]Moving models to {device}...[/cyan]")
        model.to(device)
        ref_model.to(device)

        # Show training config
        console.print(f"[cyan]Training method:[/cyan] {ft_config['alignment']['method']}")
        console.print(f"[cyan]Device:[/cyan] {device}")
        console.print(f"[cyan]Learning rate:[/cyan] {ft_config['alignment']['learning_rate']}")
        console.print(f"[cyan]Batch size:[/cyan] {ft_config['alignment']['batch_size']}")
        console.print(f"[cyan]Output dir:[/cyan] {ft_config['alignment']['output_dir']}")

        # Initialize trainer based on method
        console.print("[cyan]Initializing trainer...[/cyan]")
        if ft_config["alignment"]["method"] == "dpo":
            from lexalign.aligner.dpo_trainer import DPOTrainerWrapper
            trainer_wrapper = DPOTrainerWrapper(model, ref_model, tokenizer, ft_config["alignment"])
        else:  # gdpo
            from lexalign.aligner.gdpo_trainer import GDPOTrainerWrapper
            trainer_wrapper = GDPOTrainerWrapper(model, ref_model, tokenizer, ft_config["alignment"])

        # Train
        console.print("[cyan]Starting alignment training...[/cyan]")
        console.print("[green]" + "="*50 + "[/green]")
        train_result = trainer_wrapper.train(train_dataset)
        console.print("[green]" + "="*50 + "[/green]")
        console.print("[green]Training completed![/green]")

        # Save model
        output_dir = ft_config["alignment"]["output_dir"]
        console.print(f"[cyan]Saving model to {output_dir}...[/cyan]")
        trainer_wrapper.save_model(output_dir)
        console.print(f"[green]Model saved successfully![/green]")

    except ConfigError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise click.Abort()
    except DatasetError as e:
        console.print(f"[red]Dataset error: {e}[/red]")
        raise click.Abort()
    except KeyboardInterrupt:
        console.print("\n[yellow]Alignment interrupted by user.[/yellow]")
        raise click.Abort()


if __name__ == "__main__":
    align()
