# align.py
#!/usr/bin/env python3
"""LexAlign - Align models using DPO or GDPO."""

import click
from pathlib import Path
from rich.console import Console

from lexalign.config.align_parser import AlignConfigParser, ConfigError
from lexalign.utils.device import DeviceManager
from lexalign.aligner.dataset_prep import PreferenceDataset, DatasetError

console = Console()


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Alignment config file")
@click.option("--resume", "resume_path", default=None, type=click.Path(), help="Resume from checkpoint")
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

        # Show training config
        console.print(f"[cyan]Training method:[/cyan] {ft_config['alignment']['method']}")
        console.print(f"[cyan]Device:[/cyan] {device}")
        console.print(f"[cyan]Learning rate:[/cyan] {ft_config['alignment']['learning_rate']}")
        console.print(f"[cyan]Batch size:[/cyan] {ft_config['alignment']['batch_size']}")

        console.print("[green]Alignment configuration validated successfully![/green]")
        console.print("[yellow]Note: Training loop implementation in next tasks.[/yellow]")

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
