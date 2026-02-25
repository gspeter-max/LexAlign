import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from lexalign.config.finetune_parser import FinetuneConfigParser, ConfigError
from lexalign.finetuner.trainer import FinetuneTrainer, TrainerError
from lexalign.utils.device import DeviceManager, DeviceError

console = Console()


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Fine-tuning config file")
@click.option("--resume", type=click.Path(exists=True), help="Checkpoint directory to resume from")
@click.option("--device", type=click.Choice(["cuda", "cpu"]), help="Override device (cuda/cpu)")
@click.option("--dry-run", is_flag=True, help="Show configuration without training")
@click.option("--verbose", "-v", is_flag=True, help="Detailed training output")
def cli(config: str, resume: str, device: str, dry_run: bool, verbose: bool):
    """
    Fine-tune Hugging Face models using LoRA or QLoRA.
    """
    # Load configuration
    try:
        with open(config, "r") as f:
            yaml_content = f.read()

        parser = FinetuneConfigParser()
        ft_config = parser.parse(yaml_content)
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise click.Abort()

    # Override device if specified
    if device:
        try:
            device_manager = DeviceManager()
            ft_config["device"], fell_back = device_manager.get_device(device)
            if fell_back:
                console.print("[yellow]Warning: CUDA requested but unavailable. Using CPU.[/yellow]")
        except DeviceError as e:
            console.print(f"[red]Device error:[/red] {e}")
            raise click.Abort()

    # Dry run: show config and exit
    if dry_run:
        console.print(Panel.fit(
            f"[bold cyan]Fine-tuning Configuration[/bold cyan]\n\n"
            f"Model: {ft_config['model']['path']}\n"
            f"Dataset: {ft_config['dataset']['path']}\n"
            f"Method: {ft_config['training']['method']}\n"
            f"Device: {ft_config['device']}\n"
            f"Output: {ft_config['training'].get('output_dir', 'auto')}"
        ))
        return

    # Initialize trainer
    try:
        trainer = FinetuneTrainer(ft_config, verbose=verbose)

        if resume:
            output_dir = trainer.resume(resume)
            console.print(f"[green]Training resumed and complete![/green] Checkpoints saved to: {output_dir}")
        else:
            output_dir = trainer.train()
            console.print(f"[green]Fine-tuning complete![/green] Checkpoints saved to: {output_dir}")
    except TrainerError as e:
        console.print(f"[red]Training error:[/red] {e}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise click.Abort()


if __name__ == "__main__":
    cli()
