import click
import torch
from pathlib import Path
from torso_mapper.data.folder_dataset import create_folder_ct_dataloader
from torso_mapper.model.conv_3d_net import TorsoNet
from torso_mapper.analyzer import analyze_vertebrae
import json

@click.command()
@click.option('--input-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True,
              help='Directory containing the CT scan files (.nii or .nii.gz)')
@click.option('--model', required=True, default="marius-urbonas/torso-mapper",
              help='trained model name or file')
@click.option('--vertebrae', '-v', multiple=True, required=True,
              help='Target vertebrae to look for (can be specified multiple times)')
@click.option('--batch-size', type=int, default=4, help='Batch size for processing')
@click.option('--num-workers', type=int, default=4, help='Number of worker threads for data loading')
@click.option('--device', type=click.Choice(['cuda', 'cpu']), default='cpu', help='Device to run the model on')
@click.option('--output', type=click.Path(file_okay=True, dir_okay=False), default='analysis_results.txt',
              help='Output file to save the analysis results')
def filter_scans(input_dir, model, vertebrae, batch_size, num_workers, device, output):
    """Analyze scans for specific vertebrae."""
    click.echo(f"Analyzing scans in {input_dir} for vertebrae: {', '.join(vertebrae)}")

    # Load the model
    model = TorsoNet.from_pretrained(model)
    model.eval()

    # Create the dataloader
    dataloader = create_folder_ct_dataloader(input_dir, batch_size=batch_size, num_workers=num_workers)

    # Perform the analysis
    results = analyze_vertebrae(dataloader, model, list(vertebrae), device)

    # Process and save the results
    with open(output, 'w') as f:
        json.dump(results, f, indent=4)

    click.echo(f"Analysis complete. Results saved to {output}")


def main():
    filter_scans()
    

if __name__ == '__main__':
    main()