# torso mapper
![Tests](https://github.com/MariusUrbonas/torso-mapper/actions/workflows/ci_cd.yml/badge.svg) [![Coverage Status](https://coveralls.io/repos/github/MariusUrbonas/torso-mapper/badge.svg?branch=main)](https://coveralls.io/github/MariusUrbonas/torso-mapper?branch=main) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/your-package-name.svg)](https://badge.fury.io/py/your-package-name) ![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

A Python package for quickly identifying parts of the torso imaged in CT scans through visible vertebrae detection.

## Description

`torso_mapper` is a powerful tool designed to analyze CT scan volumes and identify visible vertebrae. It uses deep learning to process 3D medical imaging data. This package is particularly useful for medical professionals and researchers working with CT scans of the torso region.

## Features

- Fast and accurate vertebrae detection in CT scans
- Support for various CT scan file formats (.nii, .nii.gz)
- Efficient data loading and preprocessing
- 3D convolutional neural network for vertebrae classification
- Automatic trimming of CT scans to focus on the region of interest
- Visualization tools for CT scan cross-sections

## Installation

You can install `torso_mapper` using pip:

```bash
pip install torso_mapper
```

## CLI Usage

The `torso_mapper` package provides a command-line interface for analyzing CT scans. Here's how to use it:

```bash
torso_mapper filter-scans --input-dir <path_to_scans> --vertebrae <vertebra1> --vertebrae <vertebra2> [OPTIONS]
```

Arguments:
- `--input-dir`: Directory containing the CT scan files (.nii or .nii.gz)
- `--vertebrae`: Target vertebrae to look for (can be specified multiple times)

Options:
- `--model`: Trained model name or file (default: "marius-urbonas/torso-mapper")
- `--batch-size`: Batch size for processing (default: 4)
- `--num-workers`: Number of worker threads for data loading (default: 4)
- `--device`: Device to run the model on ('cuda' or 'cpu', default: 'cpu')
- `--output`: Output file to save the analysis results (default: 'analysis_results.txt')

Example:
```bash
torso_mapper filter-scans --input-dir ./ct_scans --vertebrae T1 --vertebrae T2 --vertebrae L1 --output results.json
```

This command will analyze the CT scans in the `./ct_scans` directory, looking for vertebrae T1, T2, and L1, and save the results to `results.json`.

## Specifiable Vertebrae

The `torso_mapper` can detect the following vertebrae:

- Cervical (C1-C7): Detected as a group due to their proximity
- Thoracic (T1-T12): Individually detectable
- Lumbar (L1-L5): Individually detectable

When specifying vertebrae for detection, you can use the following labels:

- `C`: Represents any cervical vertebra (C1-C7)
- `T1` to `T12`: Individual thoracic vertebrae
- `L1` to `L5`: Individual lumbar vertebrae

Note: For cervical vertebrae (C1-C7), which are located in the neck region, the model detects the general existence of any cervical vertebrae rather than identifying them individually. This is due to their close proximity to each other, which makes individual detection challenging.

Example usage:
```bash
torso_mapper filter-scans --input-dir ./ct_scans --vertebrae C --vertebrae T1 --vertebrae T12 --vertebrae L5
```

This command will look for the presence of any cervical vertebrae, as well as the specific thoracic vertebrae T1 and T12, and the lumbar vertebra L5.

In your Python code, you can specify vertebrae in the same way:

```python
target_vertebrae = ['C', 'T1', 'T12', 'L5']
results = analyze_vertebrae(dataloader, model, target_vertebrae, device='cpu')
```

Remember that when 'C' is specified, the results will indicate the presence or absence of any cervical vertebrae, not individual C1-C7 vertebrae.

## API Usage

Here's a detailed example of how to use the `torso_mapper` API:

```python
from torso_mapper.data import create_folder_ct_dataloader
from torso_mapper.model import TorsoNet
from torso_mapper.analyzer import analyze_vertebrae

# Create a dataloader for your CT scans
dataloader = create_folder_ct_dataloader("path/to/ct_scans", batch_size=4)

# Load the pre-trained model
model = TorsoNet.from_pretrained("marius-urbonas/torso-mapper")

# Specify the vertebrae you want to detect
target_vertebrae = ['T1', 'T2', 'L1']

# Analyze the CT scans
results = analyze_vertebrae(dataloader, model, target_vertebrae, device='cpu')

# Process the results
for scan_id, scan_results in results.items():
    print(f"Scan: {scan_id}")
    print(f"Contains target vertebrae: {scan_results['contains_target_vertebrae']}")
    print("Vertebrae details:")
    for vertebra, details in scan_results['vertebrae_details'].items():
        print(f"  {vertebra}: Present: {details['present']}, Probability: {details['probability']:.2f}")
    print()
```

This script will:
1. Load CT scans from a specified directory
2. Use the pre-trained TorsoNet model
3. Analyze the scans for the presence of T1, T2, and L1 vertebrae
4. Print out detailed results for each scan

You can adjust the `target_vertebrae` list to look for different vertebrae, and modify the processing of results as needed for your specific use case.

## API Usage for getting clasification labels
 
Here's a basic example of how to use `torso_mapper`:

```python
from torso_mapper.data import create_folder_ct_dataloader
from torso_mapper.model import TorsoNet
from torso_mapper.results import ResultTracker

# Load CT scans from a folder
dataloader = create_folder_ct_dataloader("path/to/ct_scans", batch_size=4)

# Initialize the TorsoNet model
from torso_mapper.model import TorsoNet

model = TorsoNet.from_pretrained("marius-urbonas/torso-mapper")
_ = model.eval()

# Initialize the result tracker
tracker = ResultTracker()

# Process CT scans
for batch, ids in dataloader:
    outputs, _ = model(batch)
    tracker.update(outputs, ids)

# Get results for a specific scan
scan_result = tracker.get_scan_result_at(0)
labels = scan_result.get_scan_labels()
```

## Dependencies

- PyTorch
- NumPy
- Nibabel
- Matplotlib

## Contributing

Interested in contributing? We welcome contributions of all forms. Please check out our contributing guidelines for more information on how to get started.

## License

`torso_mapper` is licensed under the MIT License. See the LICENSE file for more details.

## Credits

- Created by Marius Urbonas
- Developed using [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter)

## Contact

For questions, issues, or suggestions, please open an issue on our GitHub repository or contact the maintainer directly.