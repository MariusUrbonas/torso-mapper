import pytest
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from torso_mapper.data import (
    FolderCTScanIterableDataset,
    CTScanIterableDataset,
    create_folder_ct_dataloader,
    create_ct_dataloader,
)
from torso_mapper.data.utils import reorient_nifty, respace_nifty
from torso_mapper.data.auto_trim import auto_trim_ct_scan
from torso_mapper.model import TorsoNet, ResnetBlock, Block
from torso_mapper.results import TorsoScanResult, ResultTracker
from torso_mapper.analyzer import VertebraeAnalyzer, analyze_vertebrae
from torso_mapper.cli import filter_scans

# Helper function to create a dummy CT scan
def create_dummy_ct_scan(shape=(64, 64, 64)):
    return np.random.rand(*shape).astype(np.float32)

# Test FolderCTScanIterableDataset
def test_folder_ct_scan_iterable_dataset(tmp_path):
    # Create dummy .nii files
    for i in range(3):
        dummy_scan = create_dummy_ct_scan()
        nib.save(nib.Nifti1Image(dummy_scan, np.eye(4)), tmp_path / f"scan_{i}.nii")

    dataset = FolderCTScanIterableDataset(tmp_path)
    assert len(dataset.file_paths) == 3

    for block, path in dataset:
        assert block.shape == (1, 64, 64, 64)
        assert Path(path).exists()

# Test CTScanIterableDataset
def test_ct_scan_iterable_dataset():
    volumes = [create_dummy_ct_scan() for _ in range(3)]
    dataset = CTScanIterableDataset(volumes)

    for block, idx in dataset:
        assert block.shape == (1, 64, 64, 64)
        assert 0 <= idx < 3

# Test create_folder_ct_dataloader
def test_create_folder_ct_dataloader(tmp_path):
    # Create dummy .nii files
    for i in range(3):
        dummy_scan = create_dummy_ct_scan((256, 256, 256))
        nib.save(nib.Nifti1Image(dummy_scan, np.eye(4)), tmp_path / f"scan_{i}.nii")
    dataloader = create_folder_ct_dataloader(tmp_path, batch_size=2)
    for batch, paths in dataloader:
        assert batch.shape == (2, 1, 64, 64, 64)
        assert len(paths) == 2

# Test create_ct_dataloader
def test_create_ct_dataloader():
    volumes = [create_dummy_ct_scan() for _ in range(3)]
    dataloader = create_ct_dataloader(volumes, batch_size=2)
    for batch, indices in dataloader:
        assert batch.shape == (2, 1, 64, 64, 64)
        assert len(indices) == 2

# Test reorient_nifty and respace_nifty
def test_reorient_and_respace_nifty():
    dummy_scan = create_dummy_ct_scan()
    img = nib.Nifti1Image(dummy_scan, np.eye(4))

    reoriented_img = reorient_nifty(img, axcodes_to=("P", "I", "R"))
    assert reoriented_img.shape == dummy_scan.shape

    respaced_img = respace_nifty(reoriented_img, voxel_spacing=(2, 2, 2))
    assert respaced_img.shape[0] == dummy_scan.shape[0] // 2

# Test auto_trim_ct_scan
def test_auto_trim_ct_scan():
    dummy_scan = create_dummy_ct_scan((128, 128, 128))
    trimmed_scan = auto_trim_ct_scan(dummy_scan)
    assert trimmed_scan.shape == (128, 64, 64)  # Assuming default target_size

# Test TorsoNet
def test_torso_net():
    model = TorsoNet()
    input_tensor = torch.randn(1, 1, 64, 64, 64)
    output, enc = model(input_tensor)
    assert output.shape == (1, 28)  # Assuming 28 output classes
    assert enc.shape == (1, 128)  # Assuming 128 features in the encoding

# Test ResnetBlock
def test_resnet_block():
    block = ResnetBlock(in_ch=1, out_ch=64)
    input_tensor = torch.randn(1, 1, 64, 64, 64)
    output = block(input_tensor)
    assert output.shape == (1, 64, 64, 64, 64)

# Test Block
def test_block():
    block = Block(in_ch=1, out_ch=64)
    input_tensor = torch.randn(1, 1, 64, 64, 64)
    output = block(input_tensor)
    assert output.shape == (1, 64, 64, 64, 64)

# Test TorsoScanResult
def test_torso_scan_result():
    result = TorsoScanResult(id=0)
    logits = torch.randn(28)
    result.update(logits)
    assert result.scan_logits.shape == (1, 28)

    # Test get_scan_label_proba
    proba = result.get_scan_label_proba()
    assert proba.shape == (18,)  # 18 vertebrae labels
    assert np.all((proba >= 0) & (proba <= 1))

    # Test get_scan_label_vector
    label_vector = result.get_scan_label_vector()
    assert label_vector.shape == (18,)
    assert np.all((label_vector == 0) | (label_vector == 1))

    # Test get_scan_labels
    labels = result.get_scan_labels()
    assert isinstance(labels, list)
    assert all(label in TorsoScanResult.VERTIBRAE_LABELS for label in labels)

# Test ResultTracker
def test_result_tracker():
    tracker = ResultTracker()
    logits = torch.randn(2, 28)
    ids = [0, 1]
    tracker.update(logits, ids)

    assert len(tracker.get_scan_result_list()) == 2
    assert tracker.get_scan_result(0).id == 0
    assert tracker.get_scan_result(1).id == 1
    assert tracker.get_scan_result_at(0).id == 0

    # Test that ResultTracker no longer has num_results property
    with pytest.raises(AttributeError):
        _ = tracker.num_results

def test_vertebrae_analyzer():
    # Mock the model and dataloader
    mock_model = MagicMock()
    mock_dataloader = MagicMock()
    
    # Set up mock behavior
    mock_model.return_value = (torch.randn(1, 28), torch.randn(1, 128))
    mock_dataloader.__iter__.return_value = [
        (torch.randn(1, 1, 64, 64, 64), ['scan_1.nii']),
        (torch.randn(1, 1, 64, 64, 64), ['scan_2.nii'])
    ]
    
    analyzer = VertebraeAnalyzer(mock_model, ['T1', 'T2', 'L1'])
    results = analyzer.analyze_dataloader(mock_dataloader)
    
    assert len(results) == 2
    assert 'scan_1.nii' in results
    assert 'scan_2.nii' in results
    assert 'contains_target_vertebrae' in results['scan_1.nii']
    assert 'vertebrae_details' in results['scan_1.nii']


def test_analyze_vertebrae():
    # Mock dependencies
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_dataloader = MagicMock()
    
    # Set up mock behavior
    mock_model.return_value = (torch.randn(1, 28), torch.randn(1, 128))
    mock_dataloader.__iter__.return_value = [
        (torch.randn(1, 1, 64, 64, 64), ['scan_1.nii']),
        (torch.randn(1, 1, 64, 64, 64), ['scan_2.nii'])
    ]
    
    results = analyze_vertebrae(mock_dataloader, mock_model, ['T1', 'T2', 'L1'])
    
    assert len(results) == 2
    assert 'scan_1.nii' in results
    assert 'scan_2.nii' in results


if __name__ == "__main__":
    pytest.main()