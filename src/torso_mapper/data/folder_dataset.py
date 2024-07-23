import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from torso_mapper.data.utils import reorient_nifty, respace_nifty
from torso_mapper.data.auto_trim import auto_trim_ct_scan


class FolderCTScanIterableDataset(IterableDataset):
    """
    Iterable dataset for CT scan volumes loaded from a folder.

    Args:
        folder_path (str): Path to the folder containing .nii files.
        stride (int): Stride value for iterating through the volumes. Default is 32.
    """

    def __init__(self, folder_path: str, stride: int = 32):
        self.folder_path = folder_path
        self.stride = stride
        self.file_paths = self._get_nii_files()

    def _get_nii_files(self) -> List[str]:
        """Find all .nii files in the given folder."""
        return [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]

    def _load_and_process_scan(self, file_path: str) -> np.ndarray:
        """Load and process a single .nii file."""
        scan_nifty = nib.load(file_path)
        scan_nifty = reorient_nifty(scan_nifty, axcodes_to=("I", "P", "R"))
        scan_nifty = respace_nifty(scan_nifty, voxel_spacing=(4, 4, 4), order=3)
        scan_np = scan_nifty.get_fdata()
        scan_np = auto_trim_ct_scan(scan_np)
        return scan_np

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            num_workers = 1
            worker_id = 0
        else:  # multi-process data loading
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # Distribute files among workers
        worker_file_paths = self.file_paths[worker_id::num_workers]

        for path in worker_file_paths:
            volume = self._load_and_process_scan(path)
            depth, height, width = volume.shape
            start_offset = 0
            
            for start in range(start_offset, depth, self.stride):
                block = np.zeros((64, 64, 64), dtype=volume.dtype)

                vol_start = max(0, start)
                if vol_start + 64 > depth:
                    vol_start = max(depth - 64, 0)

                volume_slice = volume[vol_start : vol_start + 64, :, :]
                volume_slice = np.clip(volume_slice, a_min=-1024, a_max=8192)

                if volume_slice.size == 0:
                    continue

                if volume_slice.std() != 0:
                    volume_slice = (
                        volume_slice - volume_slice.mean()
                    ) / volume_slice.std()
                else:
                    volume_slice = volume_slice - volume_slice.mean()

                # Get the actual dimensions of the volume slice
                d, h, w = volume_slice.shape

                # Copy the volume slice into the block, handling potential size mismatches
                block[:d, :h, :w] = volume_slice

                block = np.expand_dims(block, axis=0)

                yield block, path


def create_folder_ct_dataloader(
    folder_path: str,
    batch_size: int = 4,
    stride: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a data loader for CT scan volumes from a folder.

    Args:
        folder_path (str): Path to the folder containing .nii files.
        batch_size (int, optional): The number of samples per batch. Defaults to 4.
        stride (int, optional): The stride used for sampling the volumes. Defaults to 32.
        num_workers (int, optional): The number of worker threads to use for data loading. Defaults to 0.

    Returns:
        DataLoader: A data loader object that can be used for iterating over the CT scan volumes.
    """
    dataset = FolderCTScanIterableDataset(folder_path, stride)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
