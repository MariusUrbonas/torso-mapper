from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from torso_mapper.data.auto_trim import auto_trim_ct_scan


class CTScanIterableDataset(IterableDataset):
    """
    Iterable dataset for CT scan volumes.

    Args:
        volumes (list): List of CT scan volumes.
        stride (int): Stride value for iterating through the volumes. Default is 32.
    """

    def __init__(self, volumes: Iterable[np.array], stride: int = 32):
        self.volumes = volumes
        self.stride = stride

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            num_workers = 1
            worker_id = 0
        else:  # multi-process data loading
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        for vol_idx, volume in enumerate(self.volumes):
            if vol_idx % num_workers != worker_id:
                continue

            volume = auto_trim_ct_scan(volume)
            depth, height, width = volume.shape

            # Start 64 - stride slices above the volume
            start_offset = 0

            for start in range(start_offset, depth + start_offset, self.stride):
                block = np.zeros((64, 64, 64), dtype=volume.dtype)

                # Calculate how much of the block should be filled with actual data
                vol_start = max(0, start)

                # Fill the block with available data from the volume
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

                yield block, vol_idx


def create_ct_dataloader(
    volumes: Iterable[np.array],
    batch_size: int = 4,
    stride: int = 32,
    num_workers: int = 0,
):
    """
    Create a data loader for CT scan volumes.

    Args:
        volumes (list): A list of CT scan volumes.
        batch_size (int, optional): The number of samples per batch. Defaults to 4.
        stride (int, optional): The stride used for sampling the volumes. Defaults to 32.
        num_workers (int, optional): The number of worker threads to use for data loading. Defaults to 0.

    Returns:
        DataLoader: A data loader object that can be used for iterating over the CT scan volumes.
    """
    dataset = CTScanIterableDataset(volumes, stride)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
