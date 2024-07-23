from typing import Union
import torch
import numpy as np

from torso_mapper.results.scan_result import TorsoScanResult


class ResultTracker:
    def __init__(self):
        self.result_map = {}

    def update(self, logits, ids):
        """
        Update the result tracker with a new batch of results.

        Params:
            logits: Tensor of shape (batch_size, num_classes)
            vol_indices: Tensor of shape (batch_size,) indicating which volume each block belongs to
        """
        # Convert logits to log probabilities
        logits = logits.detach().cpu()
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Update log probability sums and block counts for each volume
        for block_log_prob, vol_id in zip(logits, ids):
            if vol_id not in self.result_map:
                self.result_map[vol_id] = TorsoScanResult(id=vol_id)
            self.result_map[vol_id].update(block_log_prob)

    def get_scan_result(self, id: Union[str, int]):
        return self.result_map[id]

    def get_scan_result_at(self, idx: int):
        return self.get_scan_result_list()[idx]

    def get_scan_result_list(self):
        return list(self.result_map.values())

    @property
    def num_results(self):
        """
        Return the number of volumes currently being tracked.
        """
        return len(self.log_prob_sums)
