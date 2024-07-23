import torch
import numpy as np

from torso_mapper.results.scan_result import TorsoScanResult

class ResultTracker:
    def __init__(self):
        self.result_map = {}

    def update(self, logits, vol_indices):
        """
        Update the result tracker with a new batch of results.

        Params:
            logits: Tensor of shape (batch_size, num_classes)
            vol_indices: Tensor of shape (batch_size,) indicating which volume each block belongs to
        """
        # Convert logits to log probabilities
        logits = logits.detach().cpu()

        # Update log probability sums and block counts for each volume
        for block_log_prob, vol_idx in zip(logits, vol_indices.cpu().numpy()):
            if vol_idx not in self.result_map:
                self.result_map[vol_idx] = TorsoScanResult()
            self.result_map[vol_idx].update(block_log_prob)

    def get_scan_result(self, volume_idx: int):
        return self.result_map[volume_idx]
    
    def get_scan_result_list(self):
        return list(self.result_map.values())

    @property
    def num_results(self):
        """
        Return the number of volumes currently being tracked.
        """
        return len(self.log_prob_sums)
