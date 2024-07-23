import torch
import numpy as np

class TorsoScanResult:

    VERTIBRAE_LABELS = [
            'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
            'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
            'L1', 'L2', 'L3', 'L4', 'L5'
        ]

    def __init__(self):
        self.scan_logits = None

    def update(self, logits):
        """
        Update the scan result with additional block logits.
        
        Params:
            logits: Tensor of shape (batch_size, num_classes)
        """
        if self.scan_logits is None:
            self.scan_logits = logits.detach().cpu().unsqueeze(0)
        else:
            self.scan_logits = torch.cat([self.scan_logits, logits.detach().cpu().unsqueeze(0)], dim=0)

    def get_scan_labels(self) -> np.ndarray:
        return (torch.sigmoid(self.scan_logits) > 0.5).int().numpy()

    def get_scan_label_sum(self) -> np.ndarray:
        return self.get_scan_labels().sum(axis=0)