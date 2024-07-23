from typing import Union
import torch
import numpy as np


class TorsoScanResult:

    VERTIBRAE_LABELS = [
        "C",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "T12",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
    ]

    VERTIBRAE_LABELS_TO_INDEX_RANGE = {
        "C": (1, 8),
        "T1": (8, 9),
        "T2": (9, 10),
        "T3": (10, 11),
        "T4": (11, 12),
        "T5": (12, 13),
        "T6": (13, 14),
        "T7": (14, 15),
        "T8": (15, 16),
        "T9": (16, 17),
        "T10": (17, 18),
        "T11": (18, 19),
        "T12": (19, 20),
        "L1": (20, 21),
        "L2": (21, 22),
        "L3": (22, 23),
        "L4": (23, 24),
        "L5": (24, 25),
    }

    # These have been experimentally determined on the validation set for top_k = 2
    TRESHOLDS = np.array(
        [
            0.09618627,
            0.46659589,
            0.16770589,
            0.15182535,
            0.48832837,
            0.45266044,
            0.48427486,
            0.49375504,
            0.46781585,
            0.49439496,
            0.47530571,
            0.49903005,
            0.49336016,
            0.49913517,
            0.50019395,
            0.50179356,
            0.44180939,
            0.71680653,
        ]
    )

    def __init__(self, id: Union[str, int]):
        self.scan_logits = None
        self.id = id

    def update(self, logits):
        """
        Update the scan result with additional block logits.

        Params:
            logits: Tensor of shape (batch_size, num_classes)
        """
        if self.scan_logits is None:
            self.scan_logits = logits.detach().cpu().unsqueeze(0)
        else:
            self.scan_logits = torch.cat(
                [self.scan_logits, logits.detach().cpu().unsqueeze(0)], dim=0
            )

    def get_scan_label_proba(self, top_k=2) -> np.ndarray:
        """
        Get a label vector based on the VERTIBRAE_LABELS_TO_INDEX_RANGE.

        Args:
            top_k (int): Number of top values to consider for each column.
            threshold (float): Sigmoid threshold for considering a label as present.

        Returns:
            np.ndarray: A binary vector where each element corresponds to a vertebra or group of vertebrae.
        """
        top_k = min(top_k, self.scan_logits.shape[0])

        # Calculate the mean of top-k sigmoid values for each column
        sigmoid_values = torch.sigmoid(self.scan_logits)
        top_k_vals = torch.topk(sigmoid_values, k=top_k, dim=0).values

        # Initialize the result vector
        result = np.zeros(len(self.VERTIBRAE_LABELS_TO_INDEX_RANGE))

        for i, (label, (start, end)) in enumerate(
            self.VERTIBRAE_LABELS_TO_INDEX_RANGE.items()
        ):
            result[i] = torch.mean(top_k_vals[:, start:end])
        return result

    def get_scan_label_vector(self, top_k=2, threshold=TRESHOLDS) -> np.ndarray:
        """
        Get a label vector based on the VERTIBRAE_LABELS_TO_INDEX_RANGE.

        Args:
            top_k (int): Number of top values to consider for each column.
            threshold (float): Sigmoid threshold for considering a label as present.

        Returns:
            np.ndarray: A binary vector where each element corresponds to a vertebra or group of vertebrae.
        """
        return (self.get_scan_label_proba(top_k=top_k) >= threshold).astype(int)

    def get_scan_labels(self, top_k=2, threshold=TRESHOLDS) -> list[str]:
        """
        Get a list of detected vertebrae labels.

        Args:
            top_k (int): Number of top values to consider for each column.
            threshold (float): Sigmoid threshold for considering a label as present.

        Returns:
            list[str]: A list of detected vertebrae labels.
        """
        label_vector = self.get_scan_label_vector(top_k, threshold)
        return [
            label
            for label, present in zip(
                self.VERTIBRAE_LABELS_TO_INDEX_RANGE.keys(), label_vector
            )
            if present
        ]
