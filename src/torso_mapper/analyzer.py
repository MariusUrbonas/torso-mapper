import torch
from torso_mapper.results.result_tracker import ResultTracker
from torso_mapper.results.scan_result import TorsoScanResult

class VertebraeAnalyzer:
    def __init__(self, model, target_vertebrae):
        """
        Initialize the VertebraeAnalyzer.

        Args:
            model (torch.nn.Module): The trained model for vertebrae detection.
            target_vertebrae (list): List of target vertebrae labels to look for.
        """
        self.model = model
        self.target_vertebrae = list(map(lambda x: x.upper(), target_vertebrae))
        self.result_tracker = ResultTracker()
        assert all([v in TorsoScanResult.VERTIBRAE_LABELS for v in self.target_vertebrae]), f"Unsupported vertibrae label found, please use only labels from a list {TorsoScanResult.VERTIBRAE_LABELS}"

    def analyze_dataloader(self, dataloader, device='cpu'):
        """
        Analyze CT scans from a dataloader to detect specified vertebrae.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing CT scan blocks.
            device (str): Device to run the model on ('cuda' or 'cpu').

        Returns:
        - dict: A dictionary containing analysis results for each scan.
        """
        self.model.eval()
        self.model.to(device)

        with torch.no_grad():
            for batch, scan_ids in dataloader:
                batch = batch.float().to(device)
                outputs, _ = self.model(batch)
                self.result_tracker.update(outputs, scan_ids)

        return self._process_results()

    def _process_results(self):
        """
        Process the results to determine which scans contain the target vertebrae.

        Returns:
            dict: A dictionary containing analysis results for each scan.
        """
        results = {}
        for scan_result in self.result_tracker.get_scan_result_list():
            scan_id = scan_result.id
            vertebrae_present = self._check_vertebrae_presence(scan_result)
            results[scan_id] = {
                'contains_target_vertebrae': all([vertebrae_present[key]["present"] for key in vertebrae_present]),
                'vertebrae_details': vertebrae_present
            }
        return results

    def _check_vertebrae_presence(self, scan_result: TorsoScanResult):
        """
        Check the presence of target vertebrae in a single scan result.

        Args:
            scan_result (TorsoScanResult): The result for a single CT scan.

        Returns:
            dict: A dictionary indicating the presence of each target vertebra.
        """
        vertebrae_presence = {}
        robust_labels = scan_result.get_scan_labels()
        probas = scan_result.get_scan_label_proba()

        for vertebra in self.target_vertebrae:
            index = scan_result.VERTIBRAE_LABELS.index(vertebra)
            if vertebra in robust_labels:
                vertebrae_presence[vertebra] = {"present": True, "probability": probas[index].item()}
            else:
                vertebrae_presence[vertebra] = {"present": False, "probability": probas[index].item()}

        return vertebrae_presence

def analyze_vertebrae(dataloader, model, target_vertebrae, device='cpu'):
    """
    Analyze vertebrae in CT scans using a dataloader and a trained model.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader containing CT scan blocks.
        model (torch.nn.Module): The trained model for vertebrae detection.
        target_vertebrae (list): List of target vertebrae labels to look for.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        dict: A dictionary containing analysis results for each scan.
    """
    analyzer = VertebraeAnalyzer(model, target_vertebrae)
    return analyzer.analyze_dataloader(dataloader, device)