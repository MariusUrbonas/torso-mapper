import torch
from huggingface_hub import PyTorchModelHubMixin

from torso_mapper.model import ResnetBlock


class TorsoNet(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        in_ch: int = 1,
        output_cls: int = 28,
        feature_chs: tuple = (64,),
        kernel_sizes: tuple = (3,),
    ):
        """
        A 3D convolutional neural network.

        Args:
        - in_ch (int): Number of input channels. Default is 1.
        - output_cls (int): Number of output classes. Default is 28.
        - feature_chs (tuple): Tuple of feature channels. Default is (64,).
        """
        super(TorsoNet, self).__init__()

        network_components = []
        in_ch = in_ch
        for feature_ch, kernel_size in zip(feature_chs[:-1], kernel_sizes[:-1]):
            network_components.extend(
                [
                    ResnetBlock(
                        in_ch=in_ch, out_ch=feature_ch, kernel_size=kernel_size
                    ),
                    torch.nn.MaxPool3d(kernel_size=2, stride=2),
                ]
            )
            in_ch = feature_ch

        network_components.extend(
            [
                ResnetBlock(
                    in_ch=in_ch, out_ch=feature_chs[-1], kernel_size=kernel_sizes[-1]
                ),
                torch.nn.SiLU(),
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                torch.nn.Flatten(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(feature_chs[-1], feature_chs[-1] * 4),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(feature_chs[-1] * 4, 128),
            ]
        )
        self.clasification_head = torch.nn.Sequential(
            *[torch.nn.SiLU(), torch.nn.Linear(128, output_cls)]
        )

        self.network = torch.nn.Sequential(*network_components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        enc = self.network(x)
        out = self.clasification_head(enc)
        return out, enc
