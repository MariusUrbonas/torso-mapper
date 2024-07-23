import torch

from .block import Block


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 64, kernel_size: int = 3):
        """
        A residual block in the network.

        Args:
        - in_ch (int): Number of input channels.
        - out_ch (int): Number of output channels. Default is 64.
        - kernel_size (int): Size of the convolutional kernel. Default is 3.
        """
        super(ResnetBlock, self).__init__()
        self.block1 = Block(in_ch, out_ch, kernel_size)
        self.block2 = Block(out_ch, out_ch, kernel_size)
        self.oneXone1conv = torch.nn.Conv3d(
            in_ch, out_ch, kernel_size=1, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        _x = self.block1(x)
        _x = self.block2(_x)
        res_conv = self.oneXone1conv(x)
        return _x + res_conv
