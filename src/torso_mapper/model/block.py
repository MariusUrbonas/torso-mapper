import torch


class Block(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 64, kernel_size: int = 3):
        """
        A basic block in the network.

        Args:
        - in_ch (int): Number of input channels.
        - out_ch (int): Number of output channels. Default is 64.
        - kernel_size (int): Size of the convolutional kernel. Default is 3.
        """
        super(Block, self).__init__()
        self.conv = torch.nn.Conv3d(in_ch, out_ch, kernel_size, padding="same")
        self.norm = torch.nn.BatchNorm3d(out_ch)
        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
