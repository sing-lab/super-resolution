"""Utility classes for SRGAN model."""
from typing import Callable, Optional

import torch
from torch import nn, Tensor


class ConvolutionalBlock(nn.Module):
    """A convolutional block, comprising convolutional, batch normalization and activation layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        batch_norm: bool = False,
        activation: Optional[str] = None,
    ) -> None:
        """
        Initialize the ConvolutionalBlock class, used as a main block in the Generator structure.

        Parameters
        ----------
        in_channels: int
            The number of input channels.
        out_channels: int
            The number of output channels.
        kernel_size: int
            The kernel size.
        stride: int
            The stride.
        batch_norm: bool
            Whether to include a batch normalization layer.
        activation: Optional[str]
            The type of activation to use, should be in 'LeakyReLu', 'Prelu', 'Tanh'.

        Raises
        ------
        ValueError
            If the activation function is not in 'tanh', 'prelu'.

        """
        super().__init__()

        # A convolutional layer
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

        self.conv_block = nn.Sequential()
        self.conv_block.append(conv_layer)

        # An optional batch normalization layer
        if batch_norm is True:
            self.conv_block.append(nn.BatchNorm2d(num_features=out_channels))

        # An optional activation layer, if wanted
        if activation is not None:
            if activation.lower() == "prelu":
                self.conv_block.append(nn.PReLU())
            elif activation.lower() == "tanh":
                self.conv_block.append(nn.Tanh())
            elif activation.lower() == "leakyrelu":
                self.conv_block.append(nn.LeakyReLU(negative_slope=0.2))
            else:
                raise ValueError(
                    f"Activation should be either 'leakyrelu', 'prelu', or 'tanh' but is '{activation}'."
                )

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward  propagation.

        Parameters
        ----------
        input: Tensor
            input images, a tensor of size (N, in_channels, w, h)

        Returns
        -------
        Tensor
            Output images, a tensor of size (N, out_channels, w, h)

        """
        return self.conv_block(input)  # (N, out_channels, w, h)


class ResidualBlock(nn.Module):
    """A residual block, comprising two convolutional blocks with a residual connection across them."""

    def __init__(self, kernel_size: int = 3, n_channels: int = 64) -> None:
        """
        Initialize the ResidualBlock class, used as a main block in the Generator structure.

        Parameters
        ----------
        kernel_size: int
            The kernel size.
        n_channels: int
            The number of input and output channels, identical because the input must be added to the output with
            skip-connections.
        """
        super().__init__()

        # The first convolutional block
        self.conv_block1 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation="PReLu",
        )

        # The second convolutional block
        self.conv_block2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            batch_norm=True,
            activation=None,
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward propagation.

        Parameters
        ----------
        input: Tensor
            Input images, a tensor of size (N, n_channels, w, h).

        Returns
        -------
        Tensor
            Output images, a tensor of size (N, n_channels, w, h).
        """
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)

        # Skip-connection
        return output + input


class SubPixelConvolutionalBlock(nn.Module):
    """A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers."""

    def __init__(
        self, kernel_size: int = 3, n_channels: int = 64, scaling_factor: int = 2
    ) -> None:
        """
        Initialize the SubPixelConvolutionalBlock class, used for upscaling image generating pixels.

        Use https://arxiv.org/abs/1707.02937  for better initialize and prevent checkerboard artifacts.

        Parameters
        ----------
        kernel_size: int
            The kernel size.
        n_channels: int
            The number of input and output channels.
        scaling_factor: int
            The factor to scale input images by (along both dimensions).
        """
        super().__init__()

        # A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle
        # and PReLU
        self.conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels * (scaling_factor**2),
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        # These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling
        # factor
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)

        # Initialize weights to avoid artifacts (better than random initialization).
        kernel = ICNR(self.conv.weight, scaling_factor)
        self.conv.weight.data.copy_(kernel)

        self.activation = nn.PReLU()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward propagation.

        Parameters
        ----------
        input: Tensor
            Input images, a tensor of size (N, n_channels, w, h).

        Returns
        -------
        Tensor
            Scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(
            output
        )  # (N, n_channels, w * scaling factor, h * scaling factor)
        return self.activation(
            output
        )  # (N, n_channels, w * scaling factor, h * scaling factor)


def ICNR(
    tensor: Tensor,
    scaling_factor: int = 2,
    initializer: Callable = nn.init.kaiming_normal_,
) -> Tensor:
    """
    Prevent checkerboard artifacts.

    Fills the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution", https://arxiv.org/abs/1707.02937,
    Andrew Aitken et al. (2017), this initialization should be used in the last convolutional layer before a
    PixelShuffle operation.

    Parameters
    ----------
    tensor: Tensor
        An n-dimensional torch.Tensor or autograd.Variable
    scaling_factor: int
        Factor to increase spatial resolution by
    initializer: Callable
        Initializer to be used for sub_kernel initialization.

    Returns
    -------
    Tensor
        Kernel initialization.
    """
    new_shape = [int(tensor.shape[0] / (scaling_factor**2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = initializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0], subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, scaling_factor**2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)

    return kernel
