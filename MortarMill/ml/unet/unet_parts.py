import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """ Performs 3x3 convolution, dropout, batch normalization, and ReLU twice
        in a row.

    Parameters
    ----------
    in_channels: int
        Number of input channels of the expected input (input.shape[1]).
    out_channels: int
        Number of output channels.
    p: float (default: 0.1)
        Probability of zeroing out channels during training as part of the dropout
        process.

    Attributes
    ----------
    double_conv: torch.nn.Sequential
        Sequential deep learning layer with convolution, dropout, batchnorm, and
        ReLU twice in a row.
    """

    def __init__(self, in_channels, out_channels, p=0.1):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p,inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p,inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """ Runs the input argument `x` through the layer.

        Parameters
        ----------
        x: Tensor
            4D input tensor with shape [batch_size, in_channels, height, width].

        Returns
        -------
        ret: Tensor
            The output of the layer with shape [batch_size, out_channels, height, width].
        """

        return self.double_conv(x)


class Down(nn.Module):
    """ Downscaling with maxpool (2x2) then double convolution.
    
    Parameters
    ----------
    in_channels: int
        Number of input channels of the expected input (input.shape[1]).
    out_channels: int
        Number of output channels.

    Attributes
    ----------
    maxpool_conv: torch.nn.Sequential
        Sequential deep learning layer with max pooling and double convolution.
    """

    def __init__(self, in_channels, out_channels, p=0.1):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2,stride=2),
            DoubleConv(in_channels, out_channels, p)
        )

    def forward(self, x):
        """ Runs the input argument `x` through the layer.

        Parameters
        ----------
        x: Tensor
            4D input tensor with shape [batch_size, in_channels, height, width].

        Returns
        -------
        ret: Tensor
            The output of the layer with shape [batch_size, out_channels, height, width].
        """

        return self.maxpool_conv(x)


class Up(nn.Module):
    """ Upscaling then double convolution.

    Parameters
    ----------
    in_channels: int
        Number of input channels of the expected input (input.shape[1]).
    out_channels: int
        Number of output channels.

    Attributes
    ----------
    up: torch.nn.ConvTranspose2d
        Up-convolution which upscales the input tensor.
    conv: self.DoubleConv
        Double convolutional sequential layer (see self.DoubleConv).
    """

    def __init__(self, in_channels, out_channels, p=0.1):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, p)

    def forward(self, x1, x2):
        """ Upscales the input `x1`, then concatenates it with `x2`, then runs it
            through a double convolution layer.

        Parameters
        ----------
        x1: Tensor
            4D input tensor with shape [batch_size, in_channels, height, width].
        x2: Tensor
            4D input tensor with shape [batch_size, in_channels//2, height, width].

        Returns
        -------
        ret: Tensor
            The output of the layer with shape [batch_size, out_channels, height, width].
        """

        # upscale `x1`
        x1 = self.up(x1)
        
        # pad `x1` if necessary so that it can be concatenated with `x2` (input is BCHW)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # concatenate `x1` and `x2` then run through double convolution layer
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """ Final convolution to map to the final number of channels (classes).

    Parameters
    ----------
    in_channels: int
        Number of input channels of the expected input (input.shape[1]).
    out_channels: int
        Number of output channels.

    Attributes
    ----------
    up: torch.nn.Conv2d
        Simple 1x1 convolution to tranform input tensor to desired shape.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """ Runs the input argument `x` through the layer.

        Parameters
        ----------
        x: Tensor
            4D input tensor with shape [batch_size, in_channels, height, width].

        Returns
        -------
        ret: Tensor
            The output of the layer with shape [batch_size, out_channels, height, width].
        """

        return self.conv(x)