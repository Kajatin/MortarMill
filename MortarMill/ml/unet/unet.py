import numpy as np
import torch
from torchvision import transforms

from .unet_parts import *
from ..dataset import BrickDataset


class UNet(nn.Module):
    """ Implements the UNet deep learning model described in U-Net: Convolutional
    Networks for Biomedical Image Segmentation (https://arxiv.org/abs/1505.04597)
    by Olaf Ronneberger, Philipp Fischer, and Thomas Brox (with some modifications).

    Parameters
    ----------
    n_channels: int (default: 3)
        The number of channels in the input data (e.g. for RGB images it is 3).
    n_classes: int (default: 1)
        The number of classes in the output. Currently only supports binary
        classification, which requires this parameter to be 1.

    Attributes
    ----------
    n_channels: int
        See parameters.
    n_classes: int
        See parameters.
    
    Methods
    -------
    forward(x)
        Runs the input through the model. Returns the output logits with shape
        [batch_size, n_classes, height, width].
    """

    def __init__(self, n_channels=4, n_classes=1):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # model components
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """ Runs the input argument `x` through the model.

        Parameters
        ----------
        x: Tensor
            4D input tensor with shape [batch_size, n_channels, height, width].

        Returns
        -------
        logits: Tensor
            The output of the model with shape [batch_size, n_classes, height, width].
        """

        # contracting path, need to save the outputs since these will be
        # concatenated later
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # expanding path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # final 1x1 convolution to get the correct output shape
        logits = self.outc(x)

        return logits

    def predict(self, frame, depth, device=None, scale=1, out_threshold=0.5):
        """ Runs the input argument `frame` through the model doing inference.

        Parameters
        ----------
        frame: array
            The array containing the colour image data. The input is interpreted
            as an RGB 3-channel image.
        device: torch.device (default: None)
            Indicates whether the inference is done on CPU or GPU. If None, inference
            will be done on GPU if available, otherwise on CPU.
        scale: float (default: 0.5)
            Scale factor of resizing input frame before inference.
        out_threshold: float (default: 0.5)
            Threshold value to generate a binary mask from the sigmoid output.

        Returns
        -------
        mask: array
            The final mask image array (1 channel, binary). 0 represents brick
            pixels, 255 for the mortar.
        """

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eval()

        img = torch.from_numpy(BrickDataset.preprocess(frame, depth, scale))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = self(img)
            probs = torch.sigmoid(output)
            probs = probs.squeeze(0)

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(frame.size[1]),
                    transforms.ToTensor()
                ]
            )

            probs = tf(probs.cpu())
            full_mask = probs.squeeze().cpu().numpy()

        return ((full_mask > out_threshold) * 255).astype(np.uint8)