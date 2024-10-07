import torch.nn as nn

class CNNModel(nn.Module):
    """
    Simple CNN model for denoising tasks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_filters : int
        Number of convolutional filters.

    Methods
    -------
    forward(x)
        Forward pass of the CNN.
    """
    def __init__(self, in_channels=1, out_channels=1, num_filters=32):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(num_filters, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
