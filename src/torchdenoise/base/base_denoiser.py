import torch.nn as nn

class BaseDenoiser(nn.Module):
    """
    Abstract base class for all denoisers.

    Parameters
    ----------
    spatial_only : bool, optional
        If True, the denoiser operates only on spatial dimensions.
    """
    def __init__(self, spatial_only=False):
        super(BaseDenoiser, self).__init__()
        self.spatial_only = spatial_only

    def forward(self, x):
        """
        Forward pass to be implemented by subclasses.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be denoised.

        Returns
        -------
        torch.Tensor
            Denoised tensor.
        """
        raise NotImplementedError
