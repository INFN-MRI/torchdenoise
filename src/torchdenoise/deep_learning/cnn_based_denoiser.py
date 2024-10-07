import pytorch_lightning as pl
from ..base.base_denoiser import BaseDenoiser
from .model import CNNModel

class CNNBasedDenoiser(pl.LightningModule, BaseDenoiser):
    """
    CNN-based denoiser for 3D/4D data.

    Methods
    -------
    forward(x)
        Applies CNN-based denoising.
    training_step(batch, batch_idx)
        Performs a single training step.
    configure_optimizers()
        Sets up the optimizer.
    """
    def __init__(self):
        super(CNNBasedDenoiser, self).__init__()
        self.model = CNNModel(in_channels=1, out_channels=1, num_filters=32)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
