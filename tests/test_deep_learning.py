import pytest
import torch
from torchdenoise.deep_learning.cnn_based_denoiser import CNNBasedDenoiser

@pytest.fixture
def sample_input_3d():
    return torch.rand((1, 1, 32, 32, 32))  # 3D volume with batch size

def test_cnn_based_denoiser(sample_input_3d):
    model = CNNBasedDenoiser()
    output = model(sample_input_3d)
    assert output.shape == sample_input_3d.shape
