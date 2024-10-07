import pytest
import torch
from torchdenoise.classical.wavelet import WaveletDenoiser
from torchdenoise.classical.total_variation import TotalVariationDenoiser
from torchdenoise.classical.tgv import TGVDenoiser

@pytest.fixture
def sample_input_2d():
    return torch.rand((1, 1, 64, 64))  # 2D image with batch size

def test_wavelet_denoiser(sample_input_2d):
    model = WaveletDenoiser()
    output = model(sample_input_2d)
    assert output.shape == sample_input_2d.shape

def test_total_variation_denoiser(sample_input_2d):
    model = TotalVariationDenoiser()
    output = model(sample_input_2d)
    assert output.shape == sample_input_2d.shape

def test_tgv_denoiser(sample_input_2d):
    model = TGVDenoiser()
    output = model(sample_input_2d)
    assert output.shape == sample_input_2d.shape
