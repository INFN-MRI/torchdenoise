import pytest
import torch
from torchdenoise.dictionary_learning.dictionary_learning_denoiser import (
    DictionaryLearningDenoiser,
)


@pytest.fixture
def sample_input_2d():
    return torch.rand((1, 1, 64, 64))


def test_dictionary_learning_denoiser(sample_input_2d):
    model = DictionaryLearningDenoiser()
    output = model(sample_input_2d)
    assert output.shape == sample_input_2d.shape
