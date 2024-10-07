import pytest
import torch
from torchdenoise.caching.caching import cache_weights, load_cached_weights
from torchdenoise.deep_learning.cnn_based_denoiser import CNNBasedDenoiser


@pytest.fixture
def cnn_model():
    return CNNBasedDenoiser()


def test_cache_weights(cnn_model):
    weights = cache_weights(cnn_model)
    assert isinstance(weights, dict)
    assert len(weights) > 0


def test_load_cached_weights(cnn_model):
    cached_weights = cache_weights(cnn_model)
    new_model = CNNBasedDenoiser()
    load_cached_weights(new_model, cached_weights)
    for p1, p2 in zip(cnn_model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
