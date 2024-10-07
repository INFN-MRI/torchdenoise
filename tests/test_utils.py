import pytest
import torch
from torchdenoise.utils.utils import check_batch_dimension

@pytest.fixture
def valid_input():
    return torch.rand((4, 1, 64, 64))  # Batch size 4

@pytest.fixture
def invalid_input():
    return torch.rand((64, 64))  # No batch dimension

def test_check_batch_dimension_valid(valid_input):
    try:
        check_batch_dimension(valid_input)
    except ValueError:
        pytest.fail("check_batch_dimension() raised ValueError unexpectedly!")

def test_check_batch_dimension_invalid(invalid_input):
    with pytest.raises(ValueError):
        check_batch_dimension(invalid_input)
