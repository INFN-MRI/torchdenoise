def check_batch_dimension(x):
    """
    Ensure that the input tensor has a valid batch dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Raises
    ------
    ValueError
        If input does not have a valid batch dimension.
    """
    if len(x.shape) < 3:
        raise ValueError(
            "Input tensor must have at least a spatial and batch dimension."
        )
