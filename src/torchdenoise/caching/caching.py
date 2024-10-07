def cache_weights(model):
    """
    Cache the model's weights.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model whose weights need to be cached.

    Returns
    -------
    dict
        Cached weights as a state_dict.
    """
    return model.state_dict()

def load_cached_weights(model, cached_weights):
    """
    Load cached weights into a model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to load weights into.
    cached_weights : dict
        Cached state_dict of model weights.
    """
    if cached_weights is not None:
        model.load_state_dict(cached_weights)
    else:
        print("No cached weights available.")
