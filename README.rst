torchdenoise
============

**torchdenoise** is a PyTorch-based library for denoising MRI data with a variety of methods, ranging from classical approaches to deep learning-based solutions. It supports both spatial (2D/3D) and space-time (2D+t/3D+t) MRI denoising, and is built with flexibility in mind, allowing for a batch dimension to handle different modalities such as coils or slices.

This library includes:

- **Classical Denoisers**: Wavelet, Total Variation, Total Generalized Variation, Local Low Rank, and High-Order Low Rank.
- **Dictionary Learning**: K-SVD and OMP-based denoisers.
- **Deep Learning Methods**: CNN-based denoisers, with support for caching trainable model weights.
- **Caching Mechanism**: Cache weights for trainable denoisers.
- **Batch Dimension**: Supports handling batches in spatial or temporal dimensions.

Features
--------

- Denoisers for various MRI modalities (spatial, contrast, temporal).
- Easy-to-use interface for both classical and trainable denoisers.
- Flexible batch dimension for coils, slices, or time points.
- Built on PyTorch for easy integration with existing deep learning workflows.
- Caching mechanism for saving and reloading model weights.

Installation
------------

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/yourusername/torchdenoise.git

2. Install the required dependencies:

.. code-block:: bash

    cd torchdenoise
    pip install -r requirements.txt

Usage
-----

**Classical Denoisers**:

Classical denoisers are implemented to work on both 2D and 3D images, as well as their time or contrast dimensions.

Example usage of the Wavelet Denoiser:

.. code-block:: python

    import torch
    from torchdenoise.classical.wavelet import WaveletDenoiser

    # Example 2D input (single batch, single channel, 64x64 image)
    input_tensor = torch.rand((1, 1, 64, 64))
    wavelet_denoiser = WaveletDenoiser()
    output = wavelet_denoiser(input_tensor)

    print(output.shape)  # Should match input shape

**Deep Learning Denoisers**:

Trainable models like CNN-based denoisers can be initialized, trained, and used for inference. Here’s a basic example with the CNN-based denoiser:

.. code-block:: python

    import torch
    from torchdenoise.deep_learning.cnn_based_denoiser import CNNBasedDenoiser

    # Example 3D input (single batch, single channel, 32x32x32 volume)
    input_tensor = torch.rand((1, 1, 32, 32, 32))
    model = CNNBasedDenoiser()

    # Perform a forward pass
    output = model(input_tensor)
    print(output.shape)  # Should match input shape

    # Training the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Dummy training loop
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, input_tensor)  # Dummy target is input
        loss.backward()
        optimizer.step()

**Caching Mechanism**:

You can cache and reload model weights using the provided caching utilities:

.. code-block:: python

    from torchdenoise.caching import cache_weights, load_cached_weights

    model = CNNBasedDenoiser()
    cached_weights = cache_weights(model)

    new_model = CNNBasedDenoiser()
    load_cached_weights(new_model, cached_weights)

    # Both models should now have the same weights
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)

Testing
-------

We provide a full test suite using `pytest`. To run the tests:

.. code-block:: bash

    pytest tests/

Contributing
------------

Contributions are welcome! Please feel free to submit issues or pull requests for any improvements or new features.

License
-------

This project is licensed under the MIT License.

